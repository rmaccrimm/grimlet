use std::collections::{HashMap, VecDeque};
use std::fmt::Display;
use std::ops::{Add, AddAssign};

use anyhow::{Result, bail};
use num::integer::Average;

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalTree<T: Average + Copy + PartialEq + Display> {
    root: Option<usize>,
    nodes: HashMap<usize, Node<T>>,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Node<T: Average + Copy + PartialEq + Display> {
    center: T,
    sorted_by_first: Vec<(T, T)>,
    sorted_by_last: Vec<(T, T)>,
    left: Option<usize>,
    right: Option<usize>,
    balance: BF,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Direction {
    Left = 0,
    Right = 1,
}

// Balance factor
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BF {
    #[default]
    Balanced,
    Heavy(Direction),
    Unbalanced(Direction),
}

impl<T: Average + Copy + PartialEq + Display> IntervalTree<T> {
    /// Insert a new interval. Has no effect if tree already contains the interval.
    pub fn insert(&mut self, ival: (T, T)) {
        // Current node being searched, it's parent and direction to add new node
        let mut cur = self.root;
        let mut par: Option<(usize, Direction)> = None;
        // (Ancestor) root of the subtree for which we'll need to update balance factors, and the
        // path we took from it. Either the last unbalanced node encountered, or root.
        let mut anc = self.root;
        let mut anc_par: Option<(usize, Direction)> = None;
        let mut path: Vec<Direction> = vec![];

        // Search the tree
        while let Some(n) = cur {
            // Only need to go as far back as the last unbalanced node we saw on our way
            if self.node(n).balance != BF::Balanced {
                path.clear();
                anc = cur;
                anc_par = par;
            }
            let dir = if ival.1 < self.node(n).center {
                Direction::Left
            } else if ival.0 > self.node(n).center {
                Direction::Right
            } else {
                self.node(n).add(ival);
                return;
            };
            cur = self.node(n).child(dir);
            path.push(dir);
            par = Some((n, dir));
        }
        // Create new node
        match par {
            Some((p, dir)) => {
                self.nodes.insert(self.nodes.len(), Node::new(ival));
                let c = Some(self.nodes.len() - 1);
                self.node(p).set_child(dir, c);
            }
            None => {
                self.nodes.insert(self.nodes.len(), Node::new(ival));
                self.root = Some(0);
            }
        }
        let Some(a) = anc else { return };
        // Update balance factors
        let mut n = a;
        for dir in path {
            self.node(n).balance += dir;
            n = self.node(n).child(dir).unwrap();
        }
        self.rebalance_insert(a, anc_par);
    }

    /// Return all intervals containing b
    pub fn search(&self, b: T) -> Vec<(T, T)> {
        let mut curr = self.root;
        let mut results = vec![];

        while let Some(n) = curr {
            let node = self.nodes.get(&n).unwrap();

            if b == node.center {
                results.extend_from_slice(&node.sorted_by_first);
                return results;
            } else if b < node.center {
                let i = node.sorted_by_first.partition_point(|&r| r.0 <= b);
                results.extend_from_slice(&node.sorted_by_first[0..i]);
                curr = node.left;
            } else {
                let i = node.sorted_by_last.partition_point(|&r| r.1 >= b);
                results.extend_from_slice(&node.sorted_by_last[0..i]);
                curr = node.right;
            }
        }
        results
    }

    pub fn remove(&mut self, ival: (T, T)) -> Result<()> {
        let mut cur = self.root;
        let mut par: Option<(usize, Direction)> = None;

        while let Some(n) = cur {
            let dir = if ival.1 < self.node(n).center {
                Direction::Left
            } else if ival.0 > self.node(n).center {
                Direction::Right
            } else {
                break;
            };
            cur = self.node(n).child(dir);
            par = Some((n, dir));
        }
        let rm = match cur {
            None => bail!("interval not found"),
            Some(n) => n,
        };
        if !self.node(rm).remove(ival) {
            return Ok(());
        }

        let removed = self.nodes.remove(&rm).unwrap();
        let rep = match removed.left {
            None => removed.right,
            Some(l) => {
                if self.node(l).right.is_none() {
                    self.node(l).right = removed.right;
                    Some(l)
                } else {
                    let (pred, pred_par) = self.subtree_max(l);
                    if self.node(pred).right.is_some() {
                        bail!("invalid predecessor");
                    }
                    self.node(pred_par).right = self.node(pred).left.take();
                    self.node(pred).left = removed.left;
                    self.node(pred).right = removed.right;
                    Some(pred)
                }
            }
        };
        match par {
            Some((p, dir)) => self.node(p).set_child(dir, rep),
            None => self.root = rep,
        }
        Ok(())
    }

    // Takes the following nodes
    // a: root of the sub-tree which needs to be re-balanced
    // parent: parent of a, may be None if node is the root
    fn rebalance_insert(&mut self, a: usize, parent: Option<(usize, Direction)>) {
        if let BF::Unbalanced(dir) = self.node(a).balance {
            let b = self.node(a).child(dir).unwrap();
            if let BF::Heavy(dir_bc) = self.node(b).balance {
                if dir_bc == dir.flip() {
                    let c = self.node(b).child(dir.flip()).unwrap();
                    self.rotate(dir, b, c);
                    self.node(a).set_child(dir, Some(c));
                    self.rotate(dir.flip(), a, c);
                    self.reparent(parent, c);
                    match self.node(c).balance {
                        BF::Heavy(dir_cd) => {
                            if dir_cd == dir {
                                self.node(b).balance = BF::Balanced;
                                self.node(a).balance = BF::Heavy(dir.flip());
                            } else {
                                self.node(b).balance = BF::Heavy(dir);
                                self.node(a).balance = BF::Balanced;
                            }
                        }
                        BF::Balanced => {
                            self.node(b).balance = BF::Balanced;
                            self.node(a).balance = BF::Balanced;
                        }
                        _ => panic!("invalid balance"),
                    }
                    self.node(c).balance = BF::Balanced;
                } else {
                    self.rotate(dir.flip(), a, b);
                    self.node(b).balance = BF::Balanced;
                    self.node(a).balance = BF::Balanced;
                    self.reparent(parent, b);
                }
            } else {
                panic!("invalid balance");
            }
        }
    }

    fn node(&mut self, k: usize) -> &mut Node<T> { self.nodes.get_mut(&k).expect("node not found") }

    fn rotate(&mut self, d: Direction, p: usize, c: usize) {
        if self.node(p).child(d.flip()) != Some(c) {
            panic!("invalid rotation")
        }
        let c_child = self.node(c).child(d);
        self.node(p).set_child(d.flip(), c_child);
        self.node(c).set_child(d, Some(p));
        if self.root == Some(p) {
            self.root = Some(c);
        }
    }

    fn reparent(&mut self, par: Option<(usize, Direction)>, s: usize) {
        match par {
            Some((p, dir)) => {
                self.node(p).set_child(dir, Some(s));
            }
            None => self.root = Some(s),
        }
    }

    fn subtree_max(&self, s: usize) -> (usize, usize) {
        let mut cur = s;
        let mut par = s;
        loop {
            match self.nodes.get(&cur).unwrap().right {
                Some(r) => {
                    par = cur;
                    cur = r;
                }
                None => return (cur, par),
            }
        }
    }
}

impl<T: Average + Copy + PartialEq + Display> Display for IntervalTree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "flowchart TD")?;
        let mut q = VecDeque::new();
        match self.root {
            None => {
                return Ok(());
            }
            Some(n) => q.push_back(n),
        };
        while let Some(n) = q.pop_front() {
            let node = self.nodes.get(&n).unwrap();
            writeln!(f, "{}(\"{}\")", n, node)?;
            if let Some(l) = node.left {
                writeln!(f, "{}-->{}", n, l)?;
                q.push_back(l);
            }
            if let Some(r) = node.right {
                writeln!(f, "{}-->{}", n, r)?;
                q.push_back(r);
            }
        }
        Ok(())
    }
}

impl<T: Average + Copy + PartialEq + Display> Node<T> {
    fn new(ival: (T, T)) -> Self {
        Self {
            center: ival.0.average_floor(&ival.1),
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
            left: None,
            right: None,
            balance: BF::Balanced,
        }
    }

    fn add(&mut self, ival: (T, T)) {
        if !self.sorted_by_first.contains(&ival) {
            self.sorted_by_first.push(ival);
            self.sorted_by_first.sort_by_key(|i| i.0);
            self.sorted_by_last.push(ival);
            self.sorted_by_last.sort_by_key(|i| std::cmp::Reverse(i.1));
        }
    }

    fn remove(&mut self, ival: (T, T)) -> bool {
        self.sorted_by_first.retain(|i| *i != ival);
        self.sorted_by_last.retain(|i| *i != ival);
        self.sorted_by_first.is_empty()
    }

    fn set_child(&mut self, d: Direction, c: Option<usize>) {
        match d {
            Direction::Left => self.left = c,
            Direction::Right => self.right = c,
        }
    }

    fn child(&self, d: Direction) -> Option<usize> {
        match d {
            Direction::Left => self.left,
            Direction::Right => self.right,
        }
    }
}

impl<T: Average + Copy + PartialEq + Display> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\\n", self.center)?;
        let ival = self.sorted_by_first[0];
        write!(f, "[{}, {}]", ival.0, ival.1)?;
        for ival in self.sorted_by_first[1..].iter() {
            write!(f, ", [{}, {}]", ival.0, ival.1)?;
        }
        write!(f, "\\n{}", self.balance)?;
        Ok(())
    }
}

impl Direction {
    fn flip(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

impl Display for BF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BF::Balanced => write!(f, "-"),
            BF::Heavy(Direction::Left) => write!(f, "<-"),
            BF::Heavy(Direction::Right) => write!(f, "->"),
            BF::Unbalanced(Direction::Left) => write!(f, "<--"),
            BF::Unbalanced(Direction::Right) => write!(f, "-->"),
        }
    }
}

impl Add<Direction> for BF {
    type Output = BF;

    fn add(self, rhs: Direction) -> Self::Output {
        match (self, rhs) {
            (BF::Balanced, dir) => BF::Heavy(dir),
            (BF::Heavy(d1), d2) => {
                if d1 == d2 {
                    BF::Unbalanced(d1)
                } else {
                    BF::Balanced
                }
            }
            (BF::Unbalanced(d1), d2) => {
                if d1 != d2 {
                    BF::Heavy(d1)
                } else {
                    panic!("balance out of bounds")
                }
            }
        }
    }
}

impl AddAssign<Direction> for BF {
    fn add_assign(&mut self, rhs: Direction) { *self = self.add(rhs); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_links<T: Average + Copy + PartialEq + Display>(
        t: &IntervalTree<T>,
        i: usize,
        l: Option<usize>,
        r: Option<usize>,
    ) {
        let Node { left, right, .. } = t.nodes.get(&i).unwrap();
        assert_eq!((*left, *right), (l, r), "node {}", i);
    }

    #[test]
    fn test_in_order_inserts() {
        let mut t = IntervalTree::default();
        for i in 0..7 {
            t.insert((i, i));
        }
        assert_eq!(t.root, Some(3));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), Some(2));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(1), Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);
    }

    #[test]
    fn test_reverse_order_inserts() {
        let mut t = IntervalTree::default();
        for i in (0..7).rev() {
            t.insert((i, i));
        }
        assert_eq!(t.root, Some(3));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(2), Some(0));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(5), Some(1));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(6), Some(4));
        check_links(&t, 6, None, None);
    }

    #[test]
    fn test_double_rotates() {
        let mut t = IntervalTree::default();
        // order chosen to trigger double rotations in both directions
        for i in [2, 6, 4, 0, 1, 3, 5] {
            t.insert((i, i));
        }
        assert_eq!(t.root, Some(0));
        check_links(&t, 0, Some(4), Some(2)); // 2
        check_links(&t, 1, Some(6), None); // 6
        check_links(&t, 2, Some(5), Some(1)); // 4
        check_links(&t, 3, None, None); // 0
        check_links(&t, 4, Some(3), None); // 1
        check_links(&t, 5, None, None); // 3
        check_links(&t, 6, None, None); //5
    }

    #[test]
    fn test_search_overlapping_inserts() {
        let mut t = IntervalTree::default();
        t.insert((-20, 20));
        t.insert((5, 15));
        t.insert((-9, -1));
        t.insert((1, 9));
        println!("{}", t);

        assert_eq!(t.search(8), vec![(-20, 20), (5, 15), (1, 9)]);
    }
}

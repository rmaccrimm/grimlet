use std::ops::{Add, AddAssign};

use anyhow::{Result, bail};
use num::integer::Average;

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalTree<T: Average + Copy> {
    root: Option<usize>,
    nodes: Vec<Node<T>>,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Node<T: Average + Copy> {
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

impl<T: Average + Copy> IntervalTree<T> {
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
            if self.nodes[n].balance != BF::Balanced {
                path.clear();
                anc = cur;
                anc_par = par;
            }
            let dir = if ival.1 < self.nodes[n].center {
                Direction::Left
            } else if ival.0 > self.nodes[n].center {
                Direction::Right
            } else {
                self.nodes[n].add(ival);
                return;
            };
            cur = self.nodes[n].child(dir);
            path.push(dir);
            par = Some((n, dir));
        }
        // Create new node
        match par {
            Some((p, dir)) => {
                self.nodes.push(Node::new(ival));
                let c = Some(self.nodes.len() - 1);
                self.nodes[p].set_child(dir, c);
            }
            None => {
                self.nodes.push(Node::new(ival));
                self.root = Some(0);
            }
        }
        let Some(a) = anc else { return };
        // Update balance factors
        let mut n = a;
        for dir in path {
            self.nodes[n].balance += dir;
            n = self.nodes[n].child(dir).unwrap();
        }
        self.rebalance_insert(a, anc_par);
    }

    /// Return all intervals containing b
    pub fn search(&self, b: T) -> Vec<(T, T)> {
        let mut curr = self.root;
        let mut results = vec![];

        while let Some(n) = curr {
            let node = &self.nodes[n];

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
        let mut n = match self.root {
            None => bail!("tree is empty"),
            Some(r) => r,
        };
        loop {
            if ival.1 < self.nodes[n].center {
                match self.nodes[n].left {
                    Some(l) => {
                        n = l;
                    }
                    None => bail!("interval not found"),
                }
            } else if ival.0 > self.nodes[n].center {
                match self.nodes[n].right {
                    Some(r) => {
                        n = r;
                    }
                    None => bail!("interval not found"),
                }
            } else {
                let node = &mut self.nodes[n];
                let p1 = node.sorted_by_first.iter().position(|&i| i == ival);
                let p2 = node.sorted_by_last.iter().position(|&i| i == ival);
                match (p1, p2) {
                    (Some(i), Some(j)) => {
                        let node = &mut self.nodes[n];
                        node.sorted_by_first.remove(i);
                        node.sorted_by_last.remove(j);
                        return Ok(());
                    }
                    _ => bail!("interval not found"),
                }
            }
        }
    }

    // Takes the following nodes
    // a: root of the sub-tree which needs to be re-balanced
    // parent: parent of a, may be None if node is the root
    fn rebalance_insert(&mut self, a: usize, parent: Option<(usize, Direction)>) {
        if let BF::Unbalanced(dir) = self.nodes[a].balance {
            let b = self.nodes[a].child(dir).unwrap();
            if let BF::Heavy(dir_bc) = self.nodes[b].balance {
                if dir_bc == dir.flip() {
                    let c = self.nodes[b].child(dir.flip()).unwrap();
                    self.rotate(dir, b, c);
                    self.nodes[a].set_child(dir, Some(c));
                    self.rotate(dir.flip(), a, c);
                    self.reparent(parent, c);
                    match self.nodes[c].balance {
                        BF::Heavy(dir_cd) => {
                            if dir_cd == dir {
                                self.nodes[b].balance = BF::Balanced;
                                self.nodes[a].balance = BF::Heavy(dir.flip());
                            } else {
                                self.nodes[b].balance = BF::Heavy(dir);
                                self.nodes[a].balance = BF::Balanced;
                            }
                        }
                        BF::Balanced => {
                            self.nodes[b].balance = BF::Balanced;
                            self.nodes[a].balance = BF::Balanced;
                        }
                        _ => panic!("invalid balance"),
                    }
                    self.nodes[c].balance = BF::Balanced;
                } else {
                    self.rotate(dir.flip(), a, b);
                    self.nodes[b].balance = BF::Balanced;
                    self.nodes[a].balance = BF::Balanced;
                    self.reparent(parent, b);
                }
            } else {
                panic!("invalid balance");
            }
        }
    }

    fn rotate(&mut self, d: Direction, p: usize, c: usize) {
        if self.nodes[p].child(d.flip()) != Some(c) {
            panic!("invalid rotation")
        }
        let c_child = self.nodes[c].child(d);
        self.nodes[p].set_child(d.flip(), c_child);
        self.nodes[c].set_child(d, Some(p));
        if self.root == Some(p) {
            self.root = Some(c);
        }
    }

    fn reparent(&mut self, par: Option<(usize, Direction)>, s: usize) {
        match par {
            Some((p, dir)) => {
                self.nodes[p].set_child(dir, Some(s));
            }
            None => self.root = Some(s),
        }
    }

    fn subtree_max(&self, mut s: Option<usize>) -> Option<usize> {
        let mut smax = s;
        while let Some(n) = s {
            smax = Some(n);
            s = self.nodes[n].right;
        }
        smax
    }
}

impl<T: Average + Copy> Node<T> {
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

impl Direction {
    fn flip(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
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

    fn check_links<T: Average + Copy>(
        t: &IntervalTree<T>,
        i: usize,
        l: Option<usize>,
        r: Option<usize>,
    ) {
        let Node { left, right, .. } = t.nodes[i];
        assert_eq!((left, right), (l, r), "node {}", i);
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
    fn test_overlapping_inserts() {
        let mut t = IntervalTree::default();
        t.insert((-20, 20));
        t.insert((5, 15));
        t.insert((-9, -1));
        t.insert((1, 9));
        println!("{:#?}", t);

        assert_eq!(t.search(8), vec![(-20, 20), (5, 15), (1, 9)]);
    }
}

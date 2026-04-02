use std::cmp::Reverse;

use anyhow::{Result, bail};

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalTree {
    root: Option<usize>,
    nodes: Vec<Node>,
}

// Balance factor
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
enum BF {
    #[default]
    Balanced = 0,
    LeftHeavy = -1,
    RightHeavy = 1,
    UnbalanceLeft = -2,
    UnbalancedRight = 2,
}

#[derive(Copy, Clone)]
enum Dir {
    Left,
    Right,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
    left: Option<usize>,
    right: Option<usize>,
    balance: BF,
}

impl IntervalTree {
    /// Insert a new interval. Has no effect if tree already contains the interval.
    pub fn insert(&mut self, ival: (u32, u32)) {
        // Current node being searched, it's parent and direction to add new node
        let mut cur = self.root;
        let mut prt: Option<(usize, Dir)> = None;
        // (Ancestor) root of the subtree for which we'll need to update balance factors, and the
        // path we took from it
        let mut anc: Option<usize> = self.root;
        let mut path: Vec<Dir> = vec![];

        // Search the tree
        while let Some(n) = cur {
            // Only need to go as far back as the last unbalanced node we saw on our way
            if self.nodes[n].balance != BF::Balanced {
                path.clear();
                anc = Some(n);
            }
            let dir = if ival.1 < self.nodes[n].center {
                cur = self.nodes[n].left;
                Dir::Left
            } else if ival.0 > self.nodes[n].center {
                cur = self.nodes[n].right;
                Dir::Right
            } else {
                self.nodes[n].add(ival);
                return;
            };
            path.push(dir);
            prt = Some((n, dir));
        }
        // Create new node
        match prt {
            Some((p, Dir::Left)) => {
                self.nodes.push(Node::new(ival));
                self.nodes[p].left = Some(self.nodes.len() - 1);
            }
            Some((p, Dir::Right)) => {
                self.nodes.push(Node::new(ival));
                self.nodes[p].right = Some(self.nodes.len() - 1);
            }
            None => {
                self.nodes.push(Node::new(ival));
                self.root = Some(0);
            }
        }
        let Some(y) = anc else { return };
        // Update balance factors
        let mut n = y;
        for dir in path {
            match dir {
                Dir::Left => {
                    self.nodes[n].dec_balance();
                    n = self.nodes[n].left.unwrap();
                }
                Dir::Right => {
                    self.nodes[n].inc_balance();
                    n = self.nodes[n].right.unwrap();
                }
            }
        }
        // Perform rebalancing
        if self.nodes[y].balance == BF::UnbalanceLeft {
            let x = self.nodes[y].left.unwrap();
            if self.nodes[x].balance == BF::RightHeavy {
                let w = self.nodes[x].right.unwrap();
                self.rotate_left(x, w);
            }
            self.rotate_right(y, x);
        } else if self.nodes[y].balance == BF::UnbalancedRight {
            let x = self.nodes[y].right.unwrap();
            if self.nodes[x].balance == BF::LeftHeavy {
                let w = self.nodes[x].left.unwrap();
                self.rotate_right(x, w);
            }
            self.rotate_left(y, x);
        }
    }

    /// Return all intervals containing x
    pub fn search(&self, x: u32) -> Vec<(u32, u32)> {
        let mut curr = self.root;
        let mut results = vec![];

        while let Some(n) = curr {
            let node = &self.nodes[n];

            if x == node.center {
                results.extend_from_slice(&node.sorted_by_first);
                return results;
            } else if x < node.center {
                let i = node.sorted_by_first.partition_point(|&r| r.0 <= x);
                results.extend_from_slice(&node.sorted_by_first[0..i]);
                curr = node.left;
            } else {
                let i = node.sorted_by_last.partition_point(|&r| r.1 >= x);
                results.extend_from_slice(&node.sorted_by_last[0..i]);
                curr = node.right;
            }
        }
        results
    }

    pub fn remove(&mut self, ival: (u32, u32)) -> Result<()> {
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
                    _ => bail!("Interval not found"),
                }
            }
        }
    }

    fn rotate_left(&mut self, p: usize, c: usize) {
        if self.nodes[p].right != Some(c) {
            panic!("invalid left rotation")
        }
        let c_left_child = self.nodes[c].left;

        self.nodes[p].right = c_left_child;
        self.nodes[c].left = Some(p);
        match self.nodes[c].balance {
            BF::Balanced => {
                self.nodes[p].balance = BF::RightHeavy;
                self.nodes[c].balance = BF::LeftHeavy;
            }
            BF::RightHeavy => {
                self.nodes[p].balance = BF::Balanced;
                self.nodes[c].balance = BF::Balanced;
            }
            _ => {
                panic!("invalid left rotation")
            }
        }
    }

    fn rotate_right(&mut self, p: usize, c: usize) {
        if self.nodes[p].left != Some(c) {
            panic!("invalid right rotation")
        }
        let c_right_child = self.nodes[c].right;

        self.nodes[p].left = c_right_child;
        self.nodes[c].right = Some(p);
        match self.nodes[c].balance {
            BF::Balanced => {
                self.nodes[p].balance = BF::LeftHeavy;
                self.nodes[c].balance = BF::RightHeavy;
            }
            BF::LeftHeavy => {
                self.nodes[p].balance = BF::Balanced;
                self.nodes[c].balance = BF::Balanced;
            }
            _ => {
                panic!("invalid right rotation")
            }
        }
    }
}

impl Node {
    fn new(ival: (u32, u32)) -> Self {
        Self {
            center: (ival.0 + ival.1) / 2,
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
            left: None,
            right: None,
            balance: BF::Balanced,
        }
    }

    fn add(&mut self, ival: (u32, u32)) {
        if !self.sorted_by_first.contains(&ival) {
            self.sorted_by_first.push(ival);
            self.sorted_by_first.sort_by_key(|i| i.0);
            self.sorted_by_last.push(ival);
            self.sorted_by_last.sort_by_key(|i| Reverse(i.1));
        }
    }

    fn inc_balance(&mut self) {
        self.balance = match (self.balance as i8) + 1 {
            -1 => BF::LeftHeavy,
            0 => BF::Balanced,
            1 => BF::RightHeavy,
            2 => BF::UnbalancedRight,
            _ => panic!("invalid balance factor"),
        };
    }

    fn dec_balance(&mut self) {
        self.balance = match (self.balance as i8) - 1 {
            -2 => BF::UnbalanceLeft,
            -1 => BF::LeftHeavy,
            0 => BF::Balanced,
            1 => BF::RightHeavy,
            _ => panic!("invalid balance factor"),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_links(t: &IntervalTree, i: usize, l: Option<usize>, r: Option<usize>) {
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
    fn test_rotate_left() {
        let mut tree = IntervalTree::default();
        for i in 0..7 {
            tree.insert((i, i));
        }

        let mut t = tree.clone();
        t.rotate_left(1, 2);
        assert_eq!(t.root, Some(3));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), None);
        check_links(&t, 2, Some(1), None);
        check_links(&t, 3, Some(2), Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);

        let mut t = tree.clone();
        t.rotate_left(3, 5);
        assert_eq!(t.root, Some(5));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), Some(2));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(1), Some(4));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(3), Some(6));
        check_links(&t, 6, None, None);
    }

    #[test]
    #[should_panic(expected = "invalid left rotation")]
    fn test_invalid_rotate_left() {
        let mut tree = IntervalTree::default();
        for i in 0..7 {
            tree.insert((i, i));
        }
        let mut t = tree.clone();
        t.rotate_left(5, 3);
    }

    #[test]
    fn test_rotate_right() {
        let mut tree = IntervalTree::default();
        for i in 0..7 {
            tree.insert((i, i));
        }
        let mut t = tree.clone();
        t.rotate_right(1, 0);
        check_links(&t, 0, None, Some(1));
        check_links(&t, 1, None, Some(2));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(0), Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);

        let mut t = tree.clone();
        t.rotate_right(3, 1);
        assert_eq!(t.root, Some(1));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), Some(3));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(2), Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);
    }

    #[test]
    #[should_panic(expected = "invalid right rotation")]
    fn test_invalid_rotate_right() {
        let mut tree = IntervalTree::default();
        for i in 0..7 {
            tree.insert((i, i));
        }
        let mut t = tree.clone();
        t.rotate_right(1, 5);
    }

    #[test]
    fn test_rotate_right_left() {
        let mut tree = IntervalTree::default();
        for i in 0..15 {
            tree.insert((i, i));
        }
        let mut t = tree.clone();
        // t.rotate_right_left(7, 11).unwrap();
        assert_eq!(t.root, Some(9));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), Some(2));
        check_links(&t, 2, None, None);
        check_links(&t, 3, Some(1), Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);
        check_links(&t, 7, Some(3), Some(8));
        check_links(&t, 8, None, None);
        check_links(&t, 9, Some(7), Some(11));
        check_links(&t, 10, None, None);
        check_links(&t, 11, Some(10), Some(13));
        check_links(&t, 12, None, None);
        check_links(&t, 13, Some(12), Some(14));
        check_links(&t, 14, None, None);
    }

    #[test]
    fn test_rotate_left_right() {
        let mut tree = IntervalTree::default();
        for i in 0..15 {
            tree.insert((i, i));
        }
        let mut t = tree.clone();
        assert_eq!(t.root, Some(7));
        check_links(&t, 0, None, None);
        check_links(&t, 1, Some(0), None);
        check_links(&t, 2, Some(1), Some(3));
        check_links(&t, 3, None, Some(5));
        check_links(&t, 4, None, None);
        check_links(&t, 5, Some(4), Some(6));
        check_links(&t, 6, None, None);
        check_links(&t, 7, Some(2), Some(11));
        check_links(&t, 8, None, None);
        check_links(&t, 9, Some(8), Some(10));
        check_links(&t, 10, None, None);
        check_links(&t, 11, Some(9), Some(13));
        check_links(&t, 12, None, None);
        check_links(&t, 13, Some(12), Some(14));
        check_links(&t, 14, None, None);
    }

    #[test]
    fn test_overlapping_inserts() {
        let mut t = IntervalTree::default();
        t.insert((20, 60));
        t.insert((20, 30));
        t.insert((22, 24));
        t.insert((24, 24));
    }
}

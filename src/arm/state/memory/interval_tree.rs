use std::cmp::Reverse;

use anyhow::{Result, bail};

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IntervalTree {
    root: Option<usize>,
    nodes: Vec<Node>,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
    left: Option<usize>,
    right: Option<usize>,
    parent: Option<usize>,
    balance: i8,
}

impl IntervalTree {
    /// Insert a new interval. Has no effect if tree already contains the interval.
    pub fn insert(&mut self, ival: (u32, u32)) {
        let mut n = match self.root {
            None => {
                self.nodes.push(Node::new(ival, None));
                self.root = Some(0);
                return;
            }
            Some(r) => r,
        };
        loop {
            if ival.1 < self.nodes[n].center {
                match self.nodes[n].left {
                    Some(l) => {
                        n = l;
                    }
                    None => {
                        self.nodes.push(Node::new(ival, Some(n)));
                        let c = self.nodes.len() - 1;
                        self.nodes[n].left = Some(c);
                        self.retrace(c);
                        break;
                    }
                }
            } else if ival.0 > self.nodes[n].center {
                match self.nodes[n].right {
                    Some(r) => {
                        n = r;
                    }
                    None => {
                        self.nodes.push(Node::new(ival, Some(n)));
                        let c = self.nodes.len() - 1;
                        self.nodes[n].right = Some(c);
                        self.retrace(c);
                        break;
                    }
                }
            } else {
                self.nodes[n].add(ival);
                break;
            }
        }
    }

    /// Return all intervals containing x
    pub fn search(&self, x: u32) -> Vec<(u32, u32)> {
        let mut n = match self.root {
            None => return vec![],
            Some(r) => r,
        };
        let mut results = vec![];
        loop {
            let node = &self.nodes[n];

            if x == node.center {
                results.extend_from_slice(&node.sorted_by_first);
                return results;
            } else if x < node.center {
                let i = node.sorted_by_first.partition_point(|&r| r.0 <= x);
                results.extend_from_slice(&node.sorted_by_first[0..i]);
                match node.left {
                    Some(l) => {
                        n = l;
                    }
                    None => return results,
                }
            } else {
                let i = node.sorted_by_last.partition_point(|&r| r.1 >= x);
                results.extend_from_slice(&node.sorted_by_last[0..i]);
                match node.right {
                    Some(r) => {
                        n = r;
                    }
                    None => return results,
                }
            }
        }
    }

    /// Re-trace the tree from child node `c` towards the root, performing re-balancing operations
    /// as needed.
    fn retrace(&mut self, mut c: usize) {
        while let Some(p) = self.nodes[c].parent {
            let is_right_child = match self.nodes[p].right {
                Some(r) => c == r,
                None => false,
            };
            let c_balance = self.nodes[c].balance;
            let p_balance = self.nodes[p].balance;

            // new root of the sub-tree and grand-parent, updated at end of loop iteration
            if is_right_child {
                if self.nodes[p].balance > 0 {
                    // unbalanced to the right
                    if c_balance < 0 {
                        self.rotate_right_left(p, c).unwrap();
                    } else {
                        self.rotate_left(p, c).unwrap();
                    }
                    break;
                } else if p_balance < 0 {
                    self.nodes[p].balance = 0;
                    break;
                } else {
                    self.nodes[p].balance = 1;
                    c = p;
                    continue;
                }
            } else {
                if p_balance < 0 {
                    // unbalanced to the left
                    if c_balance > 0 {
                        self.rotate_left_right(p, c).unwrap();
                    } else {
                        self.rotate_right(p, c).unwrap();
                    }
                    break;
                } else if p_balance > 0 {
                    self.nodes[p].balance = 0;
                    break;
                } else {
                    self.nodes[p].balance = -1;
                    c = p;
                    continue;
                }
            };
        }
    }

    /// g: grand-parent
    /// p: previous parent (pre-rotation)
    /// n: new parent (post-rotation)
    fn fix_parent(&mut self, g: Option<usize>, p: usize, n: usize) {
        self.nodes[n].parent = g;
        match g {
            Some(gi) => {
                if self.nodes[gi].left == Some(p) {
                    self.nodes[gi].left = Some(n);
                } else {
                    self.nodes[gi].right = Some(n);
                }
            }
            None => {
                self.root = Some(n);
            }
        }
    }

    // p_balance = +1, c_balance in (0, +1)
    fn rotate_left(&mut self, p: usize, c: usize) -> Result<()> {
        if self.nodes[p].right != Some(c) || self.nodes[c].parent != Some(p) {
            bail!("invalid left rotation")
        }
        let g = self.nodes[p].parent;
        let c_left_child = self.nodes[c].left;

        self.nodes[p].right = c_left_child;
        self.nodes[c].left = Some(p);
        self.nodes[p].parent = Some(c);
        if let Some(n) = c_left_child {
            self.nodes[n].parent = Some(p);
        }
        if self.nodes[c].balance == 0 {
            self.nodes[p].balance = 1;
            self.nodes[c].balance = -1;
        } else if self.nodes[c].balance == 1 {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        } else {
            bail!("invalid left rotation")
        }
        self.fix_parent(g, p, c);
        Ok(())
    }

    // p_balance = +1, c_balance = -1
    fn rotate_right_left(&mut self, p: usize, c: usize) -> Result<()> {
        // Guaranteed to not be None, by balance of c
        let y = self.nodes[c].left.unwrap();

        // 1st right rotation
        let y_right_child = self.nodes[y].right;
        self.nodes[c].left = y_right_child;
        self.nodes[c].parent = Some(y);
        self.nodes[y].right = Some(c);
        self.nodes[y].parent = Some(p);
        if let Some(n) = y_right_child {
            self.nodes[n].parent = Some(c);
        }

        // 2nd left rotation
        let y_left_child = self.nodes[y].left;
        self.nodes[p].right = y_left_child;
        self.nodes[y].left = Some(p);
        self.nodes[p].parent = Some(y);
        if let Some(n) = y_left_child {
            self.nodes[n].parent = Some(p);
        }

        let y_balance = self.nodes[y].balance;
        if y_balance == 0 {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        } else if y_balance > 0 {
            self.nodes[p].balance = -1;
            self.nodes[c].balance = 0;
        } else {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 1;
        }
        self.nodes[y].balance = 0;
        Ok(())
    }

    // p_balance = -1, c_balance in (0, -1)
    fn rotate_right(&mut self, p: usize, c: usize) -> Result<()> {
        if self.nodes[p].left != Some(c) || self.nodes[c].parent != Some(p) {
            bail!("invalid right rotation")
        }
        let g = self.nodes[p].parent;
        let c_right_child = self.nodes[c].right;

        self.nodes[p].left = c_right_child;
        self.nodes[c].right = Some(p);
        self.nodes[p].parent = Some(c);
        if let Some(n) = c_right_child {
            self.nodes[n].parent = Some(p);
        }
        if self.nodes[c].balance == 0 {
            self.nodes[p].balance = -1;
            self.nodes[c].balance = 1;
        } else if self.nodes[c].balance == -1 {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        } else {
            bail!("invalid right rotation")
        }
        self.fix_parent(g, p, c);
        Ok(())
    }

    fn rotate_left_right(&mut self, p: usize, c: usize) -> Result<()> {
        // Guaranteed to not be None, by balance of c
        let y = self.nodes[c].right.unwrap();

        // 1st left rotation
        let y_left_child = self.nodes[y].left;
        self.nodes[c].right = y_left_child;
        self.nodes[c].parent = Some(y);
        self.nodes[y].left = Some(c);
        self.nodes[y].parent = Some(p);
        if let Some(n) = y_left_child {
            self.nodes[n].parent = Some(c);
        }

        // 2nd right rotation
        let y_right_child = self.nodes[y].right;
        self.nodes[p].left = y_right_child;
        self.nodes[y].right = Some(p);
        self.nodes[p].parent = Some(y);
        if let Some(n) = y_right_child {
            self.nodes[n].parent = Some(p);
        }

        let y_balance = self.nodes[y].balance;
        if y_balance == 0 {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        } else if y_balance > 0 {
            self.nodes[p].balance = 1;
            self.nodes[c].balance = 0;
        } else {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = -1;
        }
        self.nodes[y].balance = 0;
        Ok(())
    }
}

impl Node {
    fn new(ival: (u32, u32), parent: Option<usize>) -> Self {
        Self {
            center: (ival.0 + ival.1) / 2,
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
            left: None,
            right: None,
            parent,
            balance: 0,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_tree() -> IntervalTree {
        let mut t = IntervalTree::default();
        // Expected:
        //        3
        //       /  \
        //     1     5
        //    / \   / \
        //   0   2 4   6
        t.insert((0, 0));
        t.insert((1, 1));
        t.insert((2, 2));
        t.insert((3, 3));
        t.insert((4, 4));
        t.insert((5, 5));
        t.insert((6, 6));
        t
    }

    fn check_links(
        t: &IntervalTree,
        i: usize,
        l: Option<usize>,
        r: Option<usize>,
        p: Option<usize>,
    ) {
        let Node {
            left,
            right,
            parent,
            ..
        } = t.nodes[i];
        assert_eq!((left, right, parent), (l, r, p));
    }

    #[test]
    fn test_in_order_inserts() {
        let t = build_tree();
        assert_eq!(t.root, Some(3));
        check_links(&t, 0, None, None, Some(1));
        check_links(&t, 1, Some(0), Some(2), Some(3));
        check_links(&t, 2, None, None, Some(1));
        check_links(&t, 3, Some(1), Some(5), None);
        check_links(&t, 4, None, None, Some(5));
        check_links(&t, 5, Some(4), Some(6), Some(3));
        check_links(&t, 6, None, None, Some(5));
    }

    #[test]
    fn test_rotate_left() {
        let tree = build_tree();

        let mut t = tree.clone();
        t.rotate_left(1, 2).unwrap();
        assert_eq!(t.root, Some(3));
        check_links(&t, 0, None, None, Some(1));
        check_links(&t, 1, Some(0), None, Some(2));
        check_links(&t, 2, Some(1), None, Some(3));
        check_links(&t, 3, Some(2), Some(5), None);
        check_links(&t, 4, None, None, Some(5));
        check_links(&t, 5, Some(4), Some(6), Some(3));
        check_links(&t, 6, None, None, Some(5));

        let mut t = tree.clone();
        t.rotate_left(3, 5).unwrap();
        assert_eq!(t.root, Some(5));
        check_links(&t, 0, None, None, Some(1));
        check_links(&t, 1, Some(0), Some(2), Some(3));
        check_links(&t, 2, None, None, Some(1));
        check_links(&t, 3, Some(1), Some(4), Some(5));
        check_links(&t, 4, None, None, Some(3));
        check_links(&t, 5, Some(3), Some(6), None);
        check_links(&t, 6, None, None, Some(5));
    }

    #[test]
    fn test_invalid_rotate_left() {
        let tree = build_tree();
        let mut t = tree.clone();
        assert!(t.rotate_left(5, 3).is_err());
        assert_eq!(t, tree);
        assert!(t.rotate_left(0, 0).is_err());
        assert_eq!(t, tree);
        assert!(t.rotate_left(4, 1).is_err());
        assert_eq!(t, tree);
    }
}

use std::cmp::{Ordering, Reverse};

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Debug, Default)]
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
            if self.nodes[n] < ival {
                match self.nodes[n].left {
                    Some(l) => {
                        n = l;
                    }
                    None => {
                        self.nodes.push(Node::new(ival, Some(n)));
                        self.nodes[n].left = Some(self.nodes.len());
                        self.retrace(n);
                    }
                }
            } else if self.nodes[n] > ival {
                match self.nodes[n].right {
                    Some(r) => {
                        n = r;
                    }
                    None => {
                        self.nodes.push(Node::new(ival, Some(n)));
                        self.nodes[n].right = Some(self.nodes.len());
                        self.retrace(n);
                    }
                }
            } else {
                self.nodes[n].add(ival);
            }
        }
    }

    /// Return all intervals containing x
    pub fn search(&self, x: u32) -> Vec<(u32, u32)> {
        let mut n = match self.root {
            None => return vec![],
            Some(r) => r,
        };
        loop {
            let node = &self.nodes[n];

            if x == node.center {
                return node.sorted_by_first.clone();
            } else if x < node.center {
                let i = node.sorted_by_first.partition_point(|&r| r.0 <= x);
                if i == 0 {
                    match node.left {
                        Some(l) => {
                            n = l;
                            continue;
                        }
                        None => return vec![],
                    }
                } else {
                    return node.sorted_by_first[0..i].to_vec();
                }
            } else {
                let i = node.sorted_by_last.partition_point(|&r| r.1 >= x);
                if i == 0 {
                    match node.right {
                        Some(r) => {
                            n = r;
                            continue;
                        }
                        None => return vec![],
                    }
                } else {
                    return node.sorted_by_last[0..i].to_vec();
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
            let (n, g): (usize, Option<usize>) = if is_right_child {
                if self.nodes[p].balance > 0 {
                    // unbalanced to the right
                    let g = self.nodes[p].parent;
                    let n = if c_balance < 0 {
                        self.rotate_right_left(p, c)
                    } else {
                        self.rotate_left(p, c)
                    };
                    (n, g)
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
                    let g = self.nodes[p].parent;
                    let n = if c_balance > 0 {
                        self.rotate_left_right(p, c)
                    } else {
                        self.rotate_right(p, c)
                    };
                    (n, g)
                } else if p_balance > 0 {
                    self.nodes[p].balance = 0;
                    break;
                } else {
                    self.nodes[p].balance = -1;
                    c = p;
                    continue;
                }
            };

            self.nodes[n].parent = g;
            match g {
                Some(gi) => {
                    if self.nodes[gi].left == Some(c) {
                        self.nodes[gi].left = Some(n);
                    } else {
                        self.nodes[gi].right = Some(n);
                    }
                }
                None => {
                    self.root = Some(n);
                }
            }
            break;
        }
    }

    // p_balance = +1, c_balance in (0, +1)
    fn rotate_left(&mut self, p: usize, c: usize) -> usize {
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
        } else {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        }
        c
    }

    // p_balance = +1, c_balance = -1
    fn rotate_right_left(&mut self, p: usize, c: usize) -> usize {
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
        y
    }

    fn rotate_right(&mut self, p: usize, c: usize) -> usize {
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
        } else {
            self.nodes[p].balance = 0;
            self.nodes[c].balance = 0;
        }
        c
    }

    fn rotate_left_right(&mut self, p: usize, c: usize) -> usize {
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
        y
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

impl PartialOrd<(u32, u32)> for Node {
    fn partial_cmp(&self, ival: &(u32, u32)) -> Option<std::cmp::Ordering> {
        if self.center < ival.0 {
            Some(Ordering::Less)
        } else if self.center > ival.1 {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl PartialEq<(u32, u32)> for Node {
    fn eq(&self, ival: &(u32, u32)) -> bool { !(self.center < ival.0 || self.center > ival.1) }
}

use std::cmp::{Ordering, Reverse};

use num::Integer;

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Debug)]
pub struct IntervalTree {
    nodes: Vec<Option<Node>>,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
    balance: i8,
}

impl IntervalTree {
    pub fn insert(&mut self, ival: (u32, u32)) {
        if self.nodes[0].is_none() {
            self.nodes[0] = Some(Node::new(ival));
            return;
        };
        let mut curr = 0;
        loop {
            let node = match &mut self.nodes[curr] {
                None => {
                    self.nodes[curr] = Some(Node::new(ival));
                    return;
                }
                Some(node) => node,
            };
            if *node < ival {
                match self.left_ind(curr) {
                    Ok(l) => {
                        curr = l;
                    }
                    Err(l) => {
                        curr = l;
                        self.increase_height();
                    }
                }
            } else if *node > ival {
                match self.right_ind(curr) {
                    Ok(r) => {
                        curr = r;
                    }
                    Err(r) => {
                        curr = r;
                        self.increase_height();
                    }
                }
            } else {
                node.add(ival);
            }
        }
    }

    pub fn search(&self, x: u32) -> Vec<(u32, u32)> {
        let mut curr = match self.nodes[0] {
            Some(_) => 0,
            None => return vec![],
        };
        loop {
            let node = match &self.nodes[curr] {
                None => return vec![],
                Some(node) => node,
            };
            if x == node.center {
                return node.sorted_by_first.clone();
            } else if x < node.center {
                let i = node.sorted_by_first.partition_point(|&r| r.0 <= x);
                if i == 0 {
                    match self.left_ind(i) {
                        Ok(l) => {
                            curr = l;
                            continue;
                        }
                        Err(_) => return vec![],
                    }
                } else {
                    return node.sorted_by_first[0..i].to_vec();
                }
            } else {
                let i = node.sorted_by_last.partition_point(|&r| r.1 >= x);
                if i == 0 {
                    match self.right_ind(curr) {
                        Ok(r) => {
                            curr = r;
                            continue;
                        }
                        Err(_) => return vec![],
                    }
                } else {
                    return node.sorted_by_last[0..i].to_vec();
                }
            }
        }
    }

    fn left_ind(&self, i: usize) -> Result<usize, usize> {
        let l = 2 * i + 1;
        if l >= self.nodes.len() { Err(l) } else { Ok(l) }
    }

    fn right_ind(&self, i: usize) -> Result<usize, usize> {
        let r = 2 * i + 2;
        if r >= self.nodes.len() { Err(r) } else { Ok(r) }
    }

    fn parent_ind(&self, i: usize) -> Option<usize> {
        if i == 0 { None } else { Some((i - 1) / 2) }
    }

    fn increase_height(&mut self) {
        let new_len = (self.nodes.len() + 1) * 2 - 1;
        self.nodes.resize(new_len, None);
    }

    fn retrace(&mut self, mut c: usize) {
        let mut n: Option<usize> = None;
        let mut g: Option<usize> = None;

        while let Some(p) = self.parent_ind(c) {
            let curr_balance = self.nodes[c].as_ref().unwrap().balance;
            let parent = self.nodes[p].as_mut().unwrap();

            let is_right_child = c.is_odd();
            if is_right_child {
                if parent.balance > 0 {
                    // unbalanced to the right
                    g = self.parent_ind(p);
                    if curr_balance < 0 {
                        let n = Some(self.rotate_right_left(p, c));
                    } else {
                        let n = Some(self.rotate_left(p, c));
                    }
                } else if parent.balance < 0 {
                    parent.balance = 0;
                    break;
                } else {
                    parent.balance = 1;
                    c = p;
                    continue;
                }
            } else {
                if parent.balance < 0 {
                    // unbalanced to the left
                    g = self.parent_ind(p);
                    if curr_balance > 0 {
                        n = Some(self.rotate_left_right(p, c));
                    } else {
                        n = Some(self.rotate_right(p, c));
                    }
                } else if parent.balance > 0 {
                    parent.balance = 0;
                    break;
                } else {
                    parent.balance = -1;
                    c = p;
                    continue;
                }
            }
        }
    }

    fn rotate_left(&mut self, p: usize, c: usize) -> usize {
        let child = self.nodes[c].take();
        let parent = self.nodes[p].take();
        let c_lchild = self.nodes[self.left_ind(c).unwrap()].take();
        // 
        

        let 
    }

    fn rotate_right(&mut self, p: usize, c: usize) -> usize {
        todo!();
    }

    fn rotate_left_right(&mut self, p: usize, c: usize) -> usize {
        todo!();
    }

    fn rotate_right_left(&mut self, p: usize, c: usize) -> usize {
        todo!();
    }
}

impl Default for IntervalTree {
    fn default() -> Self {
        Self {
            // Just picking a power of 2 - 1 (num elements for complete tree) for initial size
            nodes: vec![None; 15],
        }
    }
}

impl Node {
    fn new(ival: (u32, u32)) -> Self {
        Self {
            center: (ival.0 + ival.1) / 2,
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
            balance: 0,
        }
    }

    fn add(&mut self, ival: (u32, u32)) {
        self.sorted_by_first.push(ival);
        self.sorted_by_first.sort_by_key(|i| i.0);
        self.sorted_by_last.push(ival);
        self.sorted_by_last.sort_by_key(|i| Reverse(i.1));
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

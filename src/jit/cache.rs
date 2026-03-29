#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};

use super::CompiledFunction;

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AddrRange {
    first: u32,
    last: u32,
}

/// Wrapper around a hashmap that also maintains a count of how often each entry is accessed
/// (I'm thinking this will be useful to clear up memory in the future by removing infrequently
/// accessed blocks).
#[derive(Default)]
pub struct FunctionCache<'ctx> {
    map: HashMap<AddrRange, CacheEntry<'ctx>>,
}

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Default)]
pub struct IntervalTree {
    root: Option<Box<Node>>,
}

struct CacheEntry<'ctx> {
    func: CompiledFunction<'ctx>,
    hits: usize,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl From<AddrRange> for (u32, u32) {
    fn from(value: AddrRange) -> Self { (value.first, value.last) }
}

impl<'ctx> FunctionCache<'ctx> {
    pub fn insert(
        &mut self,
        k: AddrRange,
        v: CompiledFunction<'ctx>,
    ) -> Option<CompiledFunction<'ctx>> {
        self.map
            .insert(k, CacheEntry { func: v, hits: 0 })
            .map(|e| e.func)
    }

    pub fn get(&mut self, k: &AddrRange) -> Option<&CompiledFunction<'ctx>> {
        if let Some(e) = self.map.get_mut(k) {
            e.hits += 1;
            Some(&e.func)
        } else {
            None
        }
    }

    pub fn remove(&mut self, k: &AddrRange) -> Option<CompiledFunction<'ctx>> {
        self.map.remove(k).map(|e| e.func)
    }
}

impl IntervalTree {
    pub fn search(&self, addr: u32) -> Vec<(u32, u32)> {
        match &self.root {
            Some(n) => n.search(addr),
            None => vec![],
        }
    }

    pub fn insert(&mut self, ival: (u32, u32)) {
        match &mut self.root {
            Some(n) => n.insert(ival),
            None => self.root = Some(Box::new(Node::new(ival))),
        }
    }

    pub fn remove(&mut self, ival: (u32, u32)) -> Result<()> {
        match &mut self.root {
            Some(n) => n.remove(ival),
            None => bail!("Tree is empty"),
        }
    }
}

fn take_predecessor(n: &mut Box<Node>) -> Box<Node> {
    let mut curr = n;
    let mut child = n.left;
    loop {
        match child.right {}
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
        }
    }

    fn search(&self, x: u32) -> Vec<(u32, u32)> {
        if x == self.center {
            self.sorted_by_first.clone()
        } else if x < self.center {
            let i = self.sorted_by_first.partition_point(|&r| r.0 <= x);
            if i == 0 {
                match &self.left {
                    Some(left) => left.search(x),
                    None => vec![],
                }
            } else {
                self.sorted_by_first[0..i].to_vec()
            }
        } else {
            let i = self.sorted_by_last.partition_point(|&r| r.1 >= x);
            if i == 0 {
                match &self.right {
                    Some(right) => right.search(x),
                    None => vec![],
                }
            } else {
                self.sorted_by_last[0..i].to_vec()
            }
        }
    }

    fn insert(&mut self, ival: (u32, u32)) {
        if ival.1 < self.center {
            match &mut self.left {
                Some(n) => n.insert(ival),
                None => self.left = Some(Box::new(Node::new(ival))),
            }
        } else if ival.0 > self.center {
            match &mut self.right {
                Some(n) => n.insert(ival),
                None => self.right = Some(Box::new(Node::new(ival))),
            }
        } else {
            self.sorted_by_first.push(ival);
            self.sorted_by_first.sort_by_key(|i| i.0);
            self.sorted_by_last.push(ival);
            self.sorted_by_last.sort_by_key(|i| Reverse(i.1));
        }
    }

    fn remove(&mut self, ival: (u32, u32)) -> Result<bool> {
        if ival.1 < self.center {
            match &mut self.left {
                Some(n) => {
                    let res = n.remove(ival);
                    if let Ok(true) = res {
                        if let Some(l) = &n.left {
                            let pred = max_child(l);
                            self.left = Some(pred.clone());
                            self.left.remove()
                            // get right-most and promote
                        } else if let Some(_) = &n.right {
                            // Make it's right child our left
                        } else {
                            // trivial case
                        }
                        Ok(false)
                    } else {
                        res
                    }
                }
                None => bail!("interval {:?} not found", ival),
            }
        } else if ival.0 > self.center {
            match &mut self.right {
                Some(n) => {
                    let res = n.remove(ival);
                    if let Ok(true) = res {
                        if let Some(_) = &n.left {
                            // get right-most and promote
                        } else if let Some(_) = &n.right {
                            // Make it's right child our right
                        } else {
                            // trivial case
                        }
                        Ok(false)
                    } else {
                        res
                    }
                }
                None => bail!("interval {:?} not found", ival),
            }
        } else {
            let i = self
                .sorted_by_first
                .iter()
                .position(|x| *x == ival)
                .ok_or(anyhow!("interval {:?} not found", ival))?;
            let j = self
                .sorted_by_first
                .iter()
                .position(|x| *x == ival)
                .ok_or(anyhow!("interval {:?} not found", ival))?;
            self.sorted_by_first.remove(i);
            self.sorted_by_last.remove(j);
            Ok(self.sorted_by_first.is_empty())
        }
    }

    fn fix(&mut self) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use super::*;

    #[test]
    fn test_node_check() {
        let center = 50;
        let intervals = vec![(49, 51), (30, 52), (50, 55), (34, 50), (1, 100)];

        let mut sorted_by_first = intervals.clone();
        sorted_by_first.sort_by_key(|r| r.0);

        let mut sorted_by_last = intervals.clone();
        sorted_by_last.sort_by_key(|r| Reverse(r.1));

        let left = Node {
            center: 5,
            sorted_by_first: vec![(3, 6)],
            sorted_by_last: vec![(3, 6)],
            left: None,
            right: None,
        };

        let n = Node {
            center,
            sorted_by_first,
            sorted_by_last,
            left: Some(Rc::new(left)),
            right: None,
        };

        println!("{:?}", n);

        assert_eq!(
            n.check(50),
            NodeCheck::Match(vec![(1, 100), (30, 52), (34, 50), (49, 51), (50, 55)])
        );
        assert_eq!(
            n.check(49),
            NodeCheck::Match(vec![(1, 100), (30, 52), (34, 50), (49, 51)])
        );
        assert_eq!(n.check(31), NodeCheck::Match(vec![(1, 100), (30, 52)]));
        assert_eq!(n.check(10), NodeCheck::Match(vec![(1, 100)]));
        assert_eq!(n.check(1), NodeCheck::Match(vec![(1, 100)]));
        assert_eq!(n.check(0), NodeCheck::Search(n.left.clone()));
        assert_eq!(
            n.check(51),
            NodeCheck::Match(vec![(1, 100), (50, 55), (30, 52), (49, 51)])
        );
        assert_eq!(n.check(55), NodeCheck::Match(vec![(1, 100), (50, 55)]));
        assert_eq!(n.check(100), NodeCheck::Match(vec![(1, 100)]));
        assert_eq!(n.check(101), NodeCheck::Search(None));
    }
}

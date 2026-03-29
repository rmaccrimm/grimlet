#![allow(dead_code)]

use std::collections::HashMap;
use std::rc::Rc;

use super::CompiledFunction;

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AddrRange {
    first: u32,
    last: u32,
}

/// Wrapper around a hashmap that also maintains a count of how often each entry is retreived
#[derive(Default)]
pub struct FunctionCache<'ctx> {
    map: HashMap<AddrRange, CacheEntry<'ctx>>,
}

// Some premature optimization, just for fun. Used to lookup all cached function blocks covering a
// given address.
#[derive(Default)]
pub struct IntervalTree {
    root: Option<Rc<Node>>,
}

struct CacheEntry<'ctx> {
    func: CompiledFunction<'ctx>,
    hits: usize,
}

#[derive(Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
    left: Option<Rc<Node>>,
    right: Option<Rc<Node>>,
}

#[derive(PartialEq, Eq, Debug)]
enum NodeCheck {
    Match(Vec<(u32, u32)>),
    Search(Option<Rc<Node>>),
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
    pub fn query(&self, addr: u32) -> Vec<AddrRange> {
        let mut curr = self.root.clone();
        loop {
            match curr {
                Some(n) => match n.check(addr) {
                    NodeCheck::Match(res) => {
                        return res
                            .into_iter()
                            .map(|(first, last)| AddrRange { first, last })
                            .collect();
                    }
                    NodeCheck::Search(child) => curr = child,
                },
                None => return vec![],
            }
        }
    }
}

impl Node {
    fn check(&self, x: u32) -> NodeCheck {
        if x == self.center {
            NodeCheck::Match(self.sorted_by_first.clone())
        } else if x < self.center {
            let i = self.sorted_by_first.partition_point(|&r| r.0 <= x);
            if i == 0 {
                NodeCheck::Search(self.left.clone())
            } else {
                NodeCheck::Match(self.sorted_by_first[0..i].to_vec())
            }
        } else {
            let i = self.sorted_by_last.partition_point(|&r| r.1 >= x);
            if i == 0 {
                NodeCheck::Search(self.right.clone())
            } else {
                NodeCheck::Match(self.sorted_by_last[0..i].to_vec())
            }
        }
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

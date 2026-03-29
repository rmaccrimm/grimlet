use std::cmp::{Ordering, Reverse};
use std::collections::BTreeSet;

use anyhow::{Result, anyhow};

#[derive(Default, Debug)]
pub struct IntervalTree {
    btree: BTreeSet<Node>,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
struct Node {
    center: u32,
    sorted_by_first: Vec<(u32, u32)>,
    sorted_by_last: Vec<(u32, u32)>,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.sorted_by_last[0].1 < other.center {
            Ordering::Less
        } else if self.sorted_by_first[0].0 > other.center {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl IntervalTree {
    pub fn insert(&mut self, ival: (u32, u32)) {
        let n = Node::new(ival);
        match self.btree.get(&n) {
            Some(overlap) => {
                let mut updated = overlap.clone();
                updated.add(ival);
                self.btree.replace(updated).unwrap();
            }
            None => {
                self.btree.insert(n);
            }
        }
    }

    pub fn remove(&mut self, ival: (u32, u32)) -> Result<()> {
        let n = Node::new(ival);
        match self.btree.get(&n) {
            Some(m) => {
                let i = m
                    .sorted_by_first
                    .iter()
                    .position(|x| *x == ival)
                    .ok_or(anyhow!("interval {:?} not found", ival))?;
                let j = m
                    .sorted_by_first
                    .iter()
                    .position(|x| *x == ival)
                    .ok_or(anyhow!("interval {:?} not found", ival))?;
                let mut updated = m.clone();
                updated.sorted_by_first.remove(i);
                updated.sorted_by_last.remove(j);

                if updated.sorted_by_first.is_empty() {
                    self.btree.remove(&n);
                } else {
                    self.btree.replace(updated);
                }
                Ok(())
            }
            None => Err(anyhow!("interval {:?} not found", ival)),
        }
    }
}

impl Node {
    fn new(ival: (u32, u32)) -> Self {
        Self {
            center: (ival.0 + ival.1) / 2,
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
        }
    }

    fn add(&mut self, ival: (u32, u32)) {
        self.sorted_by_first.push(ival);
        self.sorted_by_first.sort_by_key(|i| i.0);
        self.sorted_by_last.push(ival);
        self.sorted_by_last.sort_by_key(|i| Reverse(i.1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_tree() {
        let mut tree = IntervalTree::default();
        tree.insert((45, 55));
        tree.insert((1, 10));
        tree.insert((8, 12));
        tree.insert((56, 57));
        tree.insert((70, 80));
        tree.insert((65, 68));
        tree.insert((46, 54));
        tree.remove((45, 55)).unwrap();
        tree.remove((46, 54)).unwrap();
        println!("{:#?}", tree);
        assert!(false);
    }
}

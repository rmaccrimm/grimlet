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
    UnbalancedLeft = -2,
    UnbalancedRight = 2,
}

#[derive(Copy, Clone)]
enum Traversal {
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

trait Symmetric {
    fn flip(self) -> Self;
}

trait Transform {
    fn t<S>(s: S) -> S
    where
        S: Symmetric;
}

struct Forward();

struct Reverse();

impl IntervalTree {
    /// Insert a new interval. Has no effect if tree already contains the interval.
    pub fn insert(&mut self, ival: (u32, u32)) {
        // Current node being searched, it's parent and direction to add new node
        let mut cur = self.root;
        let mut par: Option<(usize, Traversal)> = None;
        // (Ancestor) root of the subtree for which we'll need to update balance factors, and the
        // path we took from it. Either the last unbalanced node encountered, or root.
        let mut anc = self.root;
        let mut anc_par: Option<(usize, Traversal)> = None;
        let mut path: Vec<Traversal> = vec![];

        // Search the tree
        while let Some(n) = cur {
            // Only need to go as far back as the last unbalanced node we saw on our way
            if self.nodes[n].balance != BF::Balanced {
                path.clear();
                anc = cur;
                anc_par = par;
            }
            let dir = if ival.1 < self.nodes[n].center {
                Traversal::Left
            } else if ival.0 > self.nodes[n].center {
                Traversal::Right
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
            match dir {
                Traversal::Left => {
                    self.nodes[n].dec_balance();
                    n = self.nodes[n].left.unwrap();
                }
                Traversal::Right => {
                    self.nodes[n].inc_balance();
                    n = self.nodes[n].right.unwrap();
                }
            }
        }
        // Perform rebalancing
        self.rebalance_insert::<Forward>(a, anc_par);
        self.rebalance_insert::<Reverse>(a, anc_par);
    }

    fn reparent(&mut self, par: Option<(usize, Traversal)>, s: usize) {
        match par {
            Some((p, dir)) => match dir {
                Traversal::Right => self.nodes[p].right = Some(s),
                Traversal::Left => self.nodes[p].left = Some(s),
            },
            None => self.root = Some(s),
        }
    }

    // Takes the following nodes
    // a: root of the sub-tree which needs to be re-balanced
    // parent: parent of node, may be None if node is the root
    fn rebalance_insert<T>(&mut self, a: usize, parent: Option<(usize, Traversal)>)
    where
        T: Transform,
    {
        if T::t(self.nodes[a].balance) == BF::UnbalancedLeft {
            let b = self.nodes[a].child(T::t(Traversal::Left)).unwrap();
            match T::t(self.nodes[b].balance) {
                BF::RightHeavy => {
                    let c = self.nodes[b].child(T::t(Traversal::Right)).unwrap();
                    self.rotate(T::t(Traversal::Left), b, c);
                    self.rotate(T::t(Traversal::Right), a, b);
                    match self.nodes[c].balance {
                        BF::LeftHeavy => {
                            self.nodes[b].balance = BF::Balanced;
                            self.nodes[a].balance = BF::RightHeavy;
                        }
                        BF::Balanced => {
                            self.nodes[b].balance = BF::Balanced;
                            self.nodes[a].balance = BF::Balanced;
                        }
                        BF::RightHeavy => {
                            self.nodes[b].balance = BF::LeftHeavy;
                            self.nodes[a].balance = BF::Balanced;
                        }
                        _ => panic!("unexpected imbalance"),
                    }
                    self.nodes[c].balance = BF::Balanced;
                    self.reparent(parent, c);
                }
                BF::LeftHeavy => {
                    self.rotate(T::t(Traversal::Right), a, b);
                    self.nodes[b].balance = BF::Balanced;
                    self.nodes[a].balance = BF::Balanced;
                    self.reparent(parent, b);
                }
                _ => panic!("invalid balance"),
            }
        }
    }

    /// Return all intervals containing b
    pub fn search(&self, b: u32) -> Vec<(u32, u32)> {
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

    fn rotate(&mut self, d: Traversal, p: usize, c: usize) {
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
            self.sorted_by_last.sort_by_key(|i| std::cmp::Reverse(i.1));
        }
    }

    fn set_child(&mut self, d: Traversal, c: Option<usize>) {
        match d {
            Traversal::Left => self.left = c,
            Traversal::Right => self.right = c,
        }
    }

    fn child(&self, d: Traversal) -> Option<usize> {
        match d {
            Traversal::Left => self.left,
            Traversal::Right => self.right,
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
            -2 => BF::UnbalancedLeft,
            -1 => BF::LeftHeavy,
            0 => BF::Balanced,
            1 => BF::RightHeavy,
            _ => panic!("invalid balance factor"),
        };
    }
}

impl Symmetric for Traversal {
    fn flip(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

impl Symmetric for BF {
    fn flip(self) -> Self {
        match self {
            Self::Balanced => self,
            Self::LeftHeavy => Self::RightHeavy,
            Self::RightHeavy => Self::LeftHeavy,
            Self::UnbalancedLeft => Self::UnbalancedRight,
            Self::UnbalancedRight => Self::UnbalancedLeft,
        }
    }
}

impl Transform for Forward {
    fn t<S>(s: S) -> S
    where
        S: Symmetric,
    {
        s
    }
}

impl Transform for Reverse {
    fn t<S>(s: S) -> S
    where
        S: Symmetric,
    {
        s.flip()
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
    fn test_overlapping_inserts() {
        let mut t = IntervalTree::default();
        t.insert((20, 60));
        t.insert((20, 30));
        t.insert((22, 24));
        t.insert((24, 24));
    }
}

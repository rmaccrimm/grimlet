use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, VecDeque};
use std::fmt::Display;

use slab::Slab;

use super::node::{BF, Direction, IntervalItem, Node};

// Some premature optimization, just for fun. A self-balancing interval tree used to lookup all
// cached function blocks covering a given address. Implemented as an AVL tree.
#[derive(Clone, Debug)]
pub struct IntervalTree<T: IntervalItem> {
    root: Option<usize>,
    nodes: Slab<Node<T>>,
    empty_queue: VecDeque<usize>,
}

impl<T: IntervalItem> IntervalTree<T> {
    pub fn len(&self) -> usize { self.nodes.len() }

    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Insert a new interval. Has no effect if tree already contains the interval.
    pub fn insert(&mut self, ival: (T, T)) {
        let mut cur = self.root;
        let mut par: Option<(usize, Direction)> = None;
        // (Ancestor) root of the subtree for which we'll need to update balance factors, and the
        // path we took from it. Either the last unbalanced node encountered, or root.
        let mut anc = self.root;

        while let Some(n) = cur {
            // Only need to go as far back as the last unbalanced node we saw on our way
            if self.nodes[n].balance != BF::Balanced {
                anc = cur;
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
            par = Some((n, dir));
        }
        let k = self.nodes.insert(Node::new(ival));
        match par {
            Some((p, dir)) => self.set_child(Some((p, dir)), Some(k)),
            None => self.root = Some(k),
        }
        let Some(a) = anc else { return };

        let mut n = k;
        while let Some((p, dir)) = self.get_parent(n) {
            self.nodes[p].balance += dir;
            if p == a {
                break;
            }
            n = p;
        }
        self.rebalance(a);
        self.cleanup();
    }

    /// Return all intervals containing query point q.
    pub fn get_all(&self, q: T) -> Vec<(T, T)> {
        let mut curr = self.root;
        let mut results = vec![];

        while let Some(n) = curr {
            let node = &self.nodes[n];

            match q.cmp(&node.center) {
                Ordering::Equal => {
                    results.extend_from_slice(&node.sorted_by_first);
                    return results;
                }
                Ordering::Less => {
                    let i = node.sorted_by_first.partition_point(|&r| r.0 <= q);
                    results.extend_from_slice(&node.sorted_by_first[0..i]);
                    curr = node.left;
                }
                Ordering::Greater => {
                    let i = node.sorted_by_last.partition_point(|&r| r.1 >= q);
                    results.extend_from_slice(&node.sorted_by_last[0..i]);
                    curr = node.right;
                }
            }
        }
        results
    }

    pub fn contains(&self, ival: (T, T)) -> bool {
        let mut cur = self.root;
        while let Some(n) = cur {
            let dir = if ival.1 < self.nodes[n].center {
                Direction::Left
            } else if ival.0 > self.nodes[n].center {
                Direction::Right
            } else {
                return self.nodes[n].contains(ival);
            };
            cur = self.nodes[n].child(dir);
        }
        false
    }

    pub fn remove(&mut self, ival: (T, T)) -> bool {
        let mut cur = self.root;
        while let Some(n) = cur {
            let dir = if ival.1 < self.nodes[n].center {
                Direction::Left
            } else if ival.0 > self.nodes[n].center {
                Direction::Right
            } else {
                break;
            };
            cur = self.nodes[n].child(dir);
        }
        let Some(rm) = cur else { return false };

        self.nodes[rm].remove(ival);
        if !self.nodes[rm].is_empty() {
            return true;
        }
        self.delete_node(rm);
        self.cleanup();
        true
    }

    /// Remove all entries containing query point q. Has no effect if no matches are found.
    pub fn remove_all(&mut self, q: T) -> Vec<(T, T)> {
        let mut cur = self.root;
        let mut results = vec![];

        while let Some(n) = cur {
            let center = self.nodes[n].center;
            cur = match q.cmp(&center) {
                std::cmp::Ordering::Less => {
                    results.extend_from_slice(&self.nodes[n].remove_start_leq(q));
                    self.nodes[n].left
                }
                std::cmp::Ordering::Equal => {
                    self.empty_queue.push_back(n);
                    break;
                }
                std::cmp::Ordering::Greater => {
                    results.extend_from_slice(&self.nodes[n].remove_end_geq(q));
                    self.nodes[n].right
                }
            };
            if self.nodes[n].is_empty() {
                self.empty_queue.push_back(n);
            }
        }
        self.cleanup();
        results
    }

    pub fn verify(&self) {
        if let Some(r) = self.root {
            self.verify_recursive(r, None, None);
        }
    }

    pub fn print_stats(&self) {
        let mut size_counts: HashMap<usize, usize> = HashMap::new();
        for (_, node) in &self.nodes {
            size_counts
                .entry(node.sorted_by_first.len())
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
        let mut v = size_counts.iter().collect::<Vec<_>>();
        v.sort_by_key(|k| Reverse(k.1));
        for x in v {
            println!("{x:?}");
        }
    }

    fn cleanup(&mut self) {
        while let Some(n) = self.empty_queue.pop_front() {
            if self.nodes.contains(n) {
                self.delete_node(n);
            }
        }
    }

    fn delete_node(&mut self, rm: usize) {
        let rm_parent = self.get_parent(rm);
        let removed = self.nodes.remove(rm);
        // Stores the point from which we need to do re-balancing, continuing up towards the root
        let mut rebal = match removed.left {
            None => {
                self.set_child(rm_parent, removed.right);
                rm_parent
            }
            Some(l) => {
                if self.nodes[l].right.is_none() {
                    // l Just shifts up to replace removed, and needs to be check for rebalancing
                    self.set_child(Some((l, Direction::Right)), removed.right);
                    self.set_child(rm_parent, Some(l));
                    // this is just it's "initial" balance, not final
                    self.nodes[l].balance = removed.balance;
                    Some((l, Direction::Left))
                } else {
                    let mut cur = l;
                    let pred = loop {
                        match self.nodes[cur].right {
                            Some(r) => cur = r,
                            None => break cur,
                        }
                    };
                    debug_assert!(self.nodes[pred].right.is_none(), "invalid predecessor");
                    let pred_parent = self.get_parent(pred);
                    self.set_child(pred_parent, self.nodes[pred].left);
                    self.set_child(Some((pred, Direction::Left)), removed.left);
                    self.set_child(Some((pred, Direction::Right)), removed.right);
                    self.nodes[pred].balance = removed.balance;
                    self.set_child(rm_parent, Some(pred));

                    let mut cur = pred_parent.unwrap().0;
                    while cur != pred {
                        self.take_overlapping(pred, cur);
                        cur = self.nodes[cur].parent.unwrap();
                    }
                    pred_parent
                }
            }
        };

        while let Some((n, dir)) = rebal {
            // Height decreased in the direction we travelled
            self.nodes[n].balance += dir.flip();
            match &self.nodes[n].balance {
                // tree height decreased (heavy -> balanced)
                BF::Balanced => rebal = self.get_parent(n),
                // tree height did not decrease (balanced -> heavy)
                BF::Heavy(_) => break,
                // tree height may or may not decrease
                BF::Unbalanced(_) => {
                    let (r, decreased) = self.rebalance(n);
                    if !decreased {
                        break;
                    }
                    rebal = self.get_parent(r);
                }
            }
        }
    }

    // a: root of the sub-tree which needs to be re-balanced
    // Returns new root of the subtree and true if the height of the sub-tree decreased
    fn rebalance(&mut self, a: usize) -> (usize, bool) {
        let parent = self.get_parent(a);
        if let BF::Unbalanced(dir) = self.nodes[a].balance {
            let b = self.nodes[a].child(dir).unwrap();
            let b_bal = self.nodes[b].balance;
            if let BF::Unbalanced(_) = b_bal {
                panic!("invalid balance");
            }
            if let BF::Heavy(dir_bc) = b_bal
                && dir_bc != dir
            {
                let c = self.nodes[b].child(dir.flip()).unwrap();
                self.rotate(dir, b, c);
                self.set_child(Some((a, dir)), Some(c));
                self.rotate(dir.flip(), a, c);
                self.set_child(parent, Some(c));

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
                    BF::Unbalanced(_) => panic!("invalid balance"),
                }
                self.nodes[c].balance = BF::Balanced;
                (c, true)
            } else {
                // either unbalanced in the same direction or balanced
                self.rotate(dir.flip(), a, b);
                self.set_child(parent, Some(b));
                if b_bal == BF::Balanced {
                    self.nodes[a].balance = BF::Heavy(dir);
                    self.nodes[b].balance = BF::Heavy(dir.flip());
                    (b, false)
                } else {
                    self.nodes[b].balance = BF::Balanced;
                    self.nodes[a].balance = BF::Balanced;
                    (b, true)
                }
            }
        } else {
            (a, false)
        }
    }

    fn get_parent(&self, n: usize) -> Option<(usize, Direction)> {
        if let Some(p) = self.nodes[n].parent {
            let dir = if self.nodes[p].left == Some(n) {
                Direction::Left
            } else {
                Direction::Right
            };
            Some((p, dir))
        } else {
            None
        }
    }

    // If parent is None, updates the root
    fn set_child(&mut self, parent: Option<(usize, Direction)>, child: Option<usize>) {
        if let Some(c) = child {
            self.nodes[c].parent = parent.map(|p| p.0);
        }
        match parent {
            Some((p, Direction::Left)) => self.nodes[p].left = child,
            Some((p, Direction::Right)) => self.nodes[p].right = child,
            None => self.root = child,
        }
    }

    fn rotate(&mut self, d: Direction, p: usize, c: usize) {
        debug_assert!(self.nodes[p].child(d.flip()) == Some(c), "invalid rotation");
        let c_child = self.nodes[c].child(d);
        self.set_child(Some((p, d.flip())), c_child);
        self.set_child(Some((c, d)), Some(p));
        if self.root == Some(p) {
            self.root = Some(c);
        }
        self.take_overlapping(c, p);
    }

    fn take_overlapping(&mut self, p: usize, c: usize) {
        let center = self.nodes[p].center;
        for ival in self.nodes[c].sorted_by_first.clone() {
            if ival.0 <= center && ival.1 >= center {
                self.nodes[c].remove(ival);
                self.nodes[p].add(ival);
            }
        }
        if self.nodes[c].sorted_by_first.is_empty() {
            self.empty_queue.push_back(c);
        }
    }

    fn verify_recursive(&self, n: usize, left: Option<T>, right: Option<T>) -> i32 {
        let node = &self.nodes[n];
        assert!(
            !matches!(node.balance, BF::Unbalanced(_)),
            "node {n} is not balanced",
        );
        if let Some(limit) = left {
            for ival in &node.sorted_by_first {
                assert!(
                    ival.0 > limit,
                    "node {}: {} not greater than left limit {} ",
                    n,
                    ival.0,
                    limit
                );
            }
        }
        if let Some(limit) = right {
            for ival in &node.sorted_by_first {
                assert!(
                    ival.1 < limit,
                    "node {}: {} not greater than right limit {} ",
                    n,
                    ival.1,
                    limit
                );
            }
        }
        let height_l = if let Some(l) = node.left {
            assert_eq!(self.nodes[l].parent, Some(n), "node {l}: incorrect parent");
            self.verify_recursive(l, left, Some(node.center))
        } else {
            0
        };
        let height_r = if let Some(r) = node.right {
            assert_eq!(self.nodes[r].parent, Some(n), "node {r}: incorrect parent");
            self.verify_recursive(r, Some(node.center), right)
        } else {
            0
        };
        assert_eq!(
            BF::try_from((height_l, height_r)).unwrap(),
            node.balance,
            "node {n}: incorrect balance"
        );

        std::cmp::max(height_l, height_r) + 1
    }
}

impl<T: IntervalItem> Default for IntervalTree<T> {
    fn default() -> Self {
        Self {
            root: None,
            nodes: Slab::new(),
            empty_queue: VecDeque::new(),
        }
    }
}

impl<T: IntervalItem> Display for IntervalTree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "flowchart TD")?;
        let mut q = VecDeque::new();
        match self.root {
            None => {
                return Ok(());
            }
            Some(n) => q.push_back(n),
        }
        while let Some(n) = q.pop_front() {
            let node = &self.nodes[n];
            writeln!(f, "{n}(\"{node}\")")?;
            if let Some(l) = node.left {
                writeln!(f, "{n}-->{l}")?;
                q.push_back(l);
            }
            if let Some(r) = node.right {
                writeln!(f, "{n}-->{r}")?;
                q.push_back(r);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_links<T: IntervalItem>(
        t: &IntervalTree<T>,
        i: usize,
        l: Option<usize>,
        r: Option<usize>,
    ) {
        let Node { left, right, .. } = t.nodes[i];
        assert_eq!((left, right), (l, r), "node {i}");
    }

    #[test]
    fn test_in_order_inserts() {
        let mut t = IntervalTree::default();
        for i in 0..7 {
            t.insert((i, i));
        }
        assert_eq!(t.root, Some(3));
        t.verify();

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
        t.verify();

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
        t.verify();

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

        assert_eq!(t.root, Some(0));
        t.verify();

        assert_eq!(t.get_all(8), vec![(-20, 20), (5, 15), (1, 9)]);
    }

    #[test]
    fn test_bst_delete_leaf() {
        let mut t = IntervalTree::default();
        for i in 0..15 {
            t.insert((i, i));
        }
        t.remove((0, 0));
        assert_eq!(t.root, Some(7));
        t.verify();
        check_links(&t, 1, None, Some(2));

        t.remove((6, 6));
        assert_eq!(t.root, Some(7));
        t.verify();
        check_links(&t, 5, Some(4), None);

        let mut t = IntervalTree::default();
        t.insert((10, 10));
        t.remove((10, 10));
        assert_eq!(t.root, None);
        assert_eq!(t.nodes.len(), 0);
    }

    #[test]
    fn test_bst_delete_no_left_child() {
        let mut t = IntervalTree::default();
        t.insert((0, 0));
        t.insert((1, 1));
        t.remove((0, 0));
        assert!(!t.nodes.contains(0));
        assert_eq!(t.root, Some(1));
        t.verify();
        check_links(&t, 1, None, None);

        let mut t = IntervalTree::default();
        for i in 0..15 {
            t.insert((i, i));
        }
        t.remove((8, 8));
        t.remove((9, 9));
        assert!(!t.nodes.contains(8));
        assert!(!t.nodes.contains(9));
        assert_eq!(t.root, Some(7));
        t.verify();
        check_links(&t, 11, Some(10), Some(13));
        check_links(&t, 10, None, None);
    }

    #[test]
    fn test_bst_delete_with_left_without_right() {
        let mut t = IntervalTree::default();
        for i in 0..3 {
            t.insert((i, i));
        }
        t.remove((1, 1));
        assert!(!t.nodes.contains(1));
        assert_eq!(t.root, Some(0));
        t.verify();
        check_links(&t, 0, None, Some(2));
        check_links(&t, 2, None, None);

        let mut t = IntervalTree::default();
        for i in 0..15 {
            t.insert((i, i));
        }
        t.remove((10, 10));
        t.remove((11, 11));
        assert_eq!(t.root, Some(7));
        t.verify();
        assert!(!t.nodes.contains(10));
        assert!(!t.nodes.contains(11));
        check_links(&t, 7, Some(3), Some(9));
        check_links(&t, 8, None, None);
        check_links(&t, 9, Some(8), Some(13));
    }

    #[test]
    fn test_bst_delete_with_left_with_right() {
        let mut t = IntervalTree::default();
        for i in [0, 1, 2, 3, 4, 5, 6] {
            t.insert((i, i));
        }
        t.remove((3, 3));
        assert!(!t.nodes.contains(3));
        assert_eq!(t.root, Some(2));
        t.verify();
        check_links(&t, 1, Some(0), None);
        check_links(&t, 2, Some(1), Some(5));

        let mut t = IntervalTree::default();
        for i in 0..15 {
            t.insert((i, i));
        }
        t.remove((11, 11));
        assert_eq!(t.root, Some(7));
        t.verify();
        assert!(!t.nodes.contains(11));
        check_links(&t, 7, Some(3), Some(10));
        check_links(&t, 9, Some(8), None);
        check_links(&t, 10, Some(9), Some(13));
    }

    #[test]
    fn test_avl_delete_multiple_rebalances() {
        //       __4_
        //      /    \
        //     1      9
        //    / \    / \
        //   0   3  7  10
        //      /  / \   \
        //     2  5   8  11
        //         \
        //          6
        let mut t = IntervalTree::default();
        for i in [4, 1, 9, 0, 3, 7, 10, 2, 5, 8, 11, 6] {
            t.insert((i, i));
        }
        println!("{t}");
        assert_eq!(t.root, Some(0));
        t.verify();

        t.remove((0, 0));
        assert_eq!(t.root, Some(5));
        t.verify();
    }

    #[test]
    #[should_panic(expected = "6 not greater than right limit 5")]
    fn test_verify_invalid_bst() {
        let mut nodes = Slab::new();
        nodes.insert(Node {
            center: 5,
            sorted_by_first: vec![(5, 5)],
            sorted_by_last: vec![(5, 5)],
            left: Some(1),
            right: Some(3),
            parent: None,
            balance: BF::Heavy(Direction::Left),
        });
        nodes.insert(Node {
            center: 3,
            sorted_by_first: vec![(3, 3)],
            sorted_by_last: vec![(3, 3)],
            left: None,
            right: Some(2),
            parent: Some(0),
            balance: BF::Heavy(Direction::Right),
        });
        nodes.insert(Node {
            center: 6,
            sorted_by_first: vec![(6, 6)],
            sorted_by_last: vec![(6, 6)],
            left: None,
            right: None,
            parent: Some(1),
            balance: BF::Balanced,
        });
        nodes.insert(Node {
            center: 7,
            sorted_by_first: vec![(7, 7)],
            sorted_by_last: vec![(7, 7)],
            left: None,
            right: None,
            parent: Some(0),
            balance: BF::Balanced,
        });
        let t = IntervalTree {
            root: Some(0),
            nodes,
            empty_queue: VecDeque::new(),
        };
        t.verify();
    }

    #[test]
    #[should_panic(expected = "incorrect balance")]
    fn test_verify_incorrect_balance() {
        let mut nodes = Slab::new();
        nodes.insert(Node {
            center: 3,
            sorted_by_first: vec![(3, 3)],
            sorted_by_last: vec![(3, 3)],
            left: None,
            right: Some(1),
            parent: None,
            balance: BF::Heavy(Direction::Right),
        });
        nodes.insert(Node {
            center: 5,
            sorted_by_first: vec![(5, 5)],
            sorted_by_last: vec![(5, 5)],
            left: Some(2),
            right: Some(3),
            parent: Some(0),
            balance: BF::Balanced,
        });
        nodes.insert(Node {
            center: 4,
            sorted_by_first: vec![(4, 4)],
            sorted_by_last: vec![(4, 4)],
            left: None,
            right: None,
            parent: Some(1),
            balance: BF::Balanced,
        });
        nodes.insert(Node {
            center: 7,
            sorted_by_first: vec![(7, 7)],
            sorted_by_last: vec![(7, 7)],
            left: None,
            right: None,
            parent: Some(1),
            balance: BF::Balanced,
        });
        let t = IntervalTree {
            root: Some(0),
            nodes,
            ..Default::default()
        };
        t.verify();
    }

    #[test]
    #[should_panic(expected = "node 3 is not balanced")]
    fn test_verify_invalid_avl_balance() {
        let mut nodes = Slab::new();
        nodes.insert(Node {
            center: 1,
            sorted_by_first: vec![(1, 1)],
            sorted_by_last: vec![(1, 1)],
            left: None,
            right: None,
            parent: Some(1),
            balance: BF::Balanced,
        });
        nodes.insert(Node {
            center: 2,
            sorted_by_first: vec![(2, 2)],
            sorted_by_last: vec![(2, 3)],
            left: Some(0),
            right: None,
            parent: Some(2),
            balance: BF::Heavy(Direction::Left),
        });
        nodes.insert(Node {
            center: 3,
            sorted_by_first: vec![(3, 3)],
            sorted_by_last: vec![(3, 3)],
            left: Some(1),
            right: Some(3),
            parent: None,
            balance: BF::Heavy(Direction::Right),
        });
        nodes.insert(Node {
            center: 7,
            sorted_by_first: vec![(7, 7)],
            sorted_by_last: vec![(7, 7)],
            left: Some(4),
            right: None,
            parent: Some(2),
            balance: BF::Unbalanced(Direction::Left),
        });
        nodes.insert(Node {
            center: 4,
            sorted_by_first: vec![(4, 4)],
            sorted_by_last: vec![(4, 4)],
            left: None,
            right: Some(5),
            parent: Some(3),
            balance: BF::Heavy(Direction::Right),
        });
        nodes.insert(Node {
            center: 5,
            sorted_by_first: vec![(5, 5)],
            sorted_by_last: vec![(5, 5)],
            left: None,
            right: None,
            parent: Some(4),
            balance: BF::Balanced,
        });
        let t = IntervalTree {
            root: Some(2),
            nodes,
            ..Default::default()
        };
        t.verify();
    }

    #[test]
    fn print_size_of_node() {
        println!("{}", size_of::<Node<i32>>());
    }

    #[test]
    fn test_rotation_resulting_in_delete() {
        let mut t = IntervalTree::default();
        t.insert((-100, 100)); // <- should get absorbed into 1 once we delete
        t.insert((10, 20));
        t.insert((30, 40));
        assert_eq!(t.nodes.len(), 2);
        assert_eq!(t.root, Some(1));
        t.verify();
    }

    #[test]
    fn test_cascading_deletes() {
        let mut t = IntervalTree::default();
        t.insert((601, 799)); // 7
        t.insert((-99, 699)); // 3, 300 - 1000, 300 + 1000
        t.insert((1001, 1199)); // 11
        t.insert((1, 199)); // 1
        t.insert((301, 699)); // 5
        t.insert((801, 999)); // 9
        t.insert((1201, 1399)); // 13
        t.insert((0, 0)); // 0
        t.insert((101, 299)); // 2
        t.insert((301, 499)); // 4, 
        t.insert((501, 699)); // 6
        t.insert((701, 899)); // 8
        t.insert((901, 1099)); // 10
        t.insert((1101, 1299)); // 12 
        t.insert((1301, 1499)); // 14

        // When 7 is deleted, 6 gets promoted to the root and absorbs 3 & 5
        t.remove((601, 799));
        assert_eq!(t.root, Some(10));
        assert_eq!(t.nodes.len(), 12);
        assert_eq!(
            t.nodes[t.root.unwrap()].sorted_by_first,
            vec![(-99, 699), (301, 699), (501, 699)]
        );
        t.verify();
    }

    // lots of overlapping intervals, but all unique start points
    fn build_overlapping_tree() -> IntervalTree<i32> {
        let mut t = IntervalTree::default();
        t.insert((-200, 200));
        t.insert((50, 60));
        t.insert((56, 110));
        t.insert((51, 150));
        t.insert((25, 50));
        t.insert((-210, -190));
        t.insert((-1, 140));
        t.insert((20, 30));
        t.insert((-75, -50));
        t.insert((60, 70));
        t.insert((1, 10));
        t
    }

    macro_rules! remove_all_tests {
        ($($name:ident: $q:expr, $expected:expr, $expected_len:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let mut t = build_overlapping_tree();
                    let removed = t.remove_all($q);
                    println!("{t}");
                    assert_eq!(removed, $expected, "removed elements");
                    assert_eq!(t.len(), $expected_len, "tree len");
                    for r in removed {
                        assert!(!t.contains(r), "{r:?}: still in tree");
                    }
                }
            )*
        };
    }

    remove_all_tests!(
        test_remove_all_1: 0, vec![(-200, 200), (-1, 140)], 7,
    );
}

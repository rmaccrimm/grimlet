use std::fmt::Display;
use std::ops::{Add, AddAssign};

use num::integer::{Average, Integer};

// Any integer type
pub trait IntervalItem = Integer + Average + Copy + Display;

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct Node<T: IntervalItem> {
    pub center: T,
    pub sorted_by_first: Vec<(T, T)>,
    pub sorted_by_last: Vec<(T, T)>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub parent: Option<usize>,
    pub balance: BF,
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

impl<T: IntervalItem> Node<T> {
    pub fn new(ival: (T, T)) -> Self {
        debug_assert!(ival.0 <= ival.1, "interval must be sorted ascending");
        Self {
            center: ival.0.average_floor(&ival.1),
            sorted_by_first: vec![ival],
            sorted_by_last: vec![ival],
            left: None,
            right: None,
            parent: None,
            balance: BF::Balanced,
        }
    }

    pub fn add(&mut self, ival: (T, T)) {
        debug_assert!(ival.0 <= ival.1, "interval must be sorted ascending");
        if !self.sorted_by_first.contains(&ival) {
            self.sorted_by_first.push(ival);
            self.sorted_by_first.sort_by_key(|i| i.0);
            self.sorted_by_last.push(ival);
            self.sorted_by_last.sort_by_key(|i| std::cmp::Reverse(i.1));
        }
    }

    pub fn remove(&mut self, ival: (T, T)) {
        debug_assert!(ival.0 <= ival.1, "interval must be sorted ascending");
        self.sorted_by_first.retain(|i| *i != ival);
        self.sorted_by_last.retain(|i| *i != ival);
    }

    pub fn remove_start_leq(&mut self, q: T) {
        let i = self.sorted_by_first.partition_point(|&r| r.0 <= q);
        if i == 0 {
            return;
        }
        self.sorted_by_first = Vec::from(&self.sorted_by_first[i..]);
        self.sorted_by_last = self.sorted_by_first.clone();
        self.sorted_by_last.sort_by_key(|i| std::cmp::Reverse(i.1));
    }

    pub fn remove_end_geq(&mut self, q: T) {
        let i = self.sorted_by_last.partition_point(|&r| r.1 >= q);
        if i == 0 {
            return;
        }
        self.sorted_by_last = Vec::from(&self.sorted_by_last[i..]);
        self.sorted_by_first = self.sorted_by_last.clone();
        self.sorted_by_first.sort_by_key(|i| i.0);
    }

    pub fn is_empty(&mut self) -> bool { self.sorted_by_first.is_empty() }

    pub fn child(&self, d: Direction) -> Option<usize> {
        match d {
            Direction::Left => self.left,
            Direction::Right => self.right,
        }
    }
}

impl<T: Average + Copy + PartialEq + Display> Display for Node<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\\n", self.center)?;
        let ival = self.sorted_by_first[0];
        write!(f, "[{}, {}]", ival.0, ival.1)?;
        for ival in &self.sorted_by_first[1..] {
            write!(f, ", [{}, {}]", ival.0, ival.1)?;
        }
        write!(f, "\\n{}", self.balance)?;
        Ok(())
    }
}

impl Direction {
    pub fn flip(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

impl Display for BF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BF::Balanced => write!(f, "-"),
            BF::Heavy(Direction::Left) => write!(f, "<-"),
            BF::Heavy(Direction::Right) => write!(f, "->"),
            BF::Unbalanced(Direction::Left) => write!(f, "<--"),
            BF::Unbalanced(Direction::Right) => write!(f, "-->"),
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
                if d1 == d2 {
                    panic!("balance out of bounds");
                } else {
                    BF::Heavy(d1)
                }
            }
        }
    }
}

impl AddAssign<Direction> for BF {
    fn add_assign(&mut self, rhs: Direction) { *self = self.add(rhs); }
}

impl TryFrom<(i32, i32)> for BF {
    type Error = String;

    fn try_from(heights: (i32, i32)) -> std::result::Result<Self, Self::Error> {
        let (l, r) = heights;
        match r - l {
            0 => Ok(BF::Balanced),
            1 => Ok(BF::Heavy(Direction::Right)),
            -1 => Ok(BF::Heavy(Direction::Left)),
            2 => Ok(BF::Unbalanced(Direction::Right)),
            -2 => Ok(BF::Unbalanced(Direction::Left)),
            _ => Err("Heights differ by more than 2".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_interval() {
        let node = Node::new((10, 10));
        assert_eq!(node.sorted_by_first, vec![(10, 10)]);
        assert_eq!(node.sorted_by_last, vec![(10, 10)]);
    }

    #[test]
    #[should_panic(expected = "interval must be sorted ascending")]
    fn test_new_invalid_interval() { Node::new((1, 0)); }

    #[test]
    #[should_panic(expected = "interval must be sorted ascending")]
    fn test_add_invalid_interval() {
        let mut node = Node::new((10, 20));
        node.add((3, -4));
    }

    #[test]
    #[should_panic(expected = "interval must be sorted ascending")]
    fn test_remove_invalid_interval() {
        let mut node = Node::new((10, 10));
        node.remove((1, 0));
    }

    #[test]
    fn test_add_in_order() {
        let mut node = Node::new((50, 60));
        node.add((10, 20));
        node.add((30, 40));
        node.add((70, 80));

        assert_eq!(
            node.sorted_by_first,
            vec![(10, 20), (30, 40), (50, 60), (70, 80)]
        );
        assert_eq!(
            node.sorted_by_last,
            vec![(70, 80), (50, 60), (30, 40), (10, 20)]
        );
    }

    #[test]
    fn test_add_maintains_order() {
        let mut node = Node::new((50, 50));
        node.add((20, 90));
        node.add((10, 50));
        node.add((40, 70));
        node.add((25, 400));
        node.add((60, 60));

        assert_eq!(
            node.sorted_by_first,
            vec![(10, 50), (20, 90), (25, 400), (40, 70), (50, 50), (60, 60)]
        );
        assert_eq!(
            node.sorted_by_last,
            // By insert order when there's a tie?
            vec![(25, 400), (20, 90), (40, 70), (60, 60), (50, 50), (10, 50)]
        );
    }

    #[test]
    fn test_add_duplicate_not_added() {
        let mut node = Node::new((10, 20));
        node.add((10, 20));
        node.add((10, 20));

        assert_eq!(node.sorted_by_first, vec![(10, 20)]);
        assert_eq!(node.sorted_by_last, vec![(10, 20)]);
    }

    #[test]
    fn test_remove_only_item() {
        let mut node = Node::new((10, 20));
        node.remove((10, 20));
        assert_eq!(node.sorted_by_first, vec![]);
        assert_eq!(node.sorted_by_last, vec![]);
    }

    #[test]
    fn test_remove_maintains_sort() {
        let mut node = Node::new((10, 20));
        node.add((20, 90));
        node.add((30, 40));
        node.add((0, 45));
        node.add((40, 50));

        node.remove((20, 30));
        node.remove((40, 50));

        assert_eq!(
            node.sorted_by_first,
            vec![(0, 45), (10, 20), (20, 90), (30, 40)]
        );
        assert_eq!(
            node.sorted_by_last,
            vec![(20, 90), (0, 45), (30, 40), (10, 20)]
        );
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut node = Node::new((10, 20));
        node.add((30, 40));
        node.remove((50, 60));
        assert_eq!(node.sorted_by_first, vec![(10, 20), (30, 40)]);
        assert_eq!(node.sorted_by_last, vec![(30, 40), (10, 20)]);
    }

    #[test]
    fn test_remove_start_leq_maintains_sort() {
        let mut node = Node::new((10, 20));
        node.add((20, 100));
        node.add((30, 40));
        node.add((50, 60));
        node.add((35, 40));
        node.add((36, 70));
        node.remove_start_leq(35);
        assert_eq!(node.sorted_by_first, vec![(36, 70), (50, 60)]);
        assert_eq!(node.sorted_by_last, vec![(36, 70), (50, 60)]);
    }

    #[test]
    fn test_remove_start_leq_none() {
        let mut node = Node::new((10, 20));
        node.add((30, 40));
        node.add((50, 60));
        node.remove_start_leq(0);
        assert_eq!(node.sorted_by_first, vec![(10, 20), (30, 40), (50, 60)]);
    }

    #[test]
    fn test_remove_start_leq_all() {
        let mut node = Node::new((10, 20));
        node.add((30, 40));
        node.add((100, 100));
        node.remove_start_leq(100);
        assert!(node.is_empty());
        assert_eq!(node.sorted_by_first, vec![]);
        assert_eq!(node.sorted_by_last, vec![]);
    }

    #[test]
    fn test_remove_end_geq_maintains_sort() {
        let mut node = Node::new((10, 20));
        node.add((20, 100));
        node.add((30, 40));
        node.add((50, 60));
        node.add((35, 69));
        node.add((36, 70));
        node.remove_end_geq(70);
        assert_eq!(
            node.sorted_by_first,
            vec![(10, 20), (30, 40), (35, 69), (50, 60)]
        );
        assert_eq!(
            node.sorted_by_last,
            vec![(35, 69), (50, 60), (30, 40), (10, 20)]
        );
    }

    #[test]
    fn test_remove_end_geq_none() {
        let mut node = Node::new((10, 20));
        node.add((30, 40));
        node.add((50, 60));
        node.remove_end_geq(61);
        assert_eq!(node.sorted_by_first, vec![(10, 20), (30, 40), (50, 60)]);
    }

    #[test]
    fn test_remove_end_geq_all() {
        let mut node = Node::new((10, 20));
        node.add((30, 40));
        node.add((100, 100));
        node.remove_end_geq(0);
        assert!(node.is_empty());
        assert_eq!(node.sorted_by_first, vec![]);
        assert_eq!(node.sorted_by_last, vec![]);
    }
}

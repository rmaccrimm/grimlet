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
        if !self.sorted_by_first.contains(&ival) {
            self.sorted_by_first.push(ival);
            self.sorted_by_first.sort_by_key(|i| i.0);
            self.sorted_by_last.push(ival);
            self.sorted_by_last.sort_by_key(|i| std::cmp::Reverse(i.1));
        }
    }

    pub fn remove(&mut self, ival: (T, T)) -> bool {
        self.sorted_by_first.retain(|i| *i != ival);
        self.sorted_by_last.retain(|i| *i != ival);
        self.sorted_by_first.is_empty()
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

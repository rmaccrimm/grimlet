use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc::Receiver;

use super::CompiledFunction;
use crate::utils::interval_tree::IntervalTree;

/// Wrapper around a hashmap that also maintains a count of how often each entry is accessed
/// (I'm thinking this will be useful to clear up memory in the future by removing infrequently
/// accessed blocks).
pub struct FunctionCache<'ctx> {
    map: HashMap<u32, CacheEntry<'ctx>>,
    // Shared with `MemoryManager`
    interval_tree: Rc<RefCell<IntervalTree<u32>>>,
    rx: Receiver<u32>,
}

struct CacheEntry<'ctx> {
    func: CompiledFunction<'ctx>,
    hits: u32,
}

impl<'ctx> FunctionCache<'ctx> {
    pub fn new(interval_tree: Rc<RefCell<IntervalTree<u32>>>, rx: Receiver<u32>) -> Self {
        Self {
            map: HashMap::new(),
            interval_tree,
            rx,
        }
    }

    pub fn insert(&mut self, start: u32, end: u32, v: CompiledFunction<'ctx>) {
        let res = self.map.insert(start, CacheEntry { func: v, hits: 0 });
        debug_assert!(res.is_none());
        self.interval_tree.borrow_mut().insert((start, end));
    }

    pub fn get(&mut self, k: u32) -> Option<&CompiledFunction<'ctx>> {
        if let Some(e) = self.map.get_mut(&k) {
            e.hits += 1;
            Some(&e.func)
        } else {
            None
        }
    }

    pub fn update(&mut self) {
        for addr in &self.rx {
            self.map
                .remove(&addr)
                .expect("No function cached for {start}");
        }
    }
}

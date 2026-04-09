#![allow(dead_code)]

use std::collections::HashMap;

use super::CompiledFunction;

/// Wrapper around a hashmap that also maintains a count of how often each entry is accessed
/// (I'm thinking this will be useful to clear up memory in the future by removing infrequently
/// accessed blocks).
#[derive(Default)]
pub struct FunctionCache<'ctx> {
    map: HashMap<u32, CacheEntry<'ctx>>,
}

struct CacheEntry<'ctx> {
    func: CompiledFunction<'ctx>,
    end: u32,
    hits: u32,
}

impl<'ctx> FunctionCache<'ctx> {
    pub fn insert(&mut self, start: u32, end: u32, v: CompiledFunction<'ctx>) {
        let res = self.map.insert(
            start,
            CacheEntry {
                func: v,
                hits: 0,
                end,
            },
        );
        debug_assert!(res.is_none());
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
        // Handle pending invalidations
    }
}

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

struct CacheEntry<'ctx> {
    func: CompiledFunction<'ctx>,
    hits: usize,
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

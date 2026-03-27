pub mod builder;

use std::collections::HashMap;

use inkwell::execution_engine::JitFunction;

use crate::arm::state::ArmState;

// JIT'd funxtions take a pointer to the guest machine state and return the number of (emulated)
// CPU cycles it took to execute
pub type CompiledFunction<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState) -> u32>;

pub type FunctionCache<'ctx> = HashMap<usize, CompiledFunction<'ctx>>;

pub mod codegen;
pub mod state;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, BasicValueEnum, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use codegen::CodeGen;
use state::GuestState;
use std::error::Error;

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let load = codegen
        .jit_compile_load()
        .ok_or("Unable to JIT compile `load_state`")?;

    let mut x: i32 = 10;

    println!("Calling func...");
    unsafe {
        load.call(&mut x);
    }
    println!("x: {}", x);

    Ok(())
}

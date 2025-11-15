pub mod codegen;
pub mod state;

use capstone::arch::BuildsCapstone;
use capstone::{Capstone, InsnDetail, arch};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, BasicValueEnum, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use codegen::CodeGen;
use state::GuestState;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};

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

    let bytes = std::fs::read("gba_bios.bin")?;
    println!("{}", bytes.len());

    let cs = Capstone::new()
        .arm()
        .mode(arch::arm::ArchMode::Thumb)
        .detail(true)
        .build()
        .unwrap();

    let insns = cs.disasm_all(&bytes[0x11c..], 0x11c).unwrap();
    for insn in insns.as_ref() {
        println!("{}", insn);
        // let detail = cs.insn_detail(insn).unwrap();
        // println!("{:?}", detail.arch_detail().operands());
    }

    Ok(())
}

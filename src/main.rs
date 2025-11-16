pub mod codegen;
pub mod state;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use codegen::LlvmComponents;
use inkwell::builder::Builder;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;

use codegen::Compiler;
use state::GuestState;
use std::env;

use inkwell::context::Context;
use std::fs::File;
use std::io::Read;

/// Am I sticking with this name?
struct Grimlet<'a> {
    cs: Capstone,
    ll: LlvmComponents<'a>,
    state: GuestState,
    context: Context,
}

impl<'a> Grimlet<'a> {
    pub fn new(ll: LlvmComponents<'a>, bios_path: &str) -> Result<Self> {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;

        let mut state = GuestState::new();

        let mut f = File::open(bios_path)?;
        f.read_exact(&mut *state.mem)?;

        let context = Context::create();

        Ok(Self {
            cs,
            ll,
            state,
            context,
        })
    }

    pub fn run(&mut self) {
        let mut compiler = Compiler::new();
        compiler.compile_test(&mut self.ll);
        unsafe {
            self.ll.function.clone().unwrap().call(&mut self.state);
        }
    }
}

fn main() -> Result<()> {
    let context = Context::create();
    let ll = LlvmComponents::new(&context);

    let bios_path = env::args().into_iter().next().unwrap();
    let mut grimlet = Grimlet::new(ll, &bios_path)?;
    grimlet.state.regs[0] = 12;

    println!("{:?}", grimlet.state.regs);
    grimlet.run();

    println!("{:?}", grimlet.state.regs);
    Ok(())
}

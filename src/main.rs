pub mod codegen;
pub mod state;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use std::sync::Arc;

use codegen::Compiler;
use state::GuestState;
use std::env;

use inkwell::context::Context;
use std::fs::File;
use std::io::Read;

/// Am I sticking with this name?
struct Grimlet<'ctx> {
    compiler: Compiler<'ctx>,
    cs: Capstone,
    state: GuestState,
}

impl<'ctx> Grimlet<'ctx> {
    pub fn new(context: &'ctx Context, bios_path: &str) -> Result<Self> {
        let compiler = Compiler::new(context)?;
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;
        let mut state = GuestState::new();

        let mut f = File::open(bios_path)?;
        f.read_exact(&mut *state.mem)?;

        Ok(Self {
            compiler,
            cs,
            state,
        })
    }
}

fn main() -> Result<()> {
    let context = Arc::new(Context::create());
    let bios_path = env::args().into_iter().next().unwrap();
    let mut grimlet = Grimlet::new(&context, &bios_path)?;
    grimlet.state.regs[0] = 12;
    let f = grimlet.compiler.compile_test().unwrap();
    println!("{:?}", grimlet.state.regs);
    unsafe {
        f.call(&mut grimlet.state);
    }
    println!("{:?}", grimlet.state.regs);
    Ok(())
}

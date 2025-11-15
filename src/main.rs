pub mod codegen;
pub mod state;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;

use codegen::CodeGen;
use state::GuestState;

use inkwell::context::Context;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};

/// Am I sticking with this name?
struct Grimlet<'ctx> {
    codegen: CodeGen<'ctx>,
    cs: Capstone,
    state: GuestState,
}

impl<'ctx> Grimlet<'ctx> {
    pub fn new(context: &'ctx Context, bios_path: &str) -> Result<Self> {
        let codegen = CodeGen::new(context)?;
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;
        let mut state = GuestState::new();

        let mut f = File::open(bios_path)?;
        f.read_exact(&mut state.mem.bios)?;

        Ok(Self { codegen, cs, state })
    }

    pub fn run(&self) {
        println!("Running!")
    }
}

fn main() -> Result<()> {
    let context = Context::create();
    let grimlet = Grimlet::new(&context, "gba_bios.bin")?;
    grimlet.run();
    Ok(())
}

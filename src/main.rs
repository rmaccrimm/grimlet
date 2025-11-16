pub mod codegen;
pub mod state;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use codegen::LlvmComponents;

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
        Ok(Self { cs, ll, state })
    }

    pub fn run(&mut self) {
        let mut compiler = Compiler::new();
        compiler.compile(&mut self.ll);
        println!("");
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
    for i in 0..17 {
        grimlet.state.regs[i] = i as u32;
    }

    println!("{:?}", grimlet.state.regs);
    grimlet.run();

    println!("{:?}", grimlet.state.regs);
    Ok(())
}

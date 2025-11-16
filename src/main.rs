pub mod cpu;
pub mod jit;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use jit::Compiler;

use cpu::ArmState;
use std::env;

use inkwell::context::Context;
use std::fs::File;
use std::io::Read;

/// Am I sticking with this name?
struct Grimlet<'a> {
    cs: Capstone,
    compiler: Compiler<'a>,
    state: ArmState,
}

impl<'a> Grimlet<'a> {
    pub fn new(compiler: Compiler<'a>, bios_path: &str) -> Result<Self> {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;
        let mut state = ArmState::new();
        let mut f = File::open(bios_path)?;
        f.read_exact(&mut *state.mem)?;
        Ok(Self {
            cs,
            compiler,
            state,
        })
    }

    pub fn run(&mut self) {
        self.compiler.compile(0);
        let f = self.compiler.func_cache.get(&0).unwrap();
        unsafe {
            f.call(&mut self.state);
        }
    }
}

fn main() -> Result<()> {
    let context = Context::create();
    let compiler = Compiler::new(&context);

    let bios_path = env::args().into_iter().next().unwrap();
    let mut grimlet = Grimlet::new(compiler, &bios_path)?;

    for i in 0..17 {
        grimlet.state.regs[i] = i as u32;
    }

    println!("{:?}", grimlet.state.regs);
    grimlet.run();

    println!("{:?}", grimlet.state.regs);
    assert_eq!(grimlet.state.regs[0], 9);

    Ok(())
}

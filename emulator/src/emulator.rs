use anyhow::Result;
use inkwell::context::Context;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::{ArmDisasm, Disassembler};
use crate::jit::{Compiler, EntryPoint, FunctionCache};

pub struct Emulator<'ctx> {
    state: ArmState,
    disasm: Disassembler,
    compiler: Compiler<'ctx>,
    entry_point: EntryPoint<'ctx>,
    func_cache: FunctionCache<'ctx>,
}

impl<'ctx> Emulator<'ctx> {
    pub fn new(context: &'ctx Context, bios_path: &str) -> Result<Self> {
        let state = ArmState::with_bios(bios_path)?;
        let disasm = Disassembler::default();
        let mut compiler = Compiler::new(context);
        let entry_point = compiler.compile_entry_point();
        let func_cache = FunctionCache::new();

        Ok(Self {
            state,
            disasm,
            compiler,
            entry_point,
            func_cache,
        })
    }

    pub fn run(&mut self) {
        loop {
            let pc = self.state.pc() as usize;
            let func = match self.func_cache.get(&pc) {
                Some(func) => func,
                None => {
                    let code_block = self.disasm.next_code_block(&self.state.mem, pc);
                    match self
                        .compiler
                        .new_function(pc, &self.func_cache)
                        .build_body(code_block)
                        .compile()
                    {
                        Ok(compiled) => {
                            self.func_cache.insert(pc, compiled);
                            self.func_cache.get(&pc).unwrap()
                        }
                        Err(e) => {
                            self.compiler.dump().unwrap();
                            panic!("{}", e);
                        }
                    }
                }
            };
            unsafe {
                self.entry_point.call(&mut self.state, func.as_raw());
            }
        }
    }
}

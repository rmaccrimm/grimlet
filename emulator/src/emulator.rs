use anyhow::Result;
use inkwell::context::Context;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::Disasm;
use crate::jit::{Compiler, EntryPoint, FunctionCache};

pub struct Emulator<'ctx> {
    state: ArmState,
    disasm: Box<dyn Disasm>,
    compiler: Compiler<'ctx>,
    entry_point: EntryPoint<'ctx>,
    func_cache: FunctionCache<'ctx>,
}

impl<'ctx> Emulator<'ctx> {
    pub fn new(
        context: &'ctx Context,
        disasm: impl Disasm + 'static,
        bios_path: Option<&str>,
    ) -> Result<Self> {
        let state = match bios_path {
            Some(path) => ArmState::with_bios(path)?,
            None => ArmState::default(),
        };
        let mut compiler = Compiler::new(context);
        let entry_point = compiler.compile_entry_point();
        let func_cache = FunctionCache::new();

        Ok(Self {
            state,
            disasm: Box::new(disasm),
            compiler,
            entry_point,
            func_cache,
        })
    }

    pub fn run<F>(&mut self, exit_condition: Option<F>)
    where
        F: Fn(&ArmState) -> bool,
    {
        loop {
            if let Some(exit) = exit_condition.as_ref()
                && exit(&self.state)
            {
                break;
            }
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

#[cfg(test)]
mod tests {
    use capstone::arch::arm::{ArmCC, ArmInsn};

    use super::*;
    use crate::arm::cpu::{MainMemory, Reg};
    use crate::arm::disasm::cons::*;
    use crate::arm::disasm::{ArmInstruction, CodeBlock, Disasm};
    pub struct VecDisassembler {
        pub program: Vec<ArmInstruction>,
    }

    impl VecDisassembler {
        pub fn new(mut program: Vec<ArmInstruction>) -> Self {
            for (i, p) in program.iter_mut().enumerate() {
                p.addr = 4 * i;
            }
            VecDisassembler { program }
        }
    }

    impl Disasm for VecDisassembler {
        fn next_code_block(&self, _mem: &MainMemory, addr: usize) -> CodeBlock {
            let ind = addr / 4;
            CodeBlock::from_instructions(self.program[ind..].iter().cloned(), addr)
        }
    }

    #[test]
    fn test_factorial_program() {
        // Computes factorial of R0. Result is stored in R1
        let disasm = VecDisassembler::new(vec![
            op_reg_imm(ArmInsn::ARM_INS_CMP, 0, 1, None), // 0
            op_imm(ArmInsn::ARM_INS_B, 36, Some(ArmCC::ARM_CC_LE)), // 4
            op_reg_reg(ArmInsn::ARM_INS_MOV, 1, 0, None), // 8
            op_reg_imm(ArmInsn::ARM_INS_MOV, 0, 1, None), // 12
            op_reg_reg_reg(ArmInsn::ARM_INS_MUL, 0, 0, 1, None), // 16
            op_reg_reg_imm(ArmInsn::ARM_INS_SUBS, 1, 1, 1, None), // 20
            op_imm(ArmInsn::ARM_INS_B, 16, Some(ArmCC::ARM_CC_GT)), // 24
            op_reg_reg(ArmInsn::ARM_INS_MOV, 1, 0, None), // 28
            op_imm(ArmInsn::ARM_INS_B, 44, None),         // 32
            op_reg_imm(ArmInsn::ARM_INS_MOV, 1, 1, None), // 36
            op_imm(ArmInsn::ARM_INS_B, 44, None),         // 40
        ]);

        let llvm_ctx = Context::create();
        let mut emulator = Emulator::new(&llvm_ctx, disasm, None).unwrap();

        let mut run = |n| -> u32 {
            emulator.state = ArmState::default();
            emulator.state.regs[Reg::R0 as usize] = n;
            emulator.run(Some(|st: &ArmState| -> bool { st.pc() == 44 }));
            emulator.state.regs[Reg::R1 as usize]
        };

        assert_eq!(run(2), 2);
        assert_eq!(run(3), 6);
        assert_eq!(run(4), 24);
        assert_eq!(run(5), 120);
        assert_eq!(run(12), 479001600);
    }
}

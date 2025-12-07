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
                    println!("{}", code_block);
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

    fn cond_test_case(cond: ArmCC, flags: u32, should_execute: bool) {
        // Conditionally move r1 to r0, then exit
        let disasm = VecDisassembler::new(vec![
            op_reg_reg(ArmInsn::ARM_INS_MOV, 0, 1, Some(cond)),
            op_imm(ArmInsn::ARM_INS_B, 100, None),
        ]);

        let llvm_ctx = Context::create();
        let mut emulator = Emulator::new(&llvm_ctx, disasm, None).unwrap();
        // Appears in R0 if MOV was successful
        let confirm_val = u32::MAX;
        emulator.state.regs[Reg::R1 as usize] = confirm_val;
        // Prev instruction on initialization is just a NOP, so it won't override these flags
        emulator.state.regs[Reg::CPSR as usize] = flags << 28;
        emulator.run(Some(|st: &ArmState| -> bool { st.pc() == 100 }));
        emulator.compiler.dump().unwrap();
        assert_eq!(
            emulator.state.r0(),
            if should_execute { confirm_val } else { 0 }
        );
    }

    #[test]
    fn test_conditions() {
        //                                 _z__
        cond_test_case(ArmCC::ARM_CC_EQ, 0b0001, false);
        cond_test_case(ArmCC::ARM_CC_EQ, 0b0111, true);
        cond_test_case(ArmCC::ARM_CC_NE, 0b1000, true);
        cond_test_case(ArmCC::ARM_CC_NE, 0b1110, false);
        //                                 __c_
        cond_test_case(ArmCC::ARM_CC_HS, 0b0100, false);
        cond_test_case(ArmCC::ARM_CC_HS, 0b1010, true);
        cond_test_case(ArmCC::ARM_CC_LO, 0b1001, true);
        cond_test_case(ArmCC::ARM_CC_LO, 0b1110, false);
        //                                 n___
        cond_test_case(ArmCC::ARM_CC_MI, 0b0000, false);
        cond_test_case(ArmCC::ARM_CC_MI, 0b1001, true);
        cond_test_case(ArmCC::ARM_CC_PL, 0b0000, true);
        cond_test_case(ArmCC::ARM_CC_PL, 0b1000, false);
        //                                 ___v
        cond_test_case(ArmCC::ARM_CC_VS, 0b0010, false);
        cond_test_case(ArmCC::ARM_CC_VS, 0b1001, true);
        cond_test_case(ArmCC::ARM_CC_VC, 0b1100, true);
        cond_test_case(ArmCC::ARM_CC_VC, 0b1101, false);
        //                                 _zc_
        cond_test_case(ArmCC::ARM_CC_HI, 0b0001, false);
        cond_test_case(ArmCC::ARM_CC_HI, 0b0011, true);
        cond_test_case(ArmCC::ARM_CC_HI, 0b1100, false);
        cond_test_case(ArmCC::ARM_CC_HI, 0b0110, false);
        cond_test_case(ArmCC::ARM_CC_LS, 0b0000, true);
        cond_test_case(ArmCC::ARM_CC_LS, 0b0010, false);
        cond_test_case(ArmCC::ARM_CC_LS, 0b0100, true);
        cond_test_case(ArmCC::ARM_CC_LS, 0b0110, true);
        //                                 n__v
        cond_test_case(ArmCC::ARM_CC_GE, 0b0000, true);
        cond_test_case(ArmCC::ARM_CC_GE, 0b0001, false);
        cond_test_case(ArmCC::ARM_CC_GE, 0b1000, false);
        cond_test_case(ArmCC::ARM_CC_GE, 0b1001, true);
        cond_test_case(ArmCC::ARM_CC_LT, 0b0000, false);
        cond_test_case(ArmCC::ARM_CC_LT, 0b0001, true);
        cond_test_case(ArmCC::ARM_CC_LT, 0b1000, true);
        cond_test_case(ArmCC::ARM_CC_LT, 0b1001, false);
        //                                 nz_v
        cond_test_case(ArmCC::ARM_CC_GT, 0b0010, true);
        cond_test_case(ArmCC::ARM_CC_GT, 0b0001, false);
        cond_test_case(ArmCC::ARM_CC_GT, 0b1000, false);
        cond_test_case(ArmCC::ARM_CC_GT, 0b1011, true);
        cond_test_case(ArmCC::ARM_CC_GT, 0b0110, false);
        cond_test_case(ArmCC::ARM_CC_GT, 0b0101, false);
        cond_test_case(ArmCC::ARM_CC_GT, 0b1100, false);
        cond_test_case(ArmCC::ARM_CC_GT, 0b1111, false);
        //                                 nz_v
        cond_test_case(ArmCC::ARM_CC_LE, 0b0000, false);
        cond_test_case(ArmCC::ARM_CC_LE, 0b0001, true);
        cond_test_case(ArmCC::ARM_CC_LE, 0b1000, true);
        cond_test_case(ArmCC::ARM_CC_LE, 0b1011, false);
        cond_test_case(ArmCC::ARM_CC_LE, 0b0110, true);
        cond_test_case(ArmCC::ARM_CC_LE, 0b0101, true);
        cond_test_case(ArmCC::ARM_CC_LE, 0b1100, true);
        cond_test_case(ArmCC::ARM_CC_LE, 0b1101, true);
    }
}

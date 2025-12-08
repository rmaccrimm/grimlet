use anyhow::Result;
use inkwell::context::Context;

use crate::arm::cpu::ArmState;
use crate::arm::disasm::Disasm;
use crate::jit::{Compiler, FunctionCache};

pub struct Emulator<'ctx> {
    pub state: ArmState,
    disasm: Box<dyn Disasm>,
    compiler: Compiler<'ctx>,
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
        let compiler = Compiler::new(context);
        let func_cache = FunctionCache::new();

        Ok(Self {
            state,
            disasm: Box::new(disasm),
            compiler,
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
                func.call(&mut self.state);
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
        assert_eq!(run(0), 1);
        assert_eq!(run(1), 1);
        assert_eq!(run(2), 2);
        assert_eq!(run(3), 6);
        assert_eq!(run(4), 24);
        assert_eq!(run(5), 120);
        assert_eq!(run(12), 479001600);
    }

    fn cond_test_case(cond: ArmCC, flags: u32) -> bool {
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

        // True if it ran, false if skipped
        emulator.state.r0() == confirm_val
    }

    #[rustfmt::skip]
    #[test]
    fn test_conditions() {
        //                                          _z__
        assert!(!cond_test_case(ArmCC::ARM_CC_EQ, 0b0001));
        assert!(cond_test_case(ArmCC::ARM_CC_EQ,  0b0111));
        assert!(cond_test_case(ArmCC::ARM_CC_NE,  0b1000));
        assert!(!cond_test_case(ArmCC::ARM_CC_NE, 0b1110));
        //                                          __c_
        assert!(!cond_test_case(ArmCC::ARM_CC_HS, 0b0100));
        assert!(cond_test_case(ArmCC::ARM_CC_HS,  0b1010));
        assert!(cond_test_case(ArmCC::ARM_CC_LO,  0b1001));
        assert!(!cond_test_case(ArmCC::ARM_CC_LO, 0b1110));
        //                                          n___
        assert!(!cond_test_case(ArmCC::ARM_CC_MI, 0b0000));
        assert!(cond_test_case(ArmCC::ARM_CC_MI,  0b1001));
        assert!(cond_test_case(ArmCC::ARM_CC_PL,  0b0000));
        assert!(!cond_test_case(ArmCC::ARM_CC_PL, 0b1000));
        //                                          ___v
        assert!(!cond_test_case(ArmCC::ARM_CC_VS, 0b0010));
        assert!(cond_test_case(ArmCC::ARM_CC_VS,  0b1001));
        assert!(cond_test_case(ArmCC::ARM_CC_VC,  0b1100));
        assert!(!cond_test_case(ArmCC::ARM_CC_VC, 0b1101));
        //                                          _zc_
        assert!(!cond_test_case(ArmCC::ARM_CC_HI, 0b0001));
        assert!(cond_test_case(ArmCC::ARM_CC_HI,  0b0011));
        assert!(!cond_test_case(ArmCC::ARM_CC_HI, 0b1100));
        assert!(!cond_test_case(ArmCC::ARM_CC_HI, 0b0110));
        assert!(cond_test_case(ArmCC::ARM_CC_LS,  0b0000));
        assert!(!cond_test_case(ArmCC::ARM_CC_LS, 0b0010));
        assert!(cond_test_case(ArmCC::ARM_CC_LS,  0b0100));
        assert!(cond_test_case(ArmCC::ARM_CC_LS,  0b0110));
        //                                          n__v
        assert!(cond_test_case(ArmCC::ARM_CC_GE,  0b0000));
        assert!(!cond_test_case(ArmCC::ARM_CC_GE, 0b0001));
        assert!(!cond_test_case(ArmCC::ARM_CC_GE, 0b1000));
        assert!(cond_test_case(ArmCC::ARM_CC_GE,  0b1001));
        assert!(!cond_test_case(ArmCC::ARM_CC_LT, 0b0000));
        assert!(cond_test_case(ArmCC::ARM_CC_LT,  0b0001));
        assert!(cond_test_case(ArmCC::ARM_CC_LT,  0b1000));
        assert!(!cond_test_case(ArmCC::ARM_CC_LT, 0b1001));
        //                                          nz_v
        assert!(cond_test_case(ArmCC::ARM_CC_GT,  0b0010));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b0001));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b1000));
        assert!(cond_test_case(ArmCC::ARM_CC_GT,  0b1011));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b0110));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b0101));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b1100));
        assert!(!cond_test_case(ArmCC::ARM_CC_GT, 0b1111));
        //                                          nz_v
        assert!(!cond_test_case(ArmCC::ARM_CC_LE, 0b0000));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b0001));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b1000));
        assert!(!cond_test_case(ArmCC::ARM_CC_LE, 0b1011));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b0110));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b0101));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b1100));
        assert!(cond_test_case(ArmCC::ARM_CC_LE,  0b1101));
    }

    #[test]
    fn test_cmp_flags() {
        // cmp r0, #1
        // b 100
        let disasm = VecDisassembler::new(vec![
            op_reg_imm(ArmInsn::ARM_INS_CMP, 0, 1, None),
            op_imm(ArmInsn::ARM_INS_B, 100, None),
        ]);

        let llvm_ctx = Context::create();
        let mut em = Emulator::new(&llvm_ctx, disasm, None).unwrap();

        let exit = Some(|st: &ArmState| -> bool { st.pc() == 100 });
        let r0 = Reg::R0 as usize;
        let pc = Reg::PC as usize;
        let cpsr = Reg::CPSR as usize;

        let mut test_case = |n: u32| -> u32 {
            em.state.regs[r0] = n;
            em.state.regs[pc] = 0;
            em.state.regs[cpsr] = 0;
            em.run(exit);
            em.state.regs[cpsr] >> 28
        };

        // Positive result
        assert_eq!(test_case(2), 0b0010); // nzcv
        // 0 result
        assert_eq!(test_case(1), 0b0110); // nzcv
        // negative result (unsigned underflow)
        assert_eq!(test_case(0), 0b1000); // nzcv
        // negative result (no underflow)
        assert_eq!(test_case(-1i32 as u32), 0b1010); // nzcv
        // signed underflow only (positive result)
        assert_eq!(test_case(i32::MIN as u32), 0b0011); // nzcv
    }
}

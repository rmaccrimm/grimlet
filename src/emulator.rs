use anyhow::Result;
use inkwell::context::Context;

use crate::arm::state::ArmState;
use crate::arm::disasm::Disasm;
use crate::jit::{Compiler, FunctionCache};

pub struct Emulator {
    pub state: ArmState,
    llvm_ctx: Context,
    disasm: Box<dyn Disasm>,
}

impl Emulator {
    pub fn new(disasm: impl Disasm + 'static, bios_path: Option<&str>) -> Result<Self> {
        let state = match bios_path {
            Some(path) => ArmState::with_bios(path)?,
            None => ArmState::default(),
        };
        let llvm_ctx = Context::create();
        Ok(Self {
            state,
            disasm: Box::new(disasm),
            llvm_ctx,
        })
    }

    pub fn run<F>(&mut self, exit_condition: F)
    where
        F: Fn(&ArmState) -> bool,
    {
        let mut compiler = Compiler::new(&self.llvm_ctx);
        let mut func_cache = FunctionCache::new();

        loop {
            if exit_condition(&self.state) {
                break;
            }
            let instr_addr = self.state.curr_instr_addr();
            let func = match func_cache.get(&instr_addr) {
                Some(func) => func,
                None => {
                    let code_block = self.disasm.next_code_block(&self.state.mem, instr_addr);
                    println!("{:#?}", code_block);
                    match compiler
                        .new_function(instr_addr, Some(&func_cache))
                        .build_body(code_block)
                        .compile()
                    {
                        Ok(compiled) => {
                            func_cache.insert(instr_addr, compiled);
                            func_cache.get(&instr_addr).unwrap()
                        }
                        Err(e) => {
                            compiler.dump().unwrap();
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
    use crate::arm::state::Reg;
    use crate::arm::state::memory::MainMemory;
    use crate::arm::disasm::code_block::CodeBlock;
    use crate::arm::disasm::cons::*;
    use crate::arm::disasm::instruction::ArmInstruction;
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

    fn cond_test_case(cond: ArmCC, flags: u32) -> bool {
        // Conditionally move r1 to r0, then exit
        let disasm = VecDisassembler::new(vec![
            op_reg_reg(ArmInsn::ARM_INS_MOV, 0, 1, Some(cond)),
            op_imm(ArmInsn::ARM_INS_B, 100, None),
        ]);

        let mut emulator = Emulator::new(disasm, None).unwrap();
        // Appears in R0 if MOV was successful
        let confirm_val = u32::MAX;
        emulator.state.regs[Reg::R1] = confirm_val;
        // Prev instruction on initialization is just a NOP, so it won't override these flags
        emulator.state.regs[Reg::CPSR as usize] = flags << 28;
        emulator.run(|st: &ArmState| -> bool { st.curr_instr_addr() == 100 });

        // True if it ran, false if skipped
        emulator.state.regs[Reg::R0] == confirm_val
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

        let mut em = Emulator::new(disasm, None).unwrap();

        let exit = |st: &ArmState| -> bool { st.curr_instr_addr() == 100 };
        let r0 = Reg::R0;
        let cpsr = Reg::CPSR as usize;

        let mut test_case = |n: u32| -> u32 {
            em.state.jump_to(0);
            em.state.regs[r0] = n;
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

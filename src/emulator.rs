use std::fs::{self, File};
use std::io::{BufReader, Read};

use anyhow::{Result, bail};
use clap::ValueEnum;
use inkwell::context::Context;

use crate::arm::disasm::Disasm;
use crate::arm::state::ArmState;
use crate::jit::{FunctionBuilder, FunctionCache};

pub mod video;

// Just 2^24 / 60 rounded down
const CYCLES_PER_FRAME: u32 = 279620;

/// Currently only signals a change from ARM to THUMB and vice-versa, but I'm thinking this will
/// be useful in handling things like memory mapped IO
pub enum SystemMessage {
    PauseEmulation,
    ResumeEmulation,
}

pub struct Emulator<'a> {
    pub state: ArmState,
    ctx: &'a Context,
    disasm: Box<dyn Disasm>,
    func_cache: FunctionCache<'a>,
}

/// Print codeblocks before running
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum DebugOutput {
    Assembly,
    Struct,
}

impl<'a> Emulator<'a> {
    pub fn new(disasm: impl Disasm + 'static, llvm_ctx: &'a Context) -> Self {
        Self {
            ctx: llvm_ctx,
            state: ArmState::default(),
            disasm: Box::new(disasm),
            func_cache: FunctionCache::new(),
        }
    }

    pub fn load_rom(&mut self, rom_path: impl AsRef<str>, addr: u32) -> Result<()> {
        let path = rom_path.as_ref();
        if !fs::exists(path)? {
            bail!("ROM file not found");
        }
        let f = BufReader::new(File::open(path)?);
        let (mem_ref, _) = self.state.mem.mem_map_lookup_mut(addr)?;

        for (i, byte) in f.bytes().enumerate() {
            mem_ref[i] = byte?;
        }
        Ok(())
    }

    pub fn run<F>(&mut self, exit_condition: F, print: Option<DebugOutput>)
    where
        F: Fn(&ArmState) -> bool,
    {
        loop {
            if exit_condition(&self.state) {
                // compiler.dump().unwrap();
                break;
            }
            if self.disasm.get_mode() != self.state.current_mode {
                self.disasm.set_mode(self.state.current_mode);
            }

            let instr_addr = self.state.curr_instr_addr();
            let func = match self.func_cache.get(&instr_addr) {
                Some(func) => func,
                None => {
                    let code_block = self
                        .disasm
                        .next_code_block(&self.state.mem, instr_addr)
                        .expect("disassembly failed");
                    match print {
                        Some(DebugOutput::Assembly) => println!("{}", code_block),
                        Some(DebugOutput::Struct) => println!("{:#?}", code_block),
                        None => (),
                    }
                    let builder = FunctionBuilder::new(self.ctx, instr_addr)
                        .expect("failed to initialize function")
                        .build_body(code_block)
                        .expect("failed to build function");
                    match builder.compile() {
                        Ok(compiled) => {
                            self.func_cache.insert(instr_addr, compiled);
                            self.func_cache.get(&instr_addr).unwrap()
                        }
                        Err(e) => {
                            panic!("{}", e);
                        }
                    }
                }
            };
            unsafe {
                func.call(&mut self.state);
            }
            if self.state.cycle_count >= CYCLES_PER_FRAME {
                self.state.cycle_count %= CYCLES_PER_FRAME;
                // Render frame here
                break;
            }
            if print.is_some() {
                println!("{}", self.state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use capstone::RegId;
    use capstone::arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType, ArmReg};

    use super::*;
    use crate::arm::disasm::code_block::CodeBlock;
    use crate::arm::disasm::instruction::ArmInstruction;
    use crate::arm::state::memory::MemoryManager;
    use crate::arm::state::{ArmMode, Reg};
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
        fn next_code_block(&self, _mem: &MemoryManager, addr: usize) -> Result<CodeBlock> {
            let ind = addr / 4;
            Ok(CodeBlock::from_instructions(
                self.program[ind..].iter().cloned(),
                addr,
            ))
        }

        fn get_mode(&self) -> ArmMode { ArmMode::ARM }

        fn set_mode(&mut self, _mode: ArmMode) {
            // Does nothing
        }
    }

    pub fn reg(r: usize) -> ArmOperand {
        let reg_id = match r {
            0 => ArmReg::ARM_REG_R0,
            1 => ArmReg::ARM_REG_R1,
            2 => ArmReg::ARM_REG_R2,
            3 => ArmReg::ARM_REG_R3,
            4 => ArmReg::ARM_REG_R4,
            5 => ArmReg::ARM_REG_R5,
            6 => ArmReg::ARM_REG_R6,
            7 => ArmReg::ARM_REG_R7,
            8 => ArmReg::ARM_REG_R8,
            9 => ArmReg::ARM_REG_R9,
            10 => ArmReg::ARM_REG_R10,
            11 => ArmReg::ARM_REG_R11,
            12 => ArmReg::ARM_REG_R12,
            13 => ArmReg::ARM_REG_SP,
            14 => ArmReg::ARM_REG_LR,
            15 => ArmReg::ARM_REG_PC,
            16 => ArmReg::ARM_REG_CPSR,
            _ => panic!("unhandled reg"),
        };
        ArmOperand {
            op_type: ArmOperandType::Reg(RegId(reg_id as u16)),
            ..Default::default()
        }
    }

    pub fn imm(i: i32) -> ArmOperand {
        ArmOperand {
            op_type: ArmOperandType::Imm(i),
            ..Default::default()
        }
    }

    macro_rules! op {
        ($opcode:expr, $cond:expr, $($args:expr),+) => {
            ArmInstruction {
                opcode: $opcode,
                cond: $cond.unwrap_or(ArmCC::ARM_CC_AL),
                operands: vec![$($args),+],
                ..Default::default()
            }
        };
    }

    fn cond_test_case(cond: ArmCC, flags: u32) -> bool {
        // Conditionally move r1 to r0, then exit
        let disasm = VecDisassembler::new(vec![
            op!(ArmInsn::ARM_INS_MOV, Some(cond), reg(0), reg(1)),
            op!(ArmInsn::ARM_INS_B, None, imm(100)),
        ]);

        let ctx = Context::create();
        let mut emulator = Emulator::new(disasm, &ctx);
        // Appears in R0 if MOV was successful
        let confirm_val = u32::MAX;
        emulator.state.regs[Reg::R1] = confirm_val;
        // Prev instruction on initialization is just a NOP, so it won't override these flags
        emulator.state.regs[Reg::CPSR as usize] = flags << 28;
        emulator.run(
            |st: &ArmState| -> bool { st.curr_instr_addr() == 100 },
            None,
        );

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
            op!(ArmInsn::ARM_INS_CMP, None, reg(0), imm(1)),
            op!(ArmInsn::ARM_INS_B, None, imm(100)),
        ]);

        let ctx = Context::create();
        let mut em = Emulator::new(disasm, &ctx);

        let exit = |st: &ArmState| -> bool { st.curr_instr_addr() == 100 };
        let r0 = Reg::R0;
        let cpsr = Reg::CPSR as usize;

        let mut test_case = |n: u32| -> u32 {
            em.state.jump_to(0, ArmMode::ARM as i8);
            em.state.regs[r0] = n;
            em.state.regs[cpsr] = 0;
            em.run(exit, Some(DebugOutput::Assembly));
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

    macro_rules! stmdb_tests {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (instruction, expected_sp) = $value;

                let mut instrs = vec![
                    op!(ArmInsn::ARM_INS_MOV, None, reg(13), imm(0x4000)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(0), imm(1)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(1), imm(2)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(2), imm(3)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(3), imm(5)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(4), imm(7)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(5), imm(11)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(6), imm(13)),
                    op!(ArmInsn::ARM_INS_MOV, None, reg(7), imm(17)),
                ];
                instrs.push(instruction);
                instrs.push(op!(ArmInsn::ARM_INS_B, None, imm(100)));

                let disasm = VecDisassembler::new(instrs);
                let ctx = Context::create();
                let mut emulator = Emulator::new(disasm, &ctx);

                let exit = |st: &ArmState| -> bool { st.curr_instr_addr() == 100 };
                emulator.run(exit, Some(DebugOutput::Assembly));

                assert_eq!(emulator.state.regs[Reg::SP], expected_sp);
                let expected = [1, 2, 3, 5, 7, 11, 13, 17];
                for (i, w) in emulator.state.mem.iter_word(0x4000 - 32).unwrap().enumerate() {
                    assert_eq!(u32::from_le_bytes(w.try_into().unwrap()), expected[i])
                }
            }
        )*
        }
    }

    stmdb_tests! {
        test_stmdb_writeback: (
            ArmInstruction {
                opcode: ArmInsn::ARM_INS_STMDB,
                cond: ArmCC::ARM_CC_AL,
                operands: vec![
                    reg(Reg::SP as usize),
                    // mis-ordered to testing ordering behaviour
                    reg(0),
                    reg(2),
                    reg(5),
                    reg(1),
                    reg(3),
                    reg(7),
                    reg(6),
                    reg(4),
                ],
                writeback: true,
                ..Default::default()
            },
            0x4000 - 32
        ),
        test_stmdb_no_writeback: (
            ArmInstruction {
                opcode: ArmInsn::ARM_INS_STMDB,
                cond: ArmCC::ARM_CC_AL,
                operands: vec![
                    reg(Reg::SP as usize),
                    // mis-ordered to testing ordering behaviour
                    reg(4),
                    reg(0),
                    reg(2),
                    reg(1),
                    reg(3),
                    reg(7),
                    reg(6),
                    reg(5),
                ],
                writeback: false,
                ..Default::default()
            },
            0x4000,
        ),
        test_push: (
            ArmInstruction {
                opcode: ArmInsn::ARM_INS_PUSH,
                cond: ArmCC::ARM_CC_AL,
                operands: vec![
                    reg(4),
                    reg(0),
                    reg(2),
                    reg(1),
                    reg(3),
                    reg(7),
                    reg(6),
                    reg(5),
                ],
                writeback: false,
                ..Default::default()
            },
            0x4000 - 32,
        ),
    }
}

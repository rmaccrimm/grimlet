use std::fmt::Display;

use anyhow::{Result, anyhow, bail};
use capstone::arch::ArchOperand;
use capstone::arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType, ArmShift as CsArmShift};
use capstone::{Capstone, Insn};

use crate::arm::state::{ArmMode, Reg};

// A single disassembled ARM instruction. Basically a clone of the Capstone instruction but
// easier to access since we know we're only working with ARM instructions.
#[derive(Clone, Debug)]
pub struct ArmInstruction {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: u32,
    pub repr: Option<String>,
    pub cond: ArmCC,
    pub mode: ArmMode,
    pub updates_flags: bool,
    pub writeback: bool,
    pub binary: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArmShift {
    AsrImm(u32),
    LslImm(u32),
    LsrImm(u32),
    RorImm(u32),
    AsrReg(Reg),
    LslReg(Reg),
    LsrReg(Reg),
    RorReg(Reg),
    Rrx,
}

/// `shifter_operand`: operands for data processing instructions. Parsed out ahead of time since it
/// potentially is encoded in multiple Capstone operands.
#[derive(Clone, Copy)]
pub enum ShifterOperand {
    Reg { reg: Reg, shift: Option<ArmShift> },

    // What Capstone returns is somewhat inconsistent - sometimes it seems to be a 32-bit value
    // already parsed from the 8 bit immediate + shift, other times it's the two values separately
    Imm { imm: i32, rotate: Option<i32> },
}

/// `addressing_mode`: operand for single Load/Store instructions. Parsed out ahead of time since it
/// potentially is encoded in multiple Capstone operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemOperand {
    pub base: Reg,
    pub offset: MemOffset,
    pub writeback: Option<WritebackMode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemOffset {
    Reg {
        index: Reg,
        shift: Option<ArmShift>,
        subtract: bool,
    },
    Imm(i32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritebackMode {
    PostIndex,
    PreIndex,
}

/// one of CPSR/SPSR and a mask for bits being changed
pub struct ProgramStatusReg(pub Reg, pub u32);

impl ArmInstruction {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn, mode: ArmMode) -> Self {
        let detail = cs
            .insn_detail(insn)
            .expect("failed to get instruction detail");

        let arch_detail = detail.arch_detail();
        let arm_detail = arch_detail.arm().expect("expected an arm instruction");

        let mut operands = Vec::new();

        for op in arch_detail.operands() {
            if let ArchOperand::ArmOperand(a) = op {
                operands.push(a);
            } else {
                panic!("not an ARM operand")
            }
        }
        let cond = arm_detail.cc();
        let b = match insn.len() {
            4 => u32::from_le_bytes(insn.bytes().try_into().expect("failed to parse bytes")),
            2 => u32::from(u16::from_le_bytes(
                insn.bytes().try_into().expect("failed to parse bytes"),
            )),
            _ => {
                panic!("unexpected number of bytes read: {}", insn.len());
            }
        };

        Self {
            opcode: ArmInsn::from(insn.id().0),
            operands: arch_detail
                .operands()
                .into_iter()
                .map(|a| match a {
                    ArchOperand::ArmOperand(op) => op,
                    _ => panic!("not an ARM operand"),
                })
                .collect(),
            addr: u32::try_from(insn.address()).expect("address too large"),
            repr: Some(insn.to_string()),
            cond,
            mode,
            updates_flags: arm_detail.update_flags(),
            writeback: arm_detail.writeback(),
            binary: b,
        }
    }

    pub fn get_reg_op(&self, ind: usize) -> Reg {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{self}\" missing operand {ind}"))
            .op_type
        {
            Reg::from(reg_id)
        } else {
            panic!("\"{self}\" operand {ind} is not a register");
        }
    }

    pub fn get_imm_op(&self, ind: usize) -> i32 {
        if let ArmOperandType::Imm(i) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{self}\" missing operand {ind}"))
            .op_type
        {
            i
        } else {
            panic!("\"{self}\" operand {ind} is not an immediate value");
        }
    }

    pub fn get_mem_op(&self, index: usize) -> Result<MemOperand> {
        let mut op_iter = self.operands.iter().skip(index);
        let mem_op = op_iter
            .next()
            .unwrap_or_else(|| panic!("\"{self}\" missing mem operand"));
        let post_index_op = op_iter.next();

        let ArmOperandType::Mem(inner_mem_op) = mem_op.op_type else {
            bail!("operand is not a memory address: {mem_op:?}");
        };

        let base = Reg::from(inner_mem_op.base());

        if let Some(post_op) = post_index_op {
            debug_assert_eq!(inner_mem_op.index().0, 0);
            debug_assert_eq!(inner_mem_op.disp(), 0);

            let wb_mode = Some(WritebackMode::PostIndex);
            let offset = match post_op.op_type {
                ArmOperandType::Reg(reg_id) => MemOffset::Reg {
                    index: Reg::from(reg_id),
                    shift: ArmShift::try_from(post_op.shift).ok(),
                    subtract: post_op.subtracted,
                },
                ArmOperandType::Imm(i) => {
                    debug_assert!(!(i < 0 && post_op.subtracted));
                    let imm = if post_op.subtracted { -i } else { i };
                    MemOffset::Imm(imm)
                }
                _ => bail!("unexpected post-index op_type: {:?}", post_op.op_type),
            };
            Ok(MemOperand {
                base,
                offset,
                writeback: wb_mode,
            })
        } else {
            let writeback = self.writeback.then_some(WritebackMode::PreIndex);
            let index = inner_mem_op.index();
            let disp = inner_mem_op.disp();

            let offset = if index.0 == 0 {
                MemOffset::Imm(disp)
            } else {
                debug_assert_eq!(inner_mem_op.disp(), 0);
                MemOffset::Reg {
                    index: Reg::from(index),
                    shift: ArmShift::try_from(mem_op.shift).ok(),
                    subtract: mem_op.subtracted,
                }
            };
            Ok(MemOperand {
                base,
                offset,
                writeback,
            })
        }
    }

    // return register operands after 1st in ascending order
    pub fn get_reg_list(&self, skip: usize) -> Result<Vec<Reg>> {
        let mut regs = vec![];
        for operand in self.operands.iter().skip(skip) {
            match operand.op_type {
                ArmOperandType::Reg(reg_id) => regs.push(Reg::from(reg_id)),
                _ => bail!("non-register operand in register list"),
            }
        }
        if regs.is_empty() {
            bail!("no register list provided")
        }
        regs.sort();
        Ok(regs)
    }

    pub fn get_shifter_op(&self, skip: usize) -> Result<ShifterOperand> {
        let mut op_iter = self.operands.iter().skip(skip);
        let first = op_iter
            .next()
            .ok_or(anyhow!("missing 1st shifter operand"))?;
        Ok(match first.op_type {
            ArmOperandType::Reg(reg_id) => ShifterOperand::Reg {
                reg: Reg::from(reg_id),
                shift: ArmShift::try_from(first.shift).ok(),
            },
            ArmOperandType::Imm(imm) => match op_iter.next() {
                Some(snd_op) => {
                    if let ArmOperandType::Imm(rot) = snd_op.op_type {
                        ShifterOperand::Imm {
                            imm,
                            rotate: Some(rot),
                        }
                    } else {
                        bail!("Shifter operand rotation must be an immediate value")
                    }
                }
                None => ShifterOperand::Imm { imm, rotate: None },
            },
            _ => bail!("Invalid shifter operand type"),
        })
    }

    /// This is largely based on trial and error by compiling instructions with gvasm and seeing
    /// what capstone spits out. It seems like _fc and _f and are the only possible options
    pub fn get_sys_reg_op(&self, ind: usize) -> Result<ProgramStatusReg> {
        let op = self
            .operands
            .get(ind)
            .ok_or(anyhow!("unexecpted op type"))?;
        if let ArmOperandType::SysReg(reg_id) = op.op_type {
            // Pulled from capstone.rs, not clear how you're really meant to use these
            Ok(match reg_id.0 {
                // spsr_flg / spsr_f -> ARM_SYSREG_SPSR_F
                8 => ProgramStatusReg(Reg::SPSR, 0xf000_0000),
                // cpsr / cpsr_fc / spsr / spsr_fc -> ? are these supposed to be the same?
                9 => ProgramStatusReg(Reg::SPSR, 0xf000_00ff),
                // cpsr_flg / aspr_nzcvq -> ARM_SYSREG_APSR_NZCVQ
                258 => ProgramStatusReg(Reg::CPSR, 0x0f00_00ff),
                _ => bail!("unknown sysreg id: {op:?}"),
            })
        } else {
            bail!("operand is not a sysreg: {op:?}");
        }
    }
}

impl Default for ArmInstruction {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            cond: ArmCC::ARM_CC_AL,
            operands: Vec::new(),
            addr: Default::default(),
            repr: None,
            mode: ArmMode::ARM,
            updates_flags: true,
            writeback: false,
            binary: 0,
        }
    }
}

impl Display for ArmInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.repr {
            Some(s) => write!(f, "{s}"),
            None => write!(f, "missing repr"),
        }
    }
}

impl TryFrom<CsArmShift> for ArmShift {
    type Error = String;

    fn try_from(value: CsArmShift) -> std::result::Result<Self, Self::Error> {
        let shift = match value {
            CsArmShift::Invalid => {
                return Err("Invalid shift".into());
            }
            CsArmShift::Asr(imm) => ArmShift::AsrImm(imm),
            CsArmShift::Lsl(imm) => ArmShift::LslImm(imm),
            CsArmShift::Lsr(imm) => ArmShift::LsrImm(imm),
            CsArmShift::Ror(imm) => ArmShift::RorImm(imm),
            CsArmShift::AsrReg(reg_id) => ArmShift::AsrReg(Reg::from(reg_id)),
            CsArmShift::LslReg(reg_id) => ArmShift::LslReg(Reg::from(reg_id)),
            CsArmShift::LsrReg(reg_id) => ArmShift::LsrReg(Reg::from(reg_id)),
            CsArmShift::RorReg(reg_id) => ArmShift::RorReg(Reg::from(reg_id)),
            CsArmShift::Rrx(_) | CsArmShift::RrxReg(_) => ArmShift::Rrx,
        };
        Ok(shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm::disasm::Disassembler;

    // Using assembled bytes to test as we cannot construct a capstone::ArmMemOp
    fn get_mem_op_from_assembled(bin: u32) -> MemOperand {
        let disasm = Disassembler::default();
        let instr = disasm.instr_iter(&bin.to_le_bytes(), 0).next().unwrap();
        instr.get_mem_op(1).unwrap()
    }

    #[test]
    fn test_get_mem_op_1() {
        // str r9, [r0]
        assert_eq!(
            get_mem_op_from_assembled(0xe580_9000),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Imm(0),
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_2() {
        // strlt r0, [r1, #12]
        assert_eq!(
            get_mem_op_from_assembled(0xb581_000c),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Imm(12),
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_3() {
        // ldrne r1, [r1, #-12]
        assert_eq!(
            get_mem_op_from_assembled(0x1511_100c),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Imm(-12),
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_4() {
        // ldr r2, [pc, r0]
        assert_eq!(
            get_mem_op_from_assembled(0xe79f_2000),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Reg {
                    index: Reg::R0,
                    shift: None,
                    subtract: false
                },
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_5() {
        // ldrne r1, [sp, -r14]
        assert_eq!(
            get_mem_op_from_assembled(0x171d_100e),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: None,
                    subtract: true
                },
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_6() {
        // ldr r1, [sp, -r14, asr #1]
        assert_eq!(
            get_mem_op_from_assembled(0xe71d_10ce),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: Some(ArmShift::AsrImm(1)),
                    subtract: true
                },
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_7() {
        // ldr r2, [r13, r14, lsr #3]
        assert_eq!(
            get_mem_op_from_assembled(0xe79d_21ae),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: Some(ArmShift::LsrImm(3)),
                    subtract: false
                },
                writeback: None
            }
        );
    }

    #[test]
    fn test_get_mem_op_8() {
        // str r10, [ip, #0]!
        assert_eq!(
            get_mem_op_from_assembled(0xe5ac_a000),
            MemOperand {
                base: Reg::R12,
                offset: MemOffset::Imm(0),
                writeback: Some(WritebackMode::PreIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_9() {
        // strge r5, [pc, #-12]!
        assert_eq!(
            get_mem_op_from_assembled(0xa52f_500c),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Imm(-12),
                writeback: Some(WritebackMode::PreIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_10() {
        // ldr r8, [r9, r10]!
        assert_eq!(
            get_mem_op_from_assembled(0xe7b9_800a),
            MemOperand {
                base: Reg::R9,
                offset: MemOffset::Reg {
                    index: Reg::R10,
                    shift: None,
                    subtract: false
                },
                writeback: Some(WritebackMode::PreIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_11() {
        // ldr r0, [r1, r2, lsl #31]!
        assert_eq!(
            get_mem_op_from_assembled(0xe7b1_0f82),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Reg {
                    index: Reg::R2,
                    shift: Some(ArmShift::LslImm(31)),
                    subtract: false
                },
                writeback: Some(WritebackMode::PreIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_12() {
        // ldr r7, [r0], -r0
        assert_eq!(
            get_mem_op_from_assembled(0xe610_7000),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Reg {
                    index: Reg::R0,
                    shift: None,
                    subtract: true
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_13() {
        // ldreq lr, [sp], pc
        assert_eq!(
            get_mem_op_from_assembled(0x069d_e00f),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::PC,
                    shift: None,
                    subtract: false
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_14() {
        // ldreq r3, [r0], #4095
        assert_eq!(
            get_mem_op_from_assembled(0x0490_3fff),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Imm(4095),
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_15() {
        // strne r2, [r7], #-290
        assert_eq!(
            get_mem_op_from_assembled(0x1407_2122),
            MemOperand {
                base: Reg::R7,
                offset: MemOffset::Imm(-290),
                writeback: Some(WritebackMode::PostIndex),
            }
        );
    }

    #[test]
    fn test_get_mem_op_16() {
        // ldr pc, [pc], pc, ror #20
        assert_eq!(
            get_mem_op_from_assembled(0xe69f_fa6f),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Reg {
                    index: Reg::PC,
                    shift: Some(ArmShift::RorImm(20)),
                    subtract: false
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }

    #[test]
    fn test_get_mem_op_17() {
        // ldr r0, [r1], -r2, rrx
        assert_eq!(
            get_mem_op_from_assembled(0xe611_0062),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Reg {
                    index: Reg::R2,
                    shift: Some(ArmShift::Rrx),
                    subtract: true
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }
}

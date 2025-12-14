use std::fmt::Display;

use anyhow::{Result, bail};
use capstone::arch::ArchOperand;
use capstone::arch::arm::{ArmCC, ArmInsn, ArmOperand, ArmOperandType, ArmShift};
use capstone::{Capstone, Insn};

use crate::arm::state::{ArmMode, Reg};

// A single disassembled ARM instruction. Basically a clone of the Capstone instruction but
// easier to access since we know we're only working with ARM instructions.
#[derive(Clone, Debug)]
pub struct ArmInstruction {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: usize,
    pub repr: Option<String>,
    pub cond: ArmCC,
    pub mode: ArmMode,
    pub updates_flags: bool,
    pub regs_accessed: Vec<Reg>,
    pub writeback: bool,
}

/// <addressing_mode> operand for single Load/Store instructions. For other operand types, the
/// capstone types are currently used directly
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemOperand {
    pub base: Reg,
    pub offset: MemOffset,
    pub writeback: Option<WritebackMode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemOffset {
    Reg {
        index: Reg,
        shift: ArmShift,
        subtract: bool,
    },
    Imm(i32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WritebackMode {
    PostIndex,
    PreIndex,
}

impl Default for ArmInstruction {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            cond: ArmCC::ARM_CC_AL,
            operands: Default::default(),
            addr: Default::default(),
            repr: None,
            mode: ArmMode::ARM,
            updates_flags: true,
            regs_accessed: vec![],
            writeback: false,
        }
    }
}

impl ArmInstruction {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn, mode: ArmMode) -> Self {
        let detail = cs
            .insn_detail(insn)
            .expect("failed to get instruction detail");

        let arch_detail = detail.arch_detail();
        let arm_detail = arch_detail.arm().expect("expected an arm instruction");

        let mut regs_accessed = Vec::new();
        let mut operands = Vec::new();

        for op in arch_detail.operands() {
            if let ArchOperand::ArmOperand(a) = op {
                if let ArmOperandType::Reg(reg_id) = a.op_type {
                    regs_accessed.push(Reg::from(reg_id))
                }
                operands.push(a);
            } else {
                panic!("not an ARM operand")
            }
        }

        let cond = arm_detail.cc();

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
            addr: insn.address() as usize,
            repr: Some(insn.to_string()),
            cond,
            mode,
            updates_flags: arm_detail.update_flags(),
            regs_accessed,
            writeback: arm_detail.writeback(),
        }
    }

    pub fn get_reg_op(&self, ind: usize) -> Reg {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{}\" missing operand {}", self, ind))
            .op_type
        {
            Reg::from(reg_id)
        } else {
            panic!("\"{}\" operand {} is not a register", self, ind);
        }
    }

    pub fn get_imm_op(&self, ind: usize) -> i32 {
        if let ArmOperandType::Imm(i) = self
            .operands
            .get(ind)
            .unwrap_or_else(|| panic!("\"{}\" missing operand {}", self, ind))
            .op_type
        {
            i
        } else {
            panic!("\"{}\" operand {} is not an immediate value", self, ind);
        }
    }

    pub fn get_mem_op(&self) -> Result<MemOperand> {
        let mut op_iter = self.operands.iter().skip(1);
        let mem_op = op_iter
            .next()
            .unwrap_or_else(|| panic!("\"{}\" missing mem operand", self));
        let post_index_op = op_iter.next();

        let inner_mem_op = if let ArmOperandType::Mem(mem) = mem_op.op_type {
            mem
        } else {
            bail!("operand is not a memory address: {:?}", mem_op);
        };

        let base = Reg::from(inner_mem_op.base());

        match post_index_op {
            Some(post_op) => {
                debug_assert_eq!(inner_mem_op.index().0, 0);
                debug_assert_eq!(inner_mem_op.disp(), 0);

                let wb_mode = Some(WritebackMode::PostIndex);
                let offset = match post_op.op_type {
                    ArmOperandType::Reg(reg_id) => MemOffset::Reg {
                        index: Reg::from(reg_id),
                        shift: post_op.shift,
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
            }
            None => {
                let writeback = self.writeback.then_some(WritebackMode::PreIndex);
                let index = inner_mem_op.index();
                let disp = inner_mem_op.disp();

                let offset = match index.0 {
                    0 => MemOffset::Imm(disp),
                    _ => {
                        debug_assert_eq!(inner_mem_op.disp(), 0);
                        MemOffset::Reg {
                            index: Reg::from(index),
                            shift: mem_op.shift,
                            subtract: mem_op.subtracted,
                        }
                    }
                };
                Ok(MemOperand {
                    base,
                    offset,
                    writeback,
                })
            }
        }
    }
}

impl Display for ArmInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.repr {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "missing repr"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm::disasm::Disassembler;

    // Using assembled bytes to test as we cannot construct a capstone::ArmMemOp
    fn get_mem_op_from_assembled(bin: u32) -> MemOperand {
        let instr = Disassembler::default().disasm_single(&bin.to_le_bytes(), 0);
        instr.get_mem_op().unwrap()
    }

    #[test]
    fn test_get_mem_op() {
        // str r9, [r0]
        assert_eq!(
            get_mem_op_from_assembled(0xe5809000),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Imm(0),
                writeback: None
            }
        );
        // strlt r0, [r1, #12]
        assert_eq!(
            get_mem_op_from_assembled(0xb581000c),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Imm(12),
                writeback: None
            }
        );
        // ldrne r1, [r1, #-12]
        assert_eq!(
            get_mem_op_from_assembled(0x1511100c),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Imm(-12),
                writeback: None
            }
        );
        // ldr r2, [pc, r0]
        assert_eq!(
            get_mem_op_from_assembled(0xe79f2000),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Reg {
                    index: Reg::R0,
                    shift: ArmShift::Invalid,
                    subtract: false
                },
                writeback: None
            }
        );
        // ldrne r1, [sp, -r14]
        assert_eq!(
            get_mem_op_from_assembled(0x171d100e),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: ArmShift::Invalid,
                    subtract: true
                },
                writeback: None
            }
        );
        // ldr r1, [sp, -r14, asr #1]
        assert_eq!(
            get_mem_op_from_assembled(0xe71d10ce),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: ArmShift::Asr(1),
                    subtract: true
                },
                writeback: None
            }
        );
        // ldr r2, [r13, r14, lsr #3]
        assert_eq!(
            get_mem_op_from_assembled(0xe79d21ae),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::LR,
                    shift: ArmShift::Lsr(3),
                    subtract: false
                },
                writeback: None
            }
        );
        // str r10, [ip, #0]!
        assert_eq!(
            get_mem_op_from_assembled(0xe5aca000),
            MemOperand {
                base: Reg::IP,
                offset: MemOffset::Imm(0),
                writeback: Some(WritebackMode::PreIndex)
            }
        );
        // strge r5, [pc, #-12]!
        assert_eq!(
            get_mem_op_from_assembled(0xa52f500c),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Imm(-12),
                writeback: Some(WritebackMode::PreIndex)
            }
        );
        // ldr r8, [r9, r10]!
        assert_eq!(
            get_mem_op_from_assembled(0xe7b9800a),
            MemOperand {
                base: Reg::R9,
                offset: MemOffset::Reg {
                    index: Reg::R10,
                    shift: ArmShift::Invalid,
                    subtract: false
                },
                writeback: Some(WritebackMode::PreIndex)
            }
        );
        // ldr r0, [r1, r2, lsl #31]!
        assert_eq!(
            get_mem_op_from_assembled(0xe7b10f82),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Reg {
                    index: Reg::R2,
                    shift: ArmShift::Lsl(31),
                    subtract: false
                },
                writeback: Some(WritebackMode::PreIndex)
            }
        );
        // ldr r7, [r0], -r0
        assert_eq!(
            get_mem_op_from_assembled(0xe6107000),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Reg {
                    index: Reg::R0,
                    shift: ArmShift::Invalid,
                    subtract: true
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
        // ldreq lr, [sp], pc
        assert_eq!(
            get_mem_op_from_assembled(0x69de00f),
            MemOperand {
                base: Reg::SP,
                offset: MemOffset::Reg {
                    index: Reg::PC,
                    shift: ArmShift::Invalid,
                    subtract: false
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
        // ldreq r3, [r0], #4095
        assert_eq!(
            get_mem_op_from_assembled(0x4903fff),
            MemOperand {
                base: Reg::R0,
                offset: MemOffset::Imm(4095),
                writeback: Some(WritebackMode::PostIndex)
            }
        );
        // strne r2, [r7], #-290
        assert_eq!(
            get_mem_op_from_assembled(0x14072122),
            MemOperand {
                base: Reg::R7,
                offset: MemOffset::Imm(-290),
                writeback: Some(WritebackMode::PostIndex),
            }
        );
        // ldr pc, [pc], pc, ror #20
        assert_eq!(
            get_mem_op_from_assembled(0xe69ffa6f),
            MemOperand {
                base: Reg::PC,
                offset: MemOffset::Reg {
                    index: Reg::PC,
                    shift: ArmShift::Ror(20),
                    subtract: false
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
        // ldr r0, [r1], -r2, rrx
        assert_eq!(
            get_mem_op_from_assembled(0xe6110062),
            MemOperand {
                base: Reg::R1,
                offset: MemOffset::Reg {
                    index: Reg::R2,
                    shift: ArmShift::Rrx(0),
                    subtract: true
                },
                writeback: Some(WritebackMode::PostIndex)
            }
        );
    }
}

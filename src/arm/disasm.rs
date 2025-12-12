pub mod cons;
use std::collections::HashSet;
use std::fmt::{Display, Write};

use anyhow::{Result, bail};
use capstone::arch::arm::{ArmCC, ArmInsn, ArmOpMem, ArmOperand, ArmOperandType, ArmShift};
use capstone::arch::{ArchOperand, BuildsCapstone};
use capstone::{Capstone, Insn};

use crate::arm::cpu::{ArmMode, MainMemory, Reg};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemOffset {
    Reg {
        index: Reg,
        shift: ArmShift,
        subtract: bool,
    },
    Imm(i32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WritebackMode {
    PostIndex,
    PreIndex,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemOperand {
    pub base: Reg,
    pub offset: MemOffset,
    pub writeback: Option<WritebackMode>,
}

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
                    ArmOperandType::Imm(i) => MemOffset::Imm(i),
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

#[derive(Debug)]
pub struct CodeBlock {
    pub instrs: Vec<ArmInstruction>,
    pub regs_accessed: HashSet<Reg>,
    pub start_addr: usize,
}

impl Display for CodeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "---------------")?;
        for instr in self.instrs.iter() {
            write!(f, "0x{:02x}:  ", instr.addr)?;
            match instr.repr {
                Some(_) => writeln!(f, "{}", instr)?,
                None => writeln!(f, "{:?}", instr)?,
            }
        }
        writeln!(f, "regs accessed: {:?}", self.regs_accessed)?;
        writeln!(f, "---------------")
    }
}

impl CodeBlock {
    pub fn from_instructions(
        instr_iter: impl Iterator<Item = ArmInstruction>,
        start_addr: usize,
    ) -> Self {
        let mut instrs = Vec::new();
        let mut regs_read = HashSet::new();
        for instr in instr_iter {
            instrs.push(instr);
            let instr = instrs.last().unwrap();
            match instr.opcode {
                ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => {
                    break;
                }
                _ => {}
            }
            for r in instr.regs_accessed.iter() {
                regs_read.insert(*r);
            }
        }
        CodeBlock {
            instrs,
            start_addr,
            regs_accessed: regs_read,
        }
    }
}

pub trait Disasm {
    fn next_code_block(&self, mem: &MainMemory, addr: usize) -> CodeBlock;
}

pub struct MemoryDisassembler {
    cs: Capstone,
    current_mode: ArmMode,
}

impl MemoryDisassembler {
    pub fn disasm_single(&self, chunk: &[u8], addr: usize) -> ArmInstruction {
        let instructions = self
            .cs
            .disasm_count(chunk, addr as u64, 1)
            .expect("Capstone disassembly failed");

        let i = instructions
            .as_ref()
            .first()
            .expect("Capstone returned no instructions");
        ArmInstruction::from_cs_insn(&self.cs, i, self.current_mode)
    }

    pub fn set_mode(&mut self, mode: ArmMode) -> Result<()> {
        self.current_mode = mode;
        match mode {
            ArmMode::ARM => self.cs.set_mode(capstone::Mode::Arm)?,
            ArmMode::THUMB => self.cs.set_mode(capstone::Mode::Thumb)?,
        }
        Ok(())
    }
}

impl Default for MemoryDisassembler {
    fn default() -> Self {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()
            .expect("failed to build capstone instance");
        Self {
            cs,
            current_mode: ArmMode::ARM,
        }
    }
}

impl Disasm for MemoryDisassembler {
    fn next_code_block(&self, mem: &MainMemory, start_addr: usize) -> CodeBlock {
        let instr_iter = mem.iter_word(start_addr).enumerate().map(move |(i, ch)| {
            let addr = start_addr + 4 * i;
            self.disasm_single(ch, addr)
        });
        CodeBlock::from_instructions(instr_iter, start_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disasm() {
        let bytes: [u8; 4] = [0b10010000, 0b00000001, 0b00010010, 0b11100000];
        let disasm = MemoryDisassembler::default();
        let res = &disasm.cs.disasm_all(&bytes, 0).unwrap()[0];
        println!(
            "{:#?}",
            ArmInstruction::from_cs_insn(&disasm.cs, res, ArmMode::ARM)
        );
    }

    // Using assembled bytes to test as we cannot construct a capstone::ArmMemOp
    fn get_mem_op_from_assembled(bin: u32) -> MemOperand {
        let instr = MemoryDisassembler::default().disasm_single(&bin.to_le_bytes(), 0);
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
    }
}

use anyhow::Result;
use anyhow::anyhow;
use capstone::RegId;
use capstone::arch::BuildsCapstone;
use capstone::{
    Capstone, Insn,
    arch::{
        ArchOperand,
        arm::{ArmInsn, ArmOperand, ArmOperandType},
    },
};
use std::fmt::Display;

use crate::arm::cpu::ArmMode;
use crate::arm::cpu::MainMemory;
use crate::arm::cpu::Reg;

// A single disassembled ARM instruction
#[derive(Clone, Debug)]
pub struct ArmDisasm {
    pub opcode: ArmInsn,
    pub operands: Vec<ArmOperand>,
    pub addr: usize,
    pub repr: String,
}

impl Default for ArmDisasm {
    fn default() -> Self {
        Self {
            opcode: ArmInsn::ARM_INS_NOP,
            operands: Default::default(),
            addr: Default::default(),
            repr: Default::default(),
        }
    }
}

impl ArmDisasm {
    pub fn from_cs_insn(cs: &Capstone, insn: &Insn) -> Result<Self> {
        Ok(Self {
            opcode: ArmInsn::from(insn.id().0),
            operands: cs
                .insn_detail(insn)?
                .arch_detail()
                .operands()
                .into_iter()
                .map(|a| match a {
                    ArchOperand::ArmOperand(op) => op,
                    _ => panic!("unexpected operand"),
                })
                .collect(),

            addr: insn.address() as usize,
            repr: insn.to_string(),
        })
    }

    pub fn get_reg_op(&self, ind: usize) -> Result<Reg> {
        if let ArmOperandType::Reg(reg_id) = self
            .operands
            .get(ind)
            .ok_or(anyhow!("missing operand {}", ind))?
            .op_type
        {
            Ok(Reg::from(reg_id.0 as usize))
        } else {
            Err(anyhow!("Bad operand: not a register"))
        }
    }

    pub fn get_imm_op(&self, ind: usize) -> Result<i32> {
        if let ArmOperandType::Imm(i) = self
            .operands
            .get(ind)
            .ok_or(anyhow!("missing operand {}", ind))?
            .op_type
        {
            Ok(i)
        } else {
            Err(anyhow!("Bad operand: not a register"))
        }
    }

    pub fn op_reg_imm(opcode: ArmInsn, r: u16, i: i32) -> Self {
        ArmDisasm {
            opcode,
            operands: vec![
                ArmOperand {
                    op_type: ArmOperandType::Reg(RegId(r)),
                    ..Default::default()
                },
                ArmOperand {
                    op_type: ArmOperandType::Imm(i),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }
}

impl Display for ArmDisasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr)
    }
}

pub struct CodeBlock {
    pub instrs: Vec<ArmDisasm>,
    pub start_addr: usize,
    pub labels: Vec<usize>,
}

pub struct Disassembler {
    cs: Capstone,
}

impl Disassembler {
    pub fn new() -> Result<Self> {
        let cs = Capstone::new()
            .arm()
            .mode(capstone::arch::arm::ArchMode::Arm)
            .detail(true)
            .build()?;
        Ok(Self { cs })
    }

    pub fn next_code_block(&self, mem: &MainMemory, start_addr: usize) -> Result<CodeBlock> {
        let mut instrs = Vec::new();
        let mut labels = Vec::new();

        for instr in self.iter_insns(mem, start_addr as usize, ArmMode::ARM) {
            if let ArmInsn::ARM_INS_B = instr.opcode {
                // TODO can these be negative? Pretty sure it's translated to abs address
                let target = instr.get_imm_op(0)? as usize;
                if target < instr.addr && target >= start_addr {
                    // This is a loop, need a label at the target address
                    labels.push(instr.addr);
                }
                instrs.push(instr);
            }
        }
        Ok(CodeBlock {
            instrs,
            start_addr,
            labels,
        })
    }

    pub fn iter_insns(
        &self,
        mem: &MainMemory,
        start_addr: usize,
        _mode: ArmMode,
    ) -> impl Iterator<Item = ArmDisasm> {
        mem.bios
            .chunks(4)
            .skip(start_addr as usize)
            .enumerate()
            .map(move |(i, ch)| {
                let instructions = self
                    .cs
                    .disasm_count(ch, (start_addr as u64) + 4 * i as u64, 1)
                    .unwrap();
                let i = instructions.as_ref().iter().next().unwrap();
                ArmDisasm::from_cs_insn(&self.cs, i).unwrap()
            })
    }
}

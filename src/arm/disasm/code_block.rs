use std::collections::HashSet;
use std::fmt::Display;

use capstone::arch::arm::{ArmInsn, ArmOperand, ArmOperandType};

use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::Reg;

/// A single batch of instrutions to be compiled into an LLVM function
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
        let mut regs_accessed = HashSet::new();

        for instr in instr_iter {
            println!("ITER {}", instr);
            instrs.push(instr);
            let instr = instrs.last().unwrap();
            for a in instr.operands.iter() {
                match a.op_type {
                    ArmOperandType::Reg(reg_id) => {
                        regs_accessed.insert(Reg::from(reg_id));
                    }
                    ArmOperandType::Mem(arm_op_mem) => {
                        let base = arm_op_mem.base();
                        let index = arm_op_mem.index();
                        if base.0 != 0 {
                            regs_accessed.insert(Reg::from(base));
                        }
                        if index.0 != 0 {
                            regs_accessed.insert(Reg::from(index));
                        }
                    }
                    _ => (),
                }
            }
            if is_terminator(instr) {
                break;
            }
        }
        CodeBlock {
            instrs,
            start_addr,
            regs_accessed,
        }
    }
}

/// This function ends the LLVM function (performs a branch)
fn is_terminator(instr: &ArmInstruction) -> bool {
    match instr.opcode {
        ArmInsn::ARM_INS_B | ArmInsn::ARM_INS_BX | ArmInsn::ARM_INS_BL => true,
        // Instructions that write to pc
        ArmInsn::ARM_INS_AND
        | ArmInsn::ARM_INS_EOR
        | ArmInsn::ARM_INS_SUB
        | ArmInsn::ARM_INS_RSB
        | ArmInsn::ARM_INS_ADD
        | ArmInsn::ARM_INS_ADC
        | ArmInsn::ARM_INS_SBC
        | ArmInsn::ARM_INS_RSC
        | ArmInsn::ARM_INS_ORR
        | ArmInsn::ARM_INS_MOV
        | ArmInsn::ARM_INS_MVN
        | ArmInsn::ARM_INS_BIC => match instr.operands.first() {
            Some(ArmOperand {
                op_type: ArmOperandType::Reg(reg_id),
                ..
            }) => Reg::from(*reg_id) == Reg::PC,
            _ => false,
        },
        ArmInsn::ARM_INS_LDR
        | ArmInsn::ARM_INS_LDRB
        | ArmInsn::ARM_INS_LDRSB
        | ArmInsn::ARM_INS_LDRH
        | ArmInsn::ARM_INS_LDRSH => {
            let rd = instr.get_reg_op(0);
            // At the moment this is done twice for every instruction, probably best to replace
            // the capstone operands alltogether on first load. This would also allow for some
            // separation of error types (builder can just return BuilderErrors)
            let op = instr.get_mem_op().expect("failed to get mem op");
            rd == Reg::PC || (instr.writeback && op.base == Reg::PC)
        }
        ArmInsn::ARM_INS_STR | ArmInsn::ARM_INS_STRB | ArmInsn::ARM_INS_STRH => {
            let op = instr.get_mem_op().expect("failed to get mem op");
            instr.writeback && op.base == Reg::PC
        }
        ArmInsn::ARM_INS_LDM
        | ArmInsn::ARM_INS_LDMIB
        | ArmInsn::ARM_INS_LDMDA
        | ArmInsn::ARM_INS_LDMDB => instr
            .get_reg_list(1)
            .expect("failed to get reg list")
            .into_iter()
            .any(|r| r == Reg::PC),
        ArmInsn::ARM_INS_POP => instr
            .get_reg_list(0)
            .expect("failed to get reg list")
            .into_iter()
            .any(|r| r == Reg::PC),
        // Behaviour of a number of others (mul, e.g.) is unpredictable if they write to pc.
        // Ignoring these for now.
        _ => false,
    }
}

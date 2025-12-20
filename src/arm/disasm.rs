pub mod code_block;
pub mod instruction;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;

use crate::arm::disasm::code_block::CodeBlock;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::ArmMode;
use crate::arm::state::memory::MainMemory;

/// Trait for CodeBlock producers. Mainly exists so tests can provide instructions without needing
/// an actual binary.
pub trait Disasm {
    fn next_code_block(&self, mem: &MainMemory, addr: usize) -> Result<CodeBlock>;
}

// Produces CodeBlocks from in-memory program
pub struct Disassembler {
    cs: Capstone,
    current_mode: ArmMode,
}

impl Disassembler {
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

impl Default for Disassembler {
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

impl Disasm for Disassembler {
    fn next_code_block(&self, mem: &MainMemory, start_addr: usize) -> Result<CodeBlock> {
        // TODO what's the appropriate type for addresses?
        let instr_iter = mem
            .iter_word(start_addr as u32)?
            .enumerate()
            .map(move |(i, ch)| {
                let addr = start_addr + 4 * i;
                self.disasm_single(ch, addr)
            });
        Ok(CodeBlock::from_instructions(instr_iter, start_addr))
    }
}

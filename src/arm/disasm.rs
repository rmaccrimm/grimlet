pub mod code_block;
pub mod instruction;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;

use crate::arm::disasm::code_block::CodeBlock;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::ArmMode;
use crate::arm::state::memory::MemoryManager;

/// Trait for `CodeBlock` producers. Mainly exists so tests can provide instructions without needing
/// an actual binary.
pub trait Disasm {
    fn next_code_block(&self, mem: &MemoryManager, addr: u32) -> Result<CodeBlock>;

    fn set_mode(&mut self, mode: ArmMode);

    fn get_mode(&self) -> ArmMode;
}

pub struct InstrIter<'a> {
    bytes: &'a [u8],
    cs: &'a Capstone,
    pos: usize,
    start_addr: u32,
    mode: ArmMode,
}

impl Iterator for InstrIter<'_> {
    type Item = ArmInstruction;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let instructions = self
            .cs
            .disasm_count(
                &self.bytes[self.pos..],
                u64::from(self.start_addr + u32::try_from(self.pos).expect("position too large")),
                1,
            )
            .expect("Capstone disassembly failed");
        let i = instructions
            .as_ref()
            .first()
            .expect("Capstone returned no instructions");
        self.pos += i.len();
        Some(ArmInstruction::from_cs_insn(self.cs, i, self.mode))
    }
}

// Produces CodeBlocks from in-memory program
pub struct Disassembler {
    cs: Capstone,
    current_mode: ArmMode,
}

impl<'a> Disassembler {
    pub fn instr_iter(&'a self, bytes: &'a [u8], start_addr: u32) -> InstrIter<'a> {
        InstrIter {
            bytes,
            cs: &self.cs,
            pos: 0,
            start_addr,
            mode: self.current_mode,
        }
    }
}

impl Disasm for Disassembler {
    fn next_code_block(&self, mem: &MemoryManager, start_addr: u32) -> Result<CodeBlock> {
        // TODO what's the appropriate type for addresses?
        Ok(CodeBlock::from_instructions(
            self.instr_iter(mem.mem_map_lookup(start_addr)?.0, start_addr),
            start_addr,
        ))
    }

    fn set_mode(&mut self, mode: ArmMode) {
        self.current_mode = mode;
        let res = match mode {
            ArmMode::ARM => self.cs.set_mode(capstone::Mode::Arm),
            ArmMode::THUMB => self.cs.set_mode(capstone::Mode::Thumb),
        };
        res.expect("error while updating capstone mode");
    }

    fn get_mode(&self) -> ArmMode { self.current_mode }
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

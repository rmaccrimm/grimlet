pub mod code_block;
pub mod instruction;

use std::collections::VecDeque;

use anyhow::Result;
use capstone::Capstone;
use capstone::arch::BuildsCapstone;
use capstone::arch::arm::ArchMode;

use crate::arm::disasm::code_block::CodeBlock;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::ArmMode;
use crate::arm::state::memory::MemoryManager;

pub struct InstrWindow<'a> {
    queue: VecDeque<ArmInstruction>,
    bytes: &'a [u8],
    cs: &'a Capstone,
    start_addr: usize,
    mode: ArmMode,
    pos: usize,
    window_len: usize,
}

impl InstrWindow<'_> {
    pub fn peek_one(&self) -> Option<&ArmInstruction> { self.queue.front() }

    pub fn peek_two(&self) -> Option<&ArmInstruction> { self.queue.get(1) }

    fn refill(&mut self) {
        while self.queue.len() < self.window_len {
            if self.pos >= self.bytes.len() {
                return;
            }
            let addr = (self.start_addr + self.pos) as u64;
            let instructions = self
                .cs
                .disasm_count(&self.bytes[self.pos..], addr, 1)
                .expect("Capstone disassembly failed");
            let i = instructions
                .as_ref()
                .first()
                .expect("Capstone returned no instructions");
            self.pos += i.len();
            self.queue
                .push_back(ArmInstruction::from_cs_insn(self.cs, i, self.mode));
        }
    }
}

impl Iterator for InstrWindow<'_> {
    type Item = ArmInstruction;

    // Return the next instruction to compile and refill the window
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.queue.pop_front()?;
        self.refill();
        Some(next)
    }
}

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

    pub fn new_window(
        &'a self,
        bytes: &'a [u8],
        start_addr: u32,
        window_len: usize,
    ) -> InstrWindow<'a> {
        assert!(
            window_len >= 2,
            "InstrWindow must have at least 3 instructions"
        );
        let mut window = InstrWindow {
            queue: VecDeque::with_capacity(window_len),
            bytes,
            cs: &self.cs,
            pos: 0,
            mode: self.current_mode,
            start_addr: start_addr as usize,
            window_len,
        };
        window.refill();
        window
    }
}

impl Disasm for Disassembler {
    fn next_code_block(&self, mem: &MemoryManager, start_addr: u32) -> Result<CodeBlock> {
        // TODO what's the appropriate type for addresses?
        CodeBlock::from_instructions(
            self.instr_iter(mem.mem_map_lookup(start_addr)?.0, start_addr),
            start_addr,
        )
    }

    fn set_mode(&mut self, mode: ArmMode) {
        self.current_mode = mode;
        let res = match mode {
            ArmMode::ARM => self.cs.set_mode(ArchMode::Arm.into()),
            ArmMode::THUMB => self.cs.set_mode(ArchMode::Thumb.into()),
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

#[cfg(test)]
mod tests {
    use capstone::arch::arm::ArmInsn;

    use super::*;

    #[test]
    fn test_instr_window_arm_basic_usage() {
        // 0:  mov r0, #1
        // 4:  mov r1, #0
        // 8:  str r1, [r1]
        // 12: mov r0, r0 (nop)
        let program = [
            0x01, 0x00, 0xa0, 0xe3, 0x00, 0x10, 0xa0, 0xe3, 0x00, 0x10, 0x81, 0xe5, 0x00, 0x00,
            0xa0, 0xe1,
        ];
        let disasm = Disassembler::default();
        let mut window = disasm.new_window(&program, 0x4000, 3);
        assert_eq!(window.queue.len(), 3);

        let next = window.next().unwrap();
        assert_eq!(next.opcode, ArmInsn::ARM_INS_MOV);
        assert_eq!(next.addr, 0x4000);
        assert_eq!(next.size, 4);
        assert_eq!(window.queue.len(), 3);

        let peek = window.peek_one().unwrap();
        assert_eq!(peek.opcode, ArmInsn::ARM_INS_MOV);
        assert_eq!(peek.addr, 0x4004);
        assert_eq!(peek.size, 4);
        assert_eq!(window.queue.len(), 3);

        let peek_two = window.peek_two().unwrap();
        assert_eq!(peek_two.opcode, ArmInsn::ARM_INS_STR);
        assert_eq!(peek_two.addr, 0x4008);
        assert_eq!(peek_two.size, 4);
        assert_eq!(window.queue.len(), 3);

        window.next().unwrap();
        assert_eq!(window.queue.len(), 2);
        assert!(window.peek_one().is_some());
        assert!(window.peek_two().is_some());

        window.next().unwrap();
        assert_eq!(window.queue.len(), 1);
        assert!(window.peek_one().is_some());
        assert!(window.peek_two().is_none());

        window.next().unwrap();
        assert_eq!(window.queue.len(), 0);
        assert!(window.peek_one().is_none());
        assert!(window.peek_two().is_none());

        assert!(window.next().is_none());
    }

    #[test]
    fn test_instr_window_thumb_variable_width() {
        // bl <label>
        // bl <label>
        // movs r0, #0
        // movs r1, #0
        let program = [
            0xff, 0xf7, 0xf5, 0xff, 0xff, 0xf7, 0xf5, 0xff, 0x00, 0x20, 0x00, 0x21,
        ];
        let mut disasm = Disassembler::default();
        disasm.set_mode(ArmMode::THUMB);

        let window = disasm.new_window(&program, 0x0800_0000, 2);
        assert_eq!(window.queue.len(), 2);

        let first = window.peek_one().unwrap();
        assert_eq!(first.opcode, ArmInsn::ARM_INS_BL);
        assert_eq!(first.size, 4);
        assert_eq!(first.addr, 0x0800_0000);
        assert_eq!(first.mode, ArmMode::THUMB);

        let second = window.peek_two().unwrap();
        assert_eq!(second.opcode, ArmInsn::ARM_INS_BL);
        assert_eq!(second.size, 4);
        assert_eq!(second.addr, 0x0800_0004);
        assert_eq!(second.mode, ArmMode::THUMB);

        let expected_addrs = [0x0800_0000, 0x0800_0004, 0x0800_0008, 0x0800_000a];
        for (i, instr) in window.enumerate() {
            assert_eq!(instr.addr, expected_addrs[i]);
        }
    }
}

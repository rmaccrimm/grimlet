use std::cell::RefCell;
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::rc::Rc;
use std::sync::mpsc;

use anyhow::{Result, bail};
use clap::ValueEnum;
use inkwell::context::Context;

use crate::arm::disasm::Disassembler;
use crate::arm::state::memory::MemoryManager;
use crate::arm::state::{ArmState, JumpTarget, Reg};
use crate::jit::FunctionBuilder;
use crate::jit::cache::FunctionCache;
use crate::utils::interval_tree::IntervalTree;

pub mod video;

// 2^24 / 60 rounded down
const CYCLES_PER_FRAME: u32 = 279_620;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum DebugOutput {
    Assembly,
    Struct,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum DumpLLVM {
    OnFail,
    BeforeCompilation,
    AfterCompilation,
}

#[derive(Clone, Debug, Default)]
pub struct EnvConfig {
    pub debug_output: Option<DebugOutput>,
    pub dump_llvm: Option<DumpLLVM>,
    pub print_state: bool,
    pub llvm_output_dir: String,
}

impl EnvConfig {
    fn load_from_env() -> Self {
        Self {
            debug_output: env::var("DEBUG_OUTPUT")
                .ok()
                .and_then(|s| DebugOutput::from_str(&s, true).ok()),
            dump_llvm: env::var("DUMP_LLVM")
                .ok()
                .and_then(|s| DumpLLVM::from_str(&s, true).ok()),
            print_state: env::var("PRINT_STATE")
                .ok()
                .is_some_and(|s| s.to_lowercase() == "true"),
            llvm_output_dir: env::var("LLVM_OUTPUT_DIR").unwrap_or("llvm".into()),
        }
    }
}

pub struct Emulator<'a> {
    pub state: ArmState,
    ctx: &'a Context,
    disasm: Disassembler,
    func_cache: FunctionCache<'a>,
    config: EnvConfig,
}

impl<'a> Emulator<'a> {
    pub fn new(llvm_ctx: &'a Context) -> Self {
        let (tx, rx) = mpsc::channel();
        let ival_tree = Rc::new(RefCell::new(IntervalTree::default()));
        let mem = MemoryManager::new(ival_tree.clone(), tx);
        let config = EnvConfig::load_from_env();

        Self {
            ctx: llvm_ctx,
            state: ArmState::new(mem),
            disasm: Disassembler::default(),
            func_cache: FunctionCache::new(ival_tree, rx),
            config,
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

    pub fn run<F>(&mut self, exit_condition: F)
    where
        F: Fn(&ArmState) -> bool,
    {
        loop {
            if let Some(JumpTarget { addr, mode }) = self.state.jump_target.take() {
                let (mem_ref, _) = self
                    .state
                    .mem
                    .mem_map_lookup(addr)
                    .expect("invalid address");
                // It'd be nice if we could avoid doing this initial disassembly twice
                let mut iter = self.disasm.new_window_iter(mem_ref, addr);
                iter.next();
                // Should Compiled Function handle this?
                self.state.regs[Reg::PC] = iter
                    .peek_two()
                    .map_or(addr + mode.pc_byte_offset(), |instr| instr.addr);
                // TOOD Sort this out, probably depends on where we're jumping to (i.e. can't
                // ignore the wait states above). Also where should this code go?
                self.state.add_cycles(2);
                self.disasm.set_mode(mode);
            }

            let instr_addr = self.state.curr_instr_addr();
            let func = if let Some(func) = self.func_cache.get(instr_addr) {
                if self.config.debug_output.is_some() {
                    println!("<compiled function at {instr_addr:08x}>\n");
                }
                func
            } else {
                self.compile_new_func(instr_addr);
                self.func_cache.get(instr_addr).unwrap()
            };
            unsafe {
                func.call(&mut self.state);
            }
            self.func_cache.update();

            if self.config.print_state {
                println!("{}", self.state);
            }
            if self.state.cycle_count >= CYCLES_PER_FRAME {
                self.state.cycle_count %= CYCLES_PER_FRAME;
                // Render frame here
                break;
            }
            if exit_condition(&self.state) {
                break;
            }
        }
    }

    fn compile_new_func(&mut self, addr: u32) {
        let (mem_ref, _) = self
            .state
            .mem
            .mem_map_lookup(addr)
            .expect("invalid address");

        let iter = self.disasm.new_window_iter(mem_ref, addr);

        let builder = FunctionBuilder::new(self.ctx, addr)
            .expect("failed to initialize function")
            .set_config(self.config.clone())
            .build_body(iter)
            .expect("failed to build function");

        match builder.compile() {
            Ok(compiled) => {
                self.func_cache.insert(compiled);
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm::state::{ArmMode, Reg};

    #[rustfmt::skip]
    #[test]
    fn test_early_exit_and_cache_invalidation_on_write_to_current_block() {
        // 0:  mov r0, #1.   <- make sure we exit
        // 4:  mov r1, #0
        // 8:  str r1, [r1]  <- writes to address 0 (within current code block)
        // 12: nop           
        // 16: mov r1, #4    <- last instruction executed before exiting
        // 20: mov r1, #24   <- should not be executed
        // 24: b #104        <- just so we have a stopping point
        let program = [
            0x01, 0x00, 0xa0, 0xe3, 
            0x00, 0x10, 0xa0, 0xe3, 
            0x00, 0x10, 0x81, 0xe5,
            0x00, 0x00, 0xa0, 0xe1, 
            0x01, 0x1f, 0xa0, 0xe3, 
            0x06, 0x1f, 0xa0, 0xe3,
            0x12, 0x00, 0x00, 0xea,
        ];

        let ctx = Context::create();
        let mut emulator = Emulator::new(&ctx);

        let (mem, _) = emulator.state.mem.mem_map_lookup_mut(0).unwrap();
        mem[..program.len()].copy_from_slice(&program);

        emulator.run(|st: &ArmState| -> bool { st.regs[Reg::R0] == 1 });
        assert!(emulator.func_cache.get(0).is_none());
        assert_eq!(emulator.state.regs[Reg::R1], 4);
        assert_eq!(emulator.state.regs[Reg::PC], 28);
    }

    #[rustfmt::skip]
    #[test]
    fn test_self_modifying_code_cache_invalidation() {        
        //     main:
        //  0:     mov r1, #1           <- r1 > 0 = exit
        //  4:     cmp r1, #1
        //  8:     beq func
        // 12:     b end
        //     func:
        // 16:     ldr r0, =main
        // 20:     ldr r2, =0xe3a01f06  <- writes instruction 'mov r1, #24' to 0
        // 24:     str r2, [r0]
        // 28:     b main
        //     end:
        // 32:     b end
        // 36:     .pool
        let program = [
            0x01, 0x10, 0xa0, 0xe3, 
            0x01, 0x00, 0x51, 0xe3, 
            0x00, 0x00, 0x00, 0x0a, 
            0x03, 0x00, 0x00, 0xea, 
            0x02, 0x03, 0xa0, 0xe3, 
            0x0c, 0x20, 0x9f, 0xe5, 
            0x00, 0x20, 0x80, 0xe5, 
            0xf7, 0xff, 0xff, 0xea, 
            0xfe, 0xff, 0xff, 0xea, 
            0x00, 0x00, 0x00, 0x00, 
            0x06, 0x1f, 0xa0, 0xe3,
        ];

        let ctx = Context::create();
        let mut emulator = Emulator::new(&ctx);

        // gvasm generated assembly - needs to be loaded into cartridge mem for labels to work
        // (Could this be a problem later? I don't know if this should actually be writeable)
        let cart_addr = 0x0800_0000;
        let (mem, _) = emulator.state.mem.mem_map_lookup_mut(cart_addr).unwrap();
        mem[..program.len()].copy_from_slice(&program);
        emulator.state.jump_to(cart_addr, ArmMode::ARM as i8);

        // kept always true so we can run a block at at time
        let exit_cond = |_: &ArmState| -> bool { true };

        emulator.run(exit_cond);
        assert!(emulator.func_cache.get(cart_addr).is_some());
        assert_eq!(emulator.state.curr_instr_addr(), cart_addr + 16);

        emulator.run(exit_cond);
        // first block was invalidated, second was not
        assert!(emulator.func_cache.get(cart_addr).is_none());
        assert!(emulator.func_cache.get(cart_addr + 16).is_some());
        assert_eq!(emulator.state.curr_instr_addr(), cart_addr);

        emulator.run(exit_cond);
        // Modified code ran
        assert_eq!(emulator.state.regs[Reg::R1], 24);
    }
}

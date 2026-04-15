#![allow(clippy::cast_sign_loss, clippy::cast_lossless)]

macro_rules! imm {
    ($self:ident, $i:expr) => {
        $self.i32_t.const_int($i as u64, false)
    };
}

macro_rules! imm8 {
    ($self:ident, $i:expr) => {
        $self.i8_t.const_int($i as u64, false)
    };
}

macro_rules! imm64 {
    ($self:ident, $i:expr) => {
        $self.ctx.i64_type().const_int($i as u64, false)
    };
}

macro_rules! call_intrinsic {
    ($builder:ident, $self:ident . $intrinsic:ident, $($args:expr),+) => {
        $builder
            .build_call(
                $self.$intrinsic,
                &[$($args.into()),+],
                &format!("{}_res", stringify!($intrinsic))
            )?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| anyhow!("failed to get {} return val", stringify!($intrinsic)))?
    };
}

macro_rules! call_indirect {
    ($builder:expr, $func_t:expr, $func_ptr:ident, $($args:expr),+) => {
        $builder
            .build_indirect_call(
                $func_t,
                $func_ptr,
                &[$($args.into()),+],
                &format!("{}_res", stringify!($func_ptr))
            )?
    };
}

macro_rules! exec_instr {
    ($self:ident, $wrapper:ident, $arg:ident, Self::$inner:ident) => {
        $self
            .$wrapper(&$arg, Self::$inner)
            .with_context(|| format!("{:?}", $arg))
            .expect("LLVM codegen failed")
    };
    ($self:ident, $wrapper:ident, $arg:ident, Self::$inner:ident::<$T:ty>) => {
        $self
            .$wrapper(&$arg, Self::$inner::<$T>)
            .with_context(|| format!("{:?}", $arg))
            .expect("LLVM codegen failed")
    };
}

mod alu;
mod branch;
pub mod cache;
mod flags;
mod load_store;
mod reg_map;
mod shift;

use std::collections::{HashMap, HashSet, VecDeque};
use std::{fs, ptr};

use anyhow::{Result, anyhow};
use capstone::arch::arm::ArmInsn;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::types::{IntType, PointerType, StructType, VoidType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use uuid::Uuid;

use crate::arm::disasm::InstrWindowIter;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::{ArmState, NUM_REGS, Reg};
use crate::emulator::{DebugOutput, DumpLLVM, EnvConfig};
use crate::jit::reg_map::{RegMap, RegMapItem};

macro_rules! unimpl_instr {
    ($instr:expr, $mnemonic:expr) => {
        panic!(
            "unimplemented instruction: {}",
            $instr
                .repr
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or($mnemonic)
        )
    };
}

/// Entrypoint into JIT-compiled code. Wraps the Inkwell JIT function and handles some other
/// updates required before calling into it.
pub struct CompiledFunction<'a> {
    pub start_addr: u32,
    pub end_addr: u32,
    inner: JitFunction<'a, unsafe extern "C" fn(*mut ArmState)>,
}

impl CompiledFunction<'_> {
    pub unsafe fn call(&self, state: &mut ArmState) {
        // `MemoryManager` needs to know where the currently executing function lies in memory
        // so we can return early in the event of self-modifying code.
        state
            .mem
            .set_curr_addr_range(self.start_addr, self.end_addr);
        unsafe {
            self.inner.call(ptr::from_mut(state));
        }
    }
}

/// Used to simulate instruction pipeline pre-fetch behaviour. When an instruction signals that we
/// may need to exit the currently running function based on the `exit` field on `InstrEffect`, we
/// don't check the condition right away but instead wait 2 more instruction as in a real machine
/// these will have been pre-fetched/decoded (how this affect the timing, I don't know).
#[derive(Copy, Clone)]
struct ExitCountdown<'a> {
    counter: u8,
    condition_var: IntValue<'a>,
}

impl<'a> ExitCountdown<'a> {
    fn new(condition_var: IntValue<'a>) -> Self {
        Self {
            counter: 2,
            condition_var,
        }
    }
}

#[derive(Copy, Clone)]
// A register and the value to write to it
struct RegUpdate<'a>(Reg, IntValue<'a>);

struct InstrEffect<'a> {
    updates: Vec<RegUpdate<'a>>,
    cycles: IntValue<'a>,
    exit: Option<IntValue<'a>>,
}

impl<'a> InstrEffect<'a> {
    fn new(updates: Vec<RegUpdate<'a>>, cycles: IntValue<'a>) -> Self {
        InstrEffect {
            updates,
            cycles,
            exit: None,
        }
    }

    fn with_exit(updates: Vec<RegUpdate<'a>>, cycles: IntValue<'a>, exit: IntValue<'a>) -> Self {
        InstrEffect {
            updates,
            cycles,
            exit: Some(exit),
        }
    }
}

type InstrResult<'a> = Result<InstrEffect<'a>>;

/// Builder for creating & compiling LLVM functions
pub struct FunctionBuilder<'ctx, 'a>
where
    'ctx: 'a,
{
    // Optional just to keep old tests working. A little bit clunky
    instr_iter: Option<InstrWindowIter<'a>>,
    exit_queue: VecDeque<ExitCountdown<'a>>,
    config: Option<EnvConfig>,

    // References to LLVM state
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    current_block: BasicBlock<'a>,

    name: String,
    func: FunctionValue<'a>,

    start_addr: u32,
    end_addr: u32,
    next_instr_addr: u32,
    pc_addr: u32,

    // Latest value for each register
    reg_map: RegMap<'a>,
    cycles: IntValue<'a>,

    // Pointers to emulated state
    arm_state_ptr: PointerValue<'a>,
    next_addr_ptr: PointerValue<'a>,
    mem_ptr: PointerValue<'a>,
    cycle_count_ptr: PointerValue<'a>,

    // Frequently used LLVM types
    arm_state_t: StructType<'a>,
    i8_t: IntType<'a>,
    i32_t: IntType<'a>,
    ptr_t: PointerType<'a>,
    void_t: VoidType<'a>,

    // LLVM intrinsics
    sadd_with_overflow: FunctionValue<'a>,
    ssub_with_overflow: FunctionValue<'a>,
    uadd_with_overflow: FunctionValue<'a>,
    usub_with_overflow: FunctionValue<'a>,
    fshr: FunctionValue<'a>,
}

fn get_ptr_param(func: FunctionValue<'_>, i: u32) -> Result<PointerValue<'_>> {
    func.get_nth_param(i)
        .ok_or(anyhow!(
            "{} signature has no parameter {}",
            func.get_name().to_str().unwrap(),
            i
        ))
        .map(BasicValueEnum::into_pointer_value)
}

fn get_intrinsic<'a>(name: &str, module: &Module<'a>) -> Result<FunctionValue<'a>> {
    Intrinsic::find(name)
        .ok_or(anyhow!("could not find intrinsic '{name}'"))?
        .get_declaration(module, &[module.get_context().i32_type().into()])
        .ok_or(anyhow!("failed to insert declaration for '{name}'"))
}

fn func_name(addr: u32) -> String { format!("fn_{addr:#010x}") }

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub fn new(ctx: &'ctx Context, addr: u32) -> Result<Self> {
        let bd = ctx.create_builder();
        let name = func_name(addr);
        let module = ctx.create_module(&name);
        // Less handles at least optimizing out `br i1 true, label %if, label %end`
        // branches that are emitted for AL cond
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Less)
            .expect("failed to create LLVM execution engine");

        let i8_t = ctx.i8_type();
        let i32_t = ctx.i32_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();

        // Access to the guest state is rather fragile and dependent on the exact memory
        // layout of ArmState.
        let arm_state_t = ctx.struct_type(
            &[
                i32_t.into(),                      // current_instr_addr
                i32_t.array_type(NUM_REGS).into(), // regs
                i32_t.into(),                      // cycle_count
                // This is obviously not the actual type of mem, but all we need is the address,
                // not the size, which is easy to get with an index, as long as we keep it at
                // the end of the struct
                i32_t.into(),
            ],
            false,
        );
        let fn_t = void_t.fn_type(&[ptr_t.into()], false);

        let func = module.add_function(&name, fn_t, None);
        let mut blocks = HashMap::new();
        let basic_block = ctx.append_basic_block(func, "start");
        bd.position_at_end(basic_block);
        blocks.insert(addr, basic_block);

        let arm_state_ptr = get_ptr_param(func, 0)?;
        let next_addr_ptr = unsafe {
            bd.build_gep(
                arm_state_t,
                arm_state_ptr,
                &[i32_t.const_zero(), i32_t.const_int(0, false)],
                "nxt_ptr",
            )?
        };
        let cycle_count_ptr = unsafe {
            bd.build_gep(
                arm_state_t,
                arm_state_ptr,
                &[i32_t.const_zero(), i32_t.const_int(2, false)],
                "cyc_ptr",
            )?
        };
        let mem_ptr = unsafe {
            bd.build_gep(
                arm_state_t,
                arm_state_ptr,
                &[i32_t.const_zero(), i32_t.const_int(3, false)],
                "mem_ptr",
            )?
        };
        // Declare intrinsics
        let sadd_with_overflow = get_intrinsic("llvm.sadd.with.overflow", &module)?;
        let ssub_with_overflow = get_intrinsic("llvm.ssub.with.overflow", &module)?;
        let uadd_with_overflow = get_intrinsic("llvm.uadd.with.overflow", &module)?;
        let usub_with_overflow = get_intrinsic("llvm.usub.with.overflow", &module)?;
        let fshr = get_intrinsic("llvm.fshr", &module)?;

        Ok(FunctionBuilder {
            ctx,
            name,
            func,
            start_addr: addr,
            end_addr: addr,
            next_instr_addr: addr,
            pc_addr: addr,
            reg_map: RegMap::new(),
            cycles: i32_t.const_zero(),
            arm_state_ptr,
            next_addr_ptr,
            mem_ptr,
            cycle_count_ptr,
            builder: bd,
            module,
            execution_engine,
            current_block: basic_block,
            arm_state_t,
            i8_t,
            i32_t,
            ptr_t,
            void_t,
            sadd_with_overflow,
            ssub_with_overflow,
            uadd_with_overflow,
            usub_with_overflow,
            fshr,
            exit_queue: VecDeque::new(),
            instr_iter: None,
            config: None,
        })
    }

    #[must_use]
    pub fn set_config(mut self, config: EnvConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build_body(mut self, instr_iter: InstrWindowIter<'a>) -> Result<Self> {
        let debug_output = self.config.as_ref().and_then(|c| c.debug_output);
        if debug_output.is_some() {
            println!("{}", self.name);
            println!("-------------------------");
        }
        self.instr_iter = Some(instr_iter);

        while let Some(instr) = self.instr_iter.as_mut().unwrap().next() {
            self.update_addresses(&instr);
            let new_regs = self.instr_iter.as_mut().unwrap().get_new_registers();
            self.load_initial_reg_values(&new_regs)
                .expect("loading initial register values failed");

            self.reg_map.update(Reg::PC, imm!(self, self.pc_addr));

            match debug_output {
                Some(DebugOutput::Struct) => println!("{instr:#?}"),
                Some(DebugOutput::Assembly) => println!("{instr}"),
                None => (),
            }

            if let Some(ex) = self.exit_queue.front()
                && ex.counter == 0
            {
                self.build_conditional_return(ex.condition_var)?;
                self.exit_queue.pop_front();
            }
            for ex in &mut self.exit_queue {
                ex.counter -= 1;
            }
            let returns = self.build(&instr);
            if returns {
                self.end_addr = instr.addr;
                break;
            }
        }
        if debug_output.is_some() {
            println!("-------------------------");
        }
        Ok(self)
    }

    pub fn compile(&self) -> Result<CompiledFunction<'ctx>> {
        let (dump_llvm, llvm_output_dir) =
            self.config.as_ref().map_or((None, String::new()), |c| {
                (c.dump_llvm, c.llvm_output_dir.clone())
            });

        if matches!(dump_llvm, Some(DumpLLVM::BeforeCompilation)) {
            self.dump_llvm(&llvm_output_dir);
        }
        if self.func.verify(true) {
            let compiled = unsafe {
                CompiledFunction {
                    start_addr: self.start_addr,
                    end_addr: self.end_addr,
                    inner: self.execution_engine.get_function(&self.name)?,
                }
            };
            if matches!(dump_llvm, Some(DumpLLVM::AfterCompilation)) {
                self.dump_llvm(&llvm_output_dir);
            }
            Ok(compiled)
        } else {
            if matches!(dump_llvm, Some(DumpLLVM::OnFail)) {
                self.dump_llvm(&llvm_output_dir);
            }
            Err(anyhow!("Function verification failed"))
        }
    }

    pub fn dump_llvm(&self, path: &str) {
        || -> Result<()> {
            if !fs::exists(path)? {
                fs::create_dir(path)?;
            }
            let fname = format!("{path}/mod_{}_{}.ll", self.name, Uuid::new_v4());
            self.module.print_to_file(fname).unwrap();
            Ok(())
        }()
        .inspect_err(|e| eprintln!("failed to dump generaetd llvm: {e}"))
        .ok();
    }

    fn load_initial_reg_values(&mut self, new_regs: &HashSet<Reg>) -> Result<()> {
        // initial register values are loaded lazily, as they are encountered
        let bd = &self.builder;
        for &r in new_regs {
            let i = r as usize;
            let name = format!("r{i}_elem_ptr");
            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_int(1, false),
                self.i32_t.const_int(i as u64, false),
            ];
            let ptr =
                unsafe { bd.build_gep(self.arm_state_t, self.arm_state_ptr, &gep_inds, &name)? };

            if r == Reg::PC {
                // Based on the start address passed to the Builder, not the in-memory value
                self.reg_map.init(r, imm!(self, self.pc_addr), ptr);
            } else {
                let name = format!("r{i}");
                let v = bd.build_load(self.i32_t, ptr, &name)?.into_int_value();
                self.reg_map.init(r, v, ptr);
            }
        }
        Ok(())
    }

    /// Advance the next instruction and pc addresses, falling back on a fixed 2/4 bytes step
    /// depending on mode. This could be incorrect if we reach the end a memory region (maybe
    /// handle this by always performing a branch if we get there?).
    fn update_addresses(&mut self, instr: &ArmInstruction) {
        self.next_instr_addr = self
            .instr_iter
            .as_ref()
            .unwrap()
            .peek(0)
            .map_or(instr.addr + instr.mode.pc_byte_offset(), |i| i.addr);

        self.pc_addr = self
            .instr_iter
            .as_ref()
            .unwrap()
            .peek(1)
            .map_or(self.next_instr_addr + instr.mode.pc_byte_offset(), |i| {
                i.addr
            });
    }

    fn get_external_func_pointer(&self, func_addr: usize) -> Result<PointerValue<'a>> {
        let ee = &self.execution_engine;
        let func_ptr = self.builder.build_int_to_ptr(
            self.ctx
                .ptr_sized_int_type(ee.get_target_data(), None)
                .const_int(func_addr as u64, false),
            self.ptr_t,
            "extern_ptr",
        )?;
        Ok(func_ptr)
    }

    /// When exiting the JIT'd code, write out the latest values in `reg_map` to the guest state. Note
    /// that in some cases (e.g. branch and link) we may want to save something other than
    /// `self.reg_map`, which is why it's passed as a parameter here (TBD if this is the case for
    /// cycles)
    fn write_state_out(&self, reg_map: &RegMap) -> Result<()> {
        let bd = &self.builder;
        let items: Vec<RegMapItem> = reg_map.items.iter().flatten().copied().collect();
        for r in items {
            if !r.dirty {
                continue;
            }
            bd.build_store(r.state_ptr, r.current_value)?;
        }
        bd.build_store(self.next_addr_ptr, imm!(self, self.next_instr_addr))?;
        let curr_count = bd
            .build_load(self.i32_t, self.cycle_count_ptr, "curr_cyc")?
            .into_int_value();
        bd.build_store(
            self.cycle_count_ptr,
            bd.build_int_add(curr_count, self.cycles, "upd_cyc")?,
        )?;
        Ok(())
    }

    /// Wrapper for all ARM instructions that evaluates cond, conditionally executes the instruction
    /// and performs register updates as necessary.
    fn exec_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<bool>
    where
        F: Fn(&Self, &ArmInstruction) -> InstrResult<'a>,
    {
        let bd = &self.builder;
        let if_block = self.ctx.append_basic_block(self.func, "if");
        let end_block = self.ctx.append_basic_block(self.func, "end");

        // If cond is not met (instruction not executed), just add 1 cycle to counter
        let unexec_cycles = self
            .builder
            .build_int_add(self.cycles, imm!(self, 1), "unexec_cyc")?;
        let cond = self.eval_cond(instr.cond)?;
        self.builder
            .build_conditional_branch(cond, if_block, end_block)?;

        // If cond is met, run inner and get set of updates to perform
        self.builder.position_at_end(if_block);

        let InstrEffect {
            updates,
            cycles,
            exit,
        } = inner(self, instr)?;

        let exec_cycles = self
            .builder
            .build_int_add(self.cycles, cycles, "exec_cyc")?;

        // If an instruction wrote to PC, we need to perform a branch
        let branch_target: Option<IntValue> = updates.iter().find(|r| r.0 == Reg::PC).map(|r| r.1);

        if let Some(target) = branch_target {
            let mut tmp_reg_map = self.reg_map.clone();
            for RegUpdate(reg, value) in updates {
                tmp_reg_map.update(reg, value);
            }
            self.write_state_out(&tmp_reg_map)?;
            // TODO - can we change mode here? Possibly an ARMv5 thing
            self.branch_and_return(target, imm8!(self, instr.mode as i8))?;

            self.builder.position_at_end(end_block);
            self.write_state_out(&self.reg_map)?;
            self.builder.build_return(None)?;

            return Ok(true);
        }

        self.builder.build_unconditional_branch(end_block)?;
        self.builder.position_at_end(end_block);

        // Update the values in reg_map, depending on which branch was taken. Because inner does
        // not mutate reg_map, we can use it to get the first option for phi values (cond not met).
        // The second option (cond met) comes from udpates.
        for RegUpdate(reg, value) in updates {
            let r_init = self.reg_map.get(reg);
            let r_new = value;
            let phi = bd.build_phi(self.i32_t, "phi")?;
            phi.add_incoming(&[(&r_init, self.current_block), (&r_new, if_block)]);
            self.reg_map
                .update(reg, phi.as_basic_value().into_int_value());
        }
        // Perform a similar update for cycle count.
        let cycle_phi = bd.build_phi(self.i32_t, "cyc_phi")?;
        cycle_phi.add_incoming(&[
            (&unexec_cycles, self.current_block),
            (&exec_cycles, if_block),
        ]);
        self.cycles = cycle_phi.as_basic_value().into_int_value();

        // Pre-fetch behaviour handling. See docs on `ExitCountdown`.
        // Emitting a branch for every single write seems less than ideal here, but it's probably
        // small compared to the overall write call.
        if let Some(condition_var) = exit {
            // Exit var must be defined even if we skipped executing the instructxion
            let exit_phi = bd.build_phi(self.i8_t, "exit_phi")?;
            exit_phi.add_incoming(&[
                (&imm8!(self, 0), self.current_block),
                (&condition_var, if_block),
            ]);
            let exit_var = exit_phi.as_basic_value().into_int_value();
            self.exit_queue.push_back(ExitCountdown::new(exit_var));
        }
        self.current_block = end_block;
        Ok(false)
    }

    // instr just used for it's address - there's probably a better approach
    pub fn build_conditional_return(&mut self, must_exit: IntValue<'a>) -> Result<()> {
        let bd = &self.builder;
        let if_block = self.ctx.append_basic_block(self.func, "if");
        let end_block = self.ctx.append_basic_block(self.func, "end");

        let cond =
            bd.build_int_compare(inkwell::IntPredicate::NE, must_exit, imm8!(self, 0), "exit")?;
        self.builder
            .build_conditional_branch(cond, if_block, end_block)?;

        self.builder.position_at_end(if_block);
        self.write_state_out(&self.reg_map)?;
        bd.build_return(None)?;

        self.builder.position_at_end(end_block);
        self.current_block = end_block;
        Ok(())
    }

    /// Returns true if the instruction terminates the function
    pub fn build(&mut self, instr: &ArmInstruction) -> bool {
        match instr.opcode {
            ArmInsn::ARM_INS_ADC => self.arm_adc(instr),
            ArmInsn::ARM_INS_ADD => self.arm_add(instr),
            ArmInsn::ARM_INS_ADR => self.arm_adr(instr),
            ArmInsn::ARM_INS_AND => self.arm_and(instr),
            ArmInsn::ARM_INS_ASR => self.arm_asr(instr),
            ArmInsn::ARM_INS_B => self.arm_b(instr),
            ArmInsn::ARM_INS_BIC => self.arm_bic(instr),
            ArmInsn::ARM_INS_BL => self.arm_bl(instr),
            ArmInsn::ARM_INS_BX => self.arm_bx(instr),
            ArmInsn::ARM_INS_CMN => self.arm_cmn(instr),
            ArmInsn::ARM_INS_CMP => self.arm_cmp(instr),
            ArmInsn::ARM_INS_EOR => self.arm_eor(instr),
            ArmInsn::ARM_INS_LDM => self.arm_ldmia(instr),
            ArmInsn::ARM_INS_LDMDA => self.arm_ldmda(instr),
            ArmInsn::ARM_INS_LDMDB => self.arm_ldmdb(instr),
            ArmInsn::ARM_INS_LDMIB => self.arm_ldmib(instr),
            ArmInsn::ARM_INS_LDR => self.arm_ldr(instr),
            ArmInsn::ARM_INS_LDRB => self.arm_ldrb(instr),
            ArmInsn::ARM_INS_LDRH => self.arm_ldrh(instr),
            ArmInsn::ARM_INS_LDRSB => self.arm_ldrsb(instr),
            ArmInsn::ARM_INS_LDRSH => self.arm_ldrsh(instr),
            ArmInsn::ARM_INS_LSL => self.arm_lsl(instr),
            ArmInsn::ARM_INS_LSR => self.arm_lsr(instr),
            ArmInsn::ARM_INS_MLA => self.arm_mla(instr),
            ArmInsn::ARM_INS_MOV => self.arm_mov(instr),
            ArmInsn::ARM_INS_MRS => self.arm_mrs(instr),
            ArmInsn::ARM_INS_MSR => self.arm_msr(instr),
            ArmInsn::ARM_INS_MUL => self.arm_mul(instr),
            ArmInsn::ARM_INS_MVN => self.arm_mvn(instr),
            ArmInsn::ARM_INS_ORR => self.arm_orr(instr),
            ArmInsn::ARM_INS_POP => self.arm_pop(instr),
            ArmInsn::ARM_INS_PUSH => self.arm_push(instr),
            ArmInsn::ARM_INS_ROR => self.arm_ror(instr),
            ArmInsn::ARM_INS_RRX => self.arm_rrx(instr),
            ArmInsn::ARM_INS_RSB => self.arm_rsb(instr),
            ArmInsn::ARM_INS_RSC => self.arm_rsc(instr),
            ArmInsn::ARM_INS_SBC => self.arm_sbc(instr),
            ArmInsn::ARM_INS_SMLAL => self.arm_smlal(instr),
            ArmInsn::ARM_INS_SMULL => self.arm_smull(instr),
            ArmInsn::ARM_INS_STM => self.arm_stmia(instr),
            ArmInsn::ARM_INS_STMIB => self.arm_stmib(instr),
            ArmInsn::ARM_INS_STMDA => self.arm_stmda(instr),
            ArmInsn::ARM_INS_STMDB => self.arm_stmdb(instr),
            ArmInsn::ARM_INS_STR => self.arm_str(instr),
            ArmInsn::ARM_INS_STRB => self.arm_strb(instr),
            ArmInsn::ARM_INS_STRH => self.arm_strh(instr),
            ArmInsn::ARM_INS_SUB => self.arm_sub(instr),
            // SWI?
            ArmInsn::ARM_INS_SWP => self.arm_swp(instr),
            ArmInsn::ARM_INS_SWPB => self.arm_swpb(instr),
            ArmInsn::ARM_INS_TEQ => self.arm_teq(instr),
            ArmInsn::ARM_INS_TST => self.arm_tst(instr),
            ArmInsn::ARM_INS_UMLAL => self.arm_umlal(instr),
            ArmInsn::ARM_INS_UMULL => self.arm_umull(instr),
            _ => unimpl_instr!(instr, "UNKNOWN"),
        }
    }
}

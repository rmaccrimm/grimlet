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

use std::collections::{HashMap, HashSet};
use std::fs;

use anyhow::{Context as _, Result, anyhow};
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

use crate::arm::disasm::code_block::CodeBlock;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::arm::state::{ArmMode, ArmState, NUM_REGS, Reg};
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

// JIT'd funxtions take a pointer to the guest machine state and return the number of (emulated)
// CPU cycles it took to execute
pub type CompiledFunction<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState) -> u32>;

/// Builder for creating & compiling LLVM functions
pub struct FunctionBuilder<'ctx, 'a>
where
    'ctx: 'a,
{
    name: String,
    func: FunctionValue<'a>,

    // Latest value for each register
    reg_map: RegMap<'a>,
    cycles: IntValue<'a>,

    // References to LLVM state
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    current_block: BasicBlock<'a>,

    // Pointers to emulated state
    arm_state_ptr: PointerValue<'a>,
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

struct InstrEffect<'a> {
    updates: Vec<RegUpdate<'a>>,
    cycles: IntValue<'a>,
    exit: bool,
}

type InstrResult<'a> = Result<InstrEffect<'a>>;

#[derive(Copy, Clone)]
// A register and the value to write to it
struct RegUpdate<'a>(Reg, IntValue<'a>);

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub fn new(ctx: &'ctx Context, addr: u32) -> Result<Self> {
        let bd = ctx.create_builder();
        let name = func_name(addr);
        let module = ctx.create_module(&name);
        // Less handles at least optimizing out `br i1 true, label %if, label %end`
        // branches that are emitted for AL cond
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("failed to create LLVM execution engine");

        let i8_t = ctx.i8_type();
        let i32_t = ctx.i32_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();

        // Access to the guest state is rather fragile and dependent on the exact memory
        // layout of ArmState.
        let arm_state_t = ctx.struct_type(
            &[
                i8_t.into(),                       // mode
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
        let cycle_count_ptr = unsafe {
            bd.build_gep(
                arm_state_t,
                arm_state_ptr,
                &[i32_t.const_zero(), i32_t.const_int(2, false)],
                "mem_ptr",
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
            reg_map: RegMap::new(),
            cycles: i32_t.const_zero(),
            arm_state_ptr,
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
        })
    }

    pub fn build_body(mut self, code_block: &CodeBlock) -> Result<Self> {
        self.load_initial_reg_values(&code_block.regs_accessed)
            .expect("initial register load failed");
        for instr in &code_block.instrs {
            self.build(instr);
        }
        Ok(self)
    }

    pub fn compile(self) -> Result<CompiledFunction<'ctx>> {
        if self.func.verify(true) {
            let res = unsafe { Ok(self.execution_engine.get_function(&self.name)?) };
            self.dump_llvm()
                .with_context(|| "Function verification failed: failed to dump LLVM code.")?;
            res
        } else {
            self.dump_llvm()
                .with_context(|| "Function verification failed: failed to dump LLVM code.")?;
            Err(anyhow!("Function verification failed"))
        }
    }

    fn dump_llvm(&self) -> Result<()> {
        if !fs::exists("llvm")? {
            fs::create_dir("llvm")?;
        }
        self.module
            .print_to_file(format!("llvm/mod_{}_{}.ll", self.name, Uuid::new_v4()))
            .unwrap();
        Ok(())
    }

    fn load_initial_reg_values(&mut self, regs_read: &HashSet<Reg>) -> Result<()> {
        let bd = &self.builder;
        for &r in regs_read {
            let i = r as usize;
            let name = format!("r{i}_elem_ptr");
            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_int(1, false),
                self.i32_t.const_int(i as u64, false),
            ];
            let ptr =
                unsafe { bd.build_gep(self.arm_state_t, self.arm_state_ptr, &gep_inds, &name)? };
            let name = format!("r{i}");
            let v = bd.build_load(self.i32_t, ptr, &name)?.into_int_value();
            self.reg_map.init(r, v, ptr);
        }
        Ok(())
    }

    fn increment_pc(&mut self, mode: ArmMode) {
        let curr_pc = self.reg_map.get(Reg::PC);
        let step = mode.instr_size();
        self.reg_map.update(
            Reg::PC,
            self.builder
                .build_int_add(curr_pc, imm!(self, step), "pc")
                .expect("LLVM codegen failed"),
        );
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
    fn exec_conditional<F>(&mut self, instr: &ArmInstruction, inner: F) -> Result<()>
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
            updates, cycles, ..
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
            self.increment_pc(instr.mode);
            self.write_state_out(&self.reg_map)?;
            self.builder.build_return(None)?;

            return Ok(());
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

        self.increment_pc(instr.mode);
        self.current_block = end_block;
        Ok(())
    }

    pub fn build(&mut self, instr: &ArmInstruction) {
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

impl<'a> InstrEffect<'a> {
    fn new(updates: Vec<RegUpdate<'a>>, cycles: IntValue<'a>) -> Self {
        InstrEffect {
            updates,
            cycles,
            exit: false,
        }
    }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::arm::state::{ArmState, REG_ITEMS, Reg};

    macro_rules! compile_and_run {
        ($func:ident, $state:ident) => {
            unsafe {
                $func.compile().unwrap().call(&mut $state);
            }
        };
    }

    #[test]
    fn test_jump_to_external() {
        // End result is:
        // pc <- r15 + r9
        let mut state = ArmState::default();
        for i in 0..NUM_REGS {
            state.regs[i as usize] = i * i;
        }

        let context = Context::create();
        let mut f = FunctionBuilder::new(&context, 0).unwrap();

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        f.load_initial_reg_values(&all_regs).unwrap();

        let add_res = f
            .builder
            .build_int_add(f.reg_map.get(Reg::PC), f.reg_map.get(Reg::R9), "add_res")
            .unwrap();

        let interp_fn_type = f
            .void_t
            .fn_type(&[f.ptr_t.into(), f.i32_t.into(), f.i8_t.into()], false);

        let interp_fn_ptr = f
            .get_external_func_pointer(ArmState::jump_to as fn(&mut ArmState, u32, i8) as usize)
            .unwrap();

        let call = f
            .builder
            .build_indirect_call(
                interp_fn_type,
                interp_fn_ptr,
                &[
                    f.arm_state_ptr.into(),
                    add_res.into(),
                    imm8!(f, ArmMode::ARM as i8).into(),
                ],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        f.builder.build_return(None).unwrap();

        println!("{:?}", state.regs);
        compile_and_run!(f, state);
        println!("{:?}", state.regs);
        assert_eq!(state.curr_instr_addr(), 306);
    }

    #[test]
    fn test_cross_module_calls() {
        // f1:
        //   pc <- r0 - r3 - r2
        // f2:
        //   r0 <- 999
        //   f1()
        let mut state = ArmState::default();
        for i in 0..NUM_REGS {
            state.regs[i as usize] = i * i;
        }

        let context = Context::create();
        let mut cache = HashMap::new();

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        let mut f1 = FunctionBuilder::new(&context, 0).unwrap();
        f1.load_initial_reg_values(&all_regs).unwrap();

        let r0 = f1.reg_map.get(Reg::R0);
        let r2 = f1.reg_map.get(Reg::R2);
        let r3 = f1.reg_map.get(Reg::R3);
        let v0 = f1.builder.build_int_add(r3, r2, "v0").unwrap();
        let v1 = f1.builder.build_int_sub(r0, v0, "v1").unwrap();

        // Perform context switch out before jumping to ArmState code
        f1.write_state_out(&f1.reg_map).unwrap();

        let func_ptr_param = f1
            .get_external_func_pointer(ArmState::jump_to as fn(&mut ArmState, u32, i8) as usize)
            .unwrap();

        let interp_fn_t = f1
            .void_t
            .fn_type(&[f1.ptr_t.into(), f1.i32_t.into(), f1.i8_t.into()], false);

        let call = f1
            .builder
            .build_indirect_call(
                interp_fn_t,
                func_ptr_param,
                &[
                    f1.arm_state_ptr.into(),
                    v1.into(),
                    imm8!(f1, ArmMode::ARM as i8).into(),
                ],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        f1.builder.build_return(None).unwrap();
        let compiled1 = f1.compile().unwrap();
        cache.insert(0, compiled1);

        let mut f2 = FunctionBuilder::new(&context, 1).unwrap();
        f2.load_initial_reg_values(&all_regs).unwrap();
        f2.reg_map.update(Reg::R0, f2.i32_t.const_int(999, false));
        f2.write_state_out(&f2.reg_map).unwrap();

        // Construct the function pointer using raw pointer obtained from function cache
        let func_ptr_param = unsafe {
            f2.builder
                .build_int_to_ptr(
                    f2.ctx
                        .ptr_sized_int_type(f2.execution_engine.get_target_data(), None)
                        // Double cast since usize ensures correct pointer size but inkwell
                        // expects u64
                        .const_int((cache.get(&0).unwrap().as_raw() as usize) as u64, false),
                    f2.ptr_t,
                    "f1_ptr",
                )
                .unwrap()
        };

        // Perform indirect call through pointer
        let fn_t = f2.void_t.fn_type(&[f2.ptr_t.into()], false);
        let call = f2
            .builder
            .build_indirect_call(fn_t, func_ptr_param, &[f2.arm_state_ptr.into()], "call")
            .unwrap();
        call.set_tail_call(true);
        f2.builder.build_return(None).unwrap();
        let compiled2 = f2.compile().unwrap();
        cache.insert(1, compiled2);

        println!("{:?}", state.regs);
        unsafe {
            cache.get(&1).unwrap().call(&raw mut state);
        }
        println!("{:?}", state.regs);

        assert_eq!(
            state.regs,
            // PC = jump target + 8 bytes
            //    = 994
            [
                999, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 994, 256, 289
            ]
        );
    }

    #[test]
    fn test_call_intrinsic() {
        let mut state = ArmState::default();
        let context = Context::create();
        let mut f = FunctionBuilder::new(&context, 0).unwrap();

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        f.load_initial_reg_values(&all_regs).unwrap();

        let bd = &f.builder;
        let call = bd
            .build_call(
                f.sadd_with_overflow,
                &[
                    f.i32_t.const_int(0x7fff_ffff_u64, false).into(),
                    f.i32_t.const_int(0xff, false).into(),
                ],
                "res",
            )
            .unwrap();
        let res = call
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_struct_value();

        let val = bd
            .build_extract_value(res, 0, "val")
            .unwrap()
            .into_int_value();
        let overflowed = bd
            .build_extract_value(res, 1, "overflowed")
            .unwrap()
            .into_int_value();

        f.reg_map.update(Reg::R0, val);
        f.reg_map.update(Reg::R1, overflowed);
        f.write_state_out(&f.reg_map).unwrap();
        bd.build_return(None).unwrap();
        compile_and_run!(f, state);

        println!("{:?}", state.regs);
        assert_eq!(state.regs[0], 0x8000_00fe);
        assert_eq!(state.regs[1], 1);
    }
}

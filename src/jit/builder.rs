mod alu;
mod branch;
mod flags;
mod reg_map;

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};
use capstone::arch::arm::ArmInsn;
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::types::{ArrayType, FunctionType, IntType, PointerType, StructType, VoidType};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};

use super::{CompiledFunction, FunctionCache};
use crate::arm::cpu::{ArmMode, ArmState, NUM_REGS, Reg};
use crate::arm::disasm::{ArmInstruction, CodeBlock};
use crate::jit::builder::reg_map::RegMap;

#[allow(dead_code)]
/// Builder for creating & compiling LLVM functions
pub struct FunctionBuilder<'ctx, 'a>
where
    'ctx: 'a,
{
    name: String,
    func: FunctionValue<'a>,

    // Latest value for each register
    reg_map: RegMap<'a>,

    // References to parent compiler LLVM state
    llvm_ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    execution_engine: &'a ExecutionEngine<'ctx>,
    current_block: BasicBlock<'a>,

    // Read only ref to already-compiled functions
    func_cache: &'a FunctionCache<'ctx>,

    // Function arguments
    arm_state_ptr: PointerValue<'a>,
    // reg_array_ptr: PointerValue<'a>,

    // Frequently used LLVM types
    arm_state_t: StructType<'a>,
    reg_array_t: ArrayType<'a>,
    fn_t: FunctionType<'a>,
    i32_t: IntType<'a>,
    ptr_t: PointerType<'a>,
    void_t: VoidType<'a>,
    bool_t: IntType<'a>,

    // Overflow arithmetic intrinsics
    sadd_with_overflow: FunctionValue<'a>,
    ssub_with_overflow: FunctionValue<'a>,
    uadd_with_overflow: FunctionValue<'a>,
    usub_with_overflow: FunctionValue<'a>,
}

pub(super) fn get_ptr_param<'a>(func: &FunctionValue<'a>, i: usize) -> Result<PointerValue<'a>> {
    func.get_nth_param(i as u32)
        .ok_or(anyhow!(
            "{} signature has no parameter {}",
            func.get_name().to_str().unwrap(),
            i
        ))
        .map(BasicValueEnum::into_pointer_value)
}

pub fn get_intrinsic<'a>(name: &str, module: &Module<'a>) -> Result<FunctionValue<'a>> {
    Intrinsic::find(name)
        .ok_or(anyhow!("could not find intrinsic '{}'", name))?
        .get_declaration(module, &[module.get_context().i32_type().into()])
        .ok_or(anyhow!("failed to insert declaration for '{}'", name))
}

pub fn func_name(addr: usize) -> String { format!("fn_{:#010x}", addr) }

impl<'ctx, 'a> FunctionBuilder<'ctx, 'a> {
    pub(super) fn new(
        addr: usize,
        llvm_ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        execution_engine: &'a ExecutionEngine<'ctx>,
        func_cache: &'a FunctionCache<'ctx>,
    ) -> Self {
        let build = || -> Result<Self> {
            let name = func_name(addr);
            let ctx = llvm_ctx;
            let i32_t = ctx.i32_type();
            let ptr_t = ctx.ptr_type(AddressSpace::default());
            let void_t = ctx.void_type();
            let bool_t = ctx.bool_type();
            let regs_t = i32_t.array_type(NUM_REGS as u32);
            let arm_state_t = ArmState::get_llvm_type(ctx);
            let fn_t = void_t.fn_type(&[ptr_t.into()], false);

            let bd = builder;
            let func = module.add_function(&name, fn_t, None);
            let mut blocks = HashMap::new();

            let basic_block = ctx.append_basic_block(func, "start");
            bd.position_at_end(basic_block);
            blocks.insert(addr, basic_block);

            let arm_state_ptr = get_ptr_param(&func, 0)?;

            // Declare intrinsics
            let sadd_with_overflow = get_intrinsic("llvm.sadd.with.overflow", module)?;
            let ssub_with_overflow = get_intrinsic("llvm.ssub.with.overflow", module)?;
            let uadd_with_overflow = get_intrinsic("llvm.uadd.with.overflow", module)?;
            let usub_with_overflow = get_intrinsic("llvm.usub.with.overflow", module)?;

            Ok(FunctionBuilder {
                name,
                reg_map: RegMap::new(),
                func,
                arm_state_ptr,
                // reg_array_ptr,
                llvm_ctx: ctx,
                builder,
                execution_engine,
                current_block: basic_block,
                func_cache,
                arm_state_t,
                fn_t,
                i32_t,
                ptr_t,
                reg_array_t: regs_t,
                void_t,
                bool_t,
                sadd_with_overflow,
                ssub_with_overflow,
                uadd_with_overflow,
                usub_with_overflow,
            })
        };
        build().expect("Function initialization failed")
    }

    pub fn build_body(mut self, mut code_block: CodeBlock) -> Self {
        // slightly hacky, just always load these 2 for simplicity
        code_block.regs_accessed.insert(Reg::CPSR);
        code_block.regs_accessed.insert(Reg::PC);
        self.load_initial_reg_values(&code_block.regs_accessed)
            .expect("initial register load failed");
        for instr in code_block.instrs {
            self.build(instr);
        }
        self
    }

    pub fn compile(self) -> Result<CompiledFunction<'ctx>> {
        if self.func.verify(true) {
            unsafe { Ok(self.execution_engine.get_function(&self.name)?) }
        } else {
            Err(anyhow!("Function verification failed"))
        }
    }

    fn load_initial_reg_values(&mut self, regs_read: &HashSet<Reg>) -> Result<()> {
        let bd = self.builder;
        for r in regs_read.iter() {
            let i = *r as usize;
            let name = format!("r{}_elem_ptr", i);
            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_int(1, false),
                self.i32_t.const_int(i as u64, false),
            ];
            let ptr =
                unsafe { bd.build_gep(self.arm_state_t, self.arm_state_ptr, &gep_inds, &name)? };
            let name = format!("r{}", i);
            let v = bd.build_load(self.i32_t, ptr, &name)?.into_int_value();
            // Go around the update method to not mark as dirty.
            // TODO - probably belongs in reg_map somewhere
            self.reg_map.llvm_values[i] = Some(v);
        }
        Ok(())
    }

    fn increment_pc(&mut self, mode: ArmMode) {
        let curr_pc = self.reg_map.pc();
        let step = match mode {
            ArmMode::ARM => 4,
            ArmMode::THUMB => 2,
        };
        self.reg_map.update(
            Reg::PC,
            self.builder
                .build_int_add(curr_pc, self.i32_t.const_int(step, false), "pc")
                .expect("LLVM codegen failed"),
        );
    }

    fn get_external_func_pointer(&self, func_addr: usize) -> Result<PointerValue<'a>> {
        let ee = &self.execution_engine;
        let func_ptr = self.builder.build_int_to_ptr(
            self.llvm_ctx
                .ptr_sized_int_type(ee.get_target_data(), None)
                .const_int(func_addr as u64, false),
            self.ptr_t,
            "extern_ptr",
        )?;
        Ok(func_ptr)
    }

    fn get_compiled_func_pointer(&self, key: usize) -> Result<Option<PointerValue<'a>>> {
        // TODO sort out which int type to use where
        match self.func_cache.get(&key) {
            Some(f) => {
                // pretty sure it doesn't matter which we look at
                let ee = &self.execution_engine;
                let func_ptr = unsafe {
                    self.builder.build_int_to_ptr(
                        self.llvm_ctx
                            .ptr_sized_int_type(ee.get_target_data(), None)
                            // Double cast since usize ensures correct pointer size but inkwell
                            // expects u64
                            .const_int((f.as_raw() as usize) as u64, false),
                        self.ptr_t,
                        &format!("{}_ptr", func_name(key)),
                    )?
                };
                Ok(Some(func_ptr))
            }
            None => Ok(None),
        }
    }

    /// When context switching, write out the latest values in reg_map to the guest state
    fn write_state_out(&self) -> Result<()> {
        let bd = &self.builder;
        for (i, r) in self.reg_map.llvm_values.iter().enumerate() {
            if !self.reg_map.dirty[i] {
                continue;
            }
            let name = format!("r{}_elem_ptr", i);
            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_int(1, false),
                self.i32_t.const_int(i as u64, false),
            ];
            let ptr =
                unsafe { bd.build_gep(self.arm_state_t, self.arm_state_ptr, &gep_inds, &name)? };

            bd.build_store(ptr, *r.as_ref().expect("reg has no value"))?;
        }
        Ok(())
    }

    pub fn build(&mut self, instr: ArmInstruction) {
        match instr.opcode {
            ArmInsn::ARM_INS_INVALID => todo!(),
            ArmInsn::ARM_INS_ADC => todo!(),
            ArmInsn::ARM_INS_ADD => todo!(),
            ArmInsn::ARM_INS_ADDW => todo!(),
            ArmInsn::ARM_INS_ADR => todo!(),
            ArmInsn::ARM_INS_AESD => todo!(),
            ArmInsn::ARM_INS_AESE => todo!(),
            ArmInsn::ARM_INS_AESIMC => todo!(),
            ArmInsn::ARM_INS_AESMC => todo!(),
            ArmInsn::ARM_INS_AND => todo!(),
            ArmInsn::ARM_INS_ASR => todo!(),
            ArmInsn::ARM_INS_B => self.arm_b(instr),
            ArmInsn::ARM_INS_BFC => todo!(),
            ArmInsn::ARM_INS_BFI => todo!(),
            ArmInsn::ARM_INS_BIC => todo!(),
            ArmInsn::ARM_INS_BKPT => todo!(),
            ArmInsn::ARM_INS_BL => todo!(),
            ArmInsn::ARM_INS_BLX => panic!("Unsupported ARMv5 insruction BXL"),
            ArmInsn::ARM_INS_BLXNS => todo!(),
            ArmInsn::ARM_INS_BX => todo!(),
            ArmInsn::ARM_INS_BXJ => todo!(),
            ArmInsn::ARM_INS_BXNS => todo!(),
            ArmInsn::ARM_INS_CBNZ => todo!(),
            ArmInsn::ARM_INS_CBZ => todo!(),
            ArmInsn::ARM_INS_CDP => todo!(),
            ArmInsn::ARM_INS_CDP2 => todo!(),
            ArmInsn::ARM_INS_CLREX => todo!(),
            ArmInsn::ARM_INS_CLZ => todo!(),
            ArmInsn::ARM_INS_CMN => todo!(),
            ArmInsn::ARM_INS_CMP => self.arm_cmp(instr),
            ArmInsn::ARM_INS_CPS => todo!(),
            ArmInsn::ARM_INS_CRC32B => todo!(),
            ArmInsn::ARM_INS_CRC32CB => todo!(),
            ArmInsn::ARM_INS_CRC32CH => todo!(),
            ArmInsn::ARM_INS_CRC32CW => todo!(),
            ArmInsn::ARM_INS_CRC32H => todo!(),
            ArmInsn::ARM_INS_CRC32W => todo!(),
            ArmInsn::ARM_INS_CSDB => todo!(),
            ArmInsn::ARM_INS_DBG => todo!(),
            ArmInsn::ARM_INS_DCPS1 => todo!(),
            ArmInsn::ARM_INS_DCPS2 => todo!(),
            ArmInsn::ARM_INS_DCPS3 => todo!(),
            ArmInsn::ARM_INS_DFB => todo!(),
            ArmInsn::ARM_INS_DMB => todo!(),
            ArmInsn::ARM_INS_DSB => todo!(),
            ArmInsn::ARM_INS_EOR => todo!(),
            ArmInsn::ARM_INS_ERET => todo!(),
            ArmInsn::ARM_INS_ESB => todo!(),
            ArmInsn::ARM_INS_FADDD => todo!(),
            ArmInsn::ARM_INS_FADDS => todo!(),
            ArmInsn::ARM_INS_FCMPZD => todo!(),
            ArmInsn::ARM_INS_FCMPZS => todo!(),
            ArmInsn::ARM_INS_FCONSTD => todo!(),
            ArmInsn::ARM_INS_FCONSTS => todo!(),
            ArmInsn::ARM_INS_FLDMDBX => todo!(),
            ArmInsn::ARM_INS_FLDMIAX => todo!(),
            ArmInsn::ARM_INS_FMDHR => todo!(),
            ArmInsn::ARM_INS_FMDLR => todo!(),
            ArmInsn::ARM_INS_FMSTAT => todo!(),
            ArmInsn::ARM_INS_FSTMDBX => todo!(),
            ArmInsn::ARM_INS_FSTMIAX => todo!(),
            ArmInsn::ARM_INS_FSUBD => todo!(),
            ArmInsn::ARM_INS_FSUBS => todo!(),
            ArmInsn::ARM_INS_HINT => todo!(),
            ArmInsn::ARM_INS_HLT => todo!(),
            ArmInsn::ARM_INS_HVC => todo!(),
            ArmInsn::ARM_INS_ISB => todo!(),
            ArmInsn::ARM_INS_IT => todo!(),
            ArmInsn::ARM_INS_LDA => todo!(),
            ArmInsn::ARM_INS_LDAB => todo!(),
            ArmInsn::ARM_INS_LDAEX => todo!(),
            ArmInsn::ARM_INS_LDAEXB => todo!(),
            ArmInsn::ARM_INS_LDAEXD => todo!(),
            ArmInsn::ARM_INS_LDAEXH => todo!(),
            ArmInsn::ARM_INS_LDAH => todo!(),
            ArmInsn::ARM_INS_LDC => todo!(),
            ArmInsn::ARM_INS_LDC2 => todo!(),
            ArmInsn::ARM_INS_LDC2L => todo!(),
            ArmInsn::ARM_INS_LDCL => todo!(),
            ArmInsn::ARM_INS_LDM => todo!(),
            ArmInsn::ARM_INS_LDMDA => todo!(),
            ArmInsn::ARM_INS_LDMDB => todo!(),
            ArmInsn::ARM_INS_LDMIB => todo!(),
            ArmInsn::ARM_INS_LDR => todo!(),
            ArmInsn::ARM_INS_LDRB => todo!(),
            ArmInsn::ARM_INS_LDRBT => todo!(),
            ArmInsn::ARM_INS_LDRD => todo!(),
            ArmInsn::ARM_INS_LDREX => todo!(),
            ArmInsn::ARM_INS_LDREXB => todo!(),
            ArmInsn::ARM_INS_LDREXD => todo!(),
            ArmInsn::ARM_INS_LDREXH => todo!(),
            ArmInsn::ARM_INS_LDRH => todo!(),
            ArmInsn::ARM_INS_LDRHT => todo!(),
            ArmInsn::ARM_INS_LDRSB => todo!(),
            ArmInsn::ARM_INS_LDRSBT => todo!(),
            ArmInsn::ARM_INS_LDRSH => todo!(),
            ArmInsn::ARM_INS_LDRSHT => todo!(),
            ArmInsn::ARM_INS_LDRT => todo!(),
            ArmInsn::ARM_INS_LSL => todo!(),
            ArmInsn::ARM_INS_LSR => todo!(),
            ArmInsn::ARM_INS_MCR => todo!(),
            ArmInsn::ARM_INS_MCR2 => todo!(),
            ArmInsn::ARM_INS_MCRR => todo!(),
            ArmInsn::ARM_INS_MCRR2 => todo!(),
            ArmInsn::ARM_INS_MLA => todo!(),
            ArmInsn::ARM_INS_MLS => todo!(),
            ArmInsn::ARM_INS_MOV => self.arm_mov(instr),
            ArmInsn::ARM_INS_MOVS => todo!(),
            ArmInsn::ARM_INS_MOVT => todo!(),
            ArmInsn::ARM_INS_MOVW => todo!(),
            ArmInsn::ARM_INS_MRC => todo!(),
            ArmInsn::ARM_INS_MRC2 => todo!(),
            ArmInsn::ARM_INS_MRRC => todo!(),
            ArmInsn::ARM_INS_MRRC2 => todo!(),
            ArmInsn::ARM_INS_MRS => todo!(),
            ArmInsn::ARM_INS_MSR => todo!(),
            ArmInsn::ARM_INS_MUL => self.arm_mul(instr),
            ArmInsn::ARM_INS_MVN => todo!(),
            ArmInsn::ARM_INS_NEG => todo!(),
            ArmInsn::ARM_INS_NOP => todo!(),
            ArmInsn::ARM_INS_ORN => todo!(),
            ArmInsn::ARM_INS_ORR => todo!(),
            ArmInsn::ARM_INS_PKHBT => todo!(),
            ArmInsn::ARM_INS_PKHTB => todo!(),
            ArmInsn::ARM_INS_PLD => todo!(),
            ArmInsn::ARM_INS_PLDW => todo!(),
            ArmInsn::ARM_INS_PLI => todo!(),
            ArmInsn::ARM_INS_POP => todo!(),
            ArmInsn::ARM_INS_PUSH => todo!(),
            ArmInsn::ARM_INS_QADD => todo!(),
            ArmInsn::ARM_INS_QADD16 => todo!(),
            ArmInsn::ARM_INS_QADD8 => todo!(),
            ArmInsn::ARM_INS_QASX => todo!(),
            ArmInsn::ARM_INS_QDADD => todo!(),
            ArmInsn::ARM_INS_QDSUB => todo!(),
            ArmInsn::ARM_INS_QSAX => todo!(),
            ArmInsn::ARM_INS_QSUB => todo!(),
            ArmInsn::ARM_INS_QSUB16 => todo!(),
            ArmInsn::ARM_INS_QSUB8 => todo!(),
            ArmInsn::ARM_INS_RBIT => todo!(),
            ArmInsn::ARM_INS_REV => todo!(),
            ArmInsn::ARM_INS_REV16 => todo!(),
            ArmInsn::ARM_INS_REVSH => todo!(),
            ArmInsn::ARM_INS_RFEDA => todo!(),
            ArmInsn::ARM_INS_RFEDB => todo!(),
            ArmInsn::ARM_INS_RFEIA => todo!(),
            ArmInsn::ARM_INS_RFEIB => todo!(),
            ArmInsn::ARM_INS_ROR => todo!(),
            ArmInsn::ARM_INS_RRX => todo!(),
            ArmInsn::ARM_INS_RSB => todo!(),
            ArmInsn::ARM_INS_RSC => todo!(),
            ArmInsn::ARM_INS_SADD16 => todo!(),
            ArmInsn::ARM_INS_SADD8 => todo!(),
            ArmInsn::ARM_INS_SASX => todo!(),
            ArmInsn::ARM_INS_SBC => todo!(),
            ArmInsn::ARM_INS_SBFX => todo!(),
            ArmInsn::ARM_INS_SDIV => todo!(),
            ArmInsn::ARM_INS_SEL => todo!(),
            ArmInsn::ARM_INS_SETEND => todo!(),
            ArmInsn::ARM_INS_SETPAN => todo!(),
            ArmInsn::ARM_INS_SEV => todo!(),
            ArmInsn::ARM_INS_SEVL => todo!(),
            ArmInsn::ARM_INS_SG => todo!(),
            ArmInsn::ARM_INS_SHA1C => todo!(),
            ArmInsn::ARM_INS_SHA1H => todo!(),
            ArmInsn::ARM_INS_SHA1M => todo!(),
            ArmInsn::ARM_INS_SHA1P => todo!(),
            ArmInsn::ARM_INS_SHA1SU0 => todo!(),
            ArmInsn::ARM_INS_SHA1SU1 => todo!(),
            ArmInsn::ARM_INS_SHA256H => todo!(),
            ArmInsn::ARM_INS_SHA256H2 => todo!(),
            ArmInsn::ARM_INS_SHA256SU0 => todo!(),
            ArmInsn::ARM_INS_SHA256SU1 => todo!(),
            ArmInsn::ARM_INS_SHADD16 => todo!(),
            ArmInsn::ARM_INS_SHADD8 => todo!(),
            ArmInsn::ARM_INS_SHASX => todo!(),
            ArmInsn::ARM_INS_SHSAX => todo!(),
            ArmInsn::ARM_INS_SHSUB16 => todo!(),
            ArmInsn::ARM_INS_SHSUB8 => todo!(),
            ArmInsn::ARM_INS_SMC => todo!(),
            ArmInsn::ARM_INS_SMLABB => todo!(),
            ArmInsn::ARM_INS_SMLABT => todo!(),
            ArmInsn::ARM_INS_SMLAD => todo!(),
            ArmInsn::ARM_INS_SMLADX => todo!(),
            ArmInsn::ARM_INS_SMLAL => todo!(),
            ArmInsn::ARM_INS_SMLALBB => todo!(),
            ArmInsn::ARM_INS_SMLALBT => todo!(),
            ArmInsn::ARM_INS_SMLALD => todo!(),
            ArmInsn::ARM_INS_SMLALDX => todo!(),
            ArmInsn::ARM_INS_SMLALTB => todo!(),
            ArmInsn::ARM_INS_SMLALTT => todo!(),
            ArmInsn::ARM_INS_SMLATB => todo!(),
            ArmInsn::ARM_INS_SMLATT => todo!(),
            ArmInsn::ARM_INS_SMLAWB => todo!(),
            ArmInsn::ARM_INS_SMLAWT => todo!(),
            ArmInsn::ARM_INS_SMLSD => todo!(),
            ArmInsn::ARM_INS_SMLSDX => todo!(),
            ArmInsn::ARM_INS_SMLSLD => todo!(),
            ArmInsn::ARM_INS_SMLSLDX => todo!(),
            ArmInsn::ARM_INS_SMMLA => todo!(),
            ArmInsn::ARM_INS_SMMLAR => todo!(),
            ArmInsn::ARM_INS_SMMLS => todo!(),
            ArmInsn::ARM_INS_SMMLSR => todo!(),
            ArmInsn::ARM_INS_SMMUL => todo!(),
            ArmInsn::ARM_INS_SMMULR => todo!(),
            ArmInsn::ARM_INS_SMUAD => todo!(),
            ArmInsn::ARM_INS_SMUADX => todo!(),
            ArmInsn::ARM_INS_SMULBB => todo!(),
            ArmInsn::ARM_INS_SMULBT => todo!(),
            ArmInsn::ARM_INS_SMULL => todo!(),
            ArmInsn::ARM_INS_SMULTB => todo!(),
            ArmInsn::ARM_INS_SMULTT => todo!(),
            ArmInsn::ARM_INS_SMULWB => todo!(),
            ArmInsn::ARM_INS_SMULWT => todo!(),
            ArmInsn::ARM_INS_SMUSD => todo!(),
            ArmInsn::ARM_INS_SMUSDX => todo!(),
            ArmInsn::ARM_INS_SRSDA => todo!(),
            ArmInsn::ARM_INS_SRSDB => todo!(),
            ArmInsn::ARM_INS_SRSIA => todo!(),
            ArmInsn::ARM_INS_SRSIB => todo!(),
            ArmInsn::ARM_INS_SSAT => todo!(),
            ArmInsn::ARM_INS_SSAT16 => todo!(),
            ArmInsn::ARM_INS_SSAX => todo!(),
            ArmInsn::ARM_INS_SSUB16 => todo!(),
            ArmInsn::ARM_INS_SSUB8 => todo!(),
            ArmInsn::ARM_INS_STC => todo!(),
            ArmInsn::ARM_INS_STC2 => todo!(),
            ArmInsn::ARM_INS_STC2L => todo!(),
            ArmInsn::ARM_INS_STCL => todo!(),
            ArmInsn::ARM_INS_STL => todo!(),
            ArmInsn::ARM_INS_STLB => todo!(),
            ArmInsn::ARM_INS_STLEX => todo!(),
            ArmInsn::ARM_INS_STLEXB => todo!(),
            ArmInsn::ARM_INS_STLEXD => todo!(),
            ArmInsn::ARM_INS_STLEXH => todo!(),
            ArmInsn::ARM_INS_STLH => todo!(),
            ArmInsn::ARM_INS_STM => todo!(),
            ArmInsn::ARM_INS_STMDA => todo!(),
            ArmInsn::ARM_INS_STMDB => todo!(),
            ArmInsn::ARM_INS_STMIB => todo!(),
            ArmInsn::ARM_INS_STR => todo!(),
            ArmInsn::ARM_INS_STRB => todo!(),
            ArmInsn::ARM_INS_STRBT => todo!(),
            ArmInsn::ARM_INS_STRD => todo!(),
            ArmInsn::ARM_INS_STREX => todo!(),
            ArmInsn::ARM_INS_STREXB => todo!(),
            ArmInsn::ARM_INS_STREXD => todo!(),
            ArmInsn::ARM_INS_STREXH => todo!(),
            ArmInsn::ARM_INS_STRH => todo!(),
            ArmInsn::ARM_INS_STRHT => todo!(),
            ArmInsn::ARM_INS_STRT => todo!(),
            ArmInsn::ARM_INS_SUB => self.arm_sub(instr),
            ArmInsn::ARM_INS_SUBS => self.arm_sub(instr),
            ArmInsn::ARM_INS_SUBW => todo!(),
            ArmInsn::ARM_INS_SVC => todo!(),
            ArmInsn::ARM_INS_SWP => todo!(),
            ArmInsn::ARM_INS_SWPB => todo!(),
            ArmInsn::ARM_INS_SXTAB => todo!(),
            ArmInsn::ARM_INS_SXTAB16 => todo!(),
            ArmInsn::ARM_INS_SXTAH => todo!(),
            ArmInsn::ARM_INS_SXTB => todo!(),
            ArmInsn::ARM_INS_SXTB16 => todo!(),
            ArmInsn::ARM_INS_SXTH => todo!(),
            ArmInsn::ARM_INS_TBB => todo!(),
            ArmInsn::ARM_INS_TBH => todo!(),
            ArmInsn::ARM_INS_TEQ => todo!(),
            ArmInsn::ARM_INS_TRAP => todo!(),
            ArmInsn::ARM_INS_TSB => todo!(),
            ArmInsn::ARM_INS_TST => todo!(),
            ArmInsn::ARM_INS_TT => todo!(),
            ArmInsn::ARM_INS_TTA => todo!(),
            ArmInsn::ARM_INS_TTAT => todo!(),
            ArmInsn::ARM_INS_TTT => todo!(),
            ArmInsn::ARM_INS_UADD16 => todo!(),
            ArmInsn::ARM_INS_UADD8 => todo!(),
            ArmInsn::ARM_INS_UASX => todo!(),
            ArmInsn::ARM_INS_UBFX => todo!(),
            ArmInsn::ARM_INS_UDF => todo!(),
            ArmInsn::ARM_INS_UDIV => todo!(),
            ArmInsn::ARM_INS_UHADD16 => todo!(),
            ArmInsn::ARM_INS_UHADD8 => todo!(),
            ArmInsn::ARM_INS_UHASX => todo!(),
            ArmInsn::ARM_INS_UHSAX => todo!(),
            ArmInsn::ARM_INS_UHSUB16 => todo!(),
            ArmInsn::ARM_INS_UHSUB8 => todo!(),
            ArmInsn::ARM_INS_UMAAL => todo!(),
            ArmInsn::ARM_INS_UMLAL => todo!(),
            ArmInsn::ARM_INS_UMULL => todo!(),
            ArmInsn::ARM_INS_UQADD16 => todo!(),
            ArmInsn::ARM_INS_UQADD8 => todo!(),
            ArmInsn::ARM_INS_UQASX => todo!(),
            ArmInsn::ARM_INS_UQSAX => todo!(),
            ArmInsn::ARM_INS_UQSUB16 => todo!(),
            ArmInsn::ARM_INS_UQSUB8 => todo!(),
            ArmInsn::ARM_INS_USAD8 => todo!(),
            ArmInsn::ARM_INS_USADA8 => todo!(),
            ArmInsn::ARM_INS_USAT => todo!(),
            ArmInsn::ARM_INS_USAT16 => todo!(),
            ArmInsn::ARM_INS_USAX => todo!(),
            ArmInsn::ARM_INS_USUB16 => todo!(),
            ArmInsn::ARM_INS_USUB8 => todo!(),
            ArmInsn::ARM_INS_UXTAB => todo!(),
            ArmInsn::ARM_INS_UXTAB16 => todo!(),
            ArmInsn::ARM_INS_UXTAH => todo!(),
            ArmInsn::ARM_INS_UXTB => todo!(),
            ArmInsn::ARM_INS_UXTB16 => todo!(),
            ArmInsn::ARM_INS_UXTH => todo!(),
            ArmInsn::ARM_INS_VABA => todo!(),
            ArmInsn::ARM_INS_VABAL => todo!(),
            ArmInsn::ARM_INS_VABD => todo!(),
            ArmInsn::ARM_INS_VABDL => todo!(),
            ArmInsn::ARM_INS_VABS => todo!(),
            ArmInsn::ARM_INS_VACGE => todo!(),
            ArmInsn::ARM_INS_VACGT => todo!(),
            ArmInsn::ARM_INS_VACLE => todo!(),
            ArmInsn::ARM_INS_VACLT => todo!(),
            ArmInsn::ARM_INS_VADD => todo!(),
            ArmInsn::ARM_INS_VADDHN => todo!(),
            ArmInsn::ARM_INS_VADDL => todo!(),
            ArmInsn::ARM_INS_VADDW => todo!(),
            ArmInsn::ARM_INS_VAND => todo!(),
            ArmInsn::ARM_INS_VBIC => todo!(),
            ArmInsn::ARM_INS_VBIF => todo!(),
            ArmInsn::ARM_INS_VBIT => todo!(),
            ArmInsn::ARM_INS_VBSL => todo!(),
            ArmInsn::ARM_INS_VCADD => todo!(),
            ArmInsn::ARM_INS_VCEQ => todo!(),
            ArmInsn::ARM_INS_VCGE => todo!(),
            ArmInsn::ARM_INS_VCGT => todo!(),
            ArmInsn::ARM_INS_VCLE => todo!(),
            ArmInsn::ARM_INS_VCLS => todo!(),
            ArmInsn::ARM_INS_VCLT => todo!(),
            ArmInsn::ARM_INS_VCLZ => todo!(),
            ArmInsn::ARM_INS_VCMLA => todo!(),
            ArmInsn::ARM_INS_VCMP => todo!(),
            ArmInsn::ARM_INS_VCMPE => todo!(),
            ArmInsn::ARM_INS_VCNT => todo!(),
            ArmInsn::ARM_INS_VCVT => todo!(),
            ArmInsn::ARM_INS_VCVTA => todo!(),
            ArmInsn::ARM_INS_VCVTB => todo!(),
            ArmInsn::ARM_INS_VCVTM => todo!(),
            ArmInsn::ARM_INS_VCVTN => todo!(),
            ArmInsn::ARM_INS_VCVTP => todo!(),
            ArmInsn::ARM_INS_VCVTR => todo!(),
            ArmInsn::ARM_INS_VCVTT => todo!(),
            ArmInsn::ARM_INS_VDIV => todo!(),
            ArmInsn::ARM_INS_VDUP => todo!(),
            ArmInsn::ARM_INS_VEOR => todo!(),
            ArmInsn::ARM_INS_VEXT => todo!(),
            ArmInsn::ARM_INS_VFMA => todo!(),
            ArmInsn::ARM_INS_VFMS => todo!(),
            ArmInsn::ARM_INS_VFNMA => todo!(),
            ArmInsn::ARM_INS_VFNMS => todo!(),
            ArmInsn::ARM_INS_VHADD => todo!(),
            ArmInsn::ARM_INS_VHSUB => todo!(),
            ArmInsn::ARM_INS_VINS => todo!(),
            ArmInsn::ARM_INS_VJCVT => todo!(),
            ArmInsn::ARM_INS_VLD1 => todo!(),
            ArmInsn::ARM_INS_VLD2 => todo!(),
            ArmInsn::ARM_INS_VLD3 => todo!(),
            ArmInsn::ARM_INS_VLD4 => todo!(),
            ArmInsn::ARM_INS_VLDMDB => todo!(),
            ArmInsn::ARM_INS_VLDMIA => todo!(),
            ArmInsn::ARM_INS_VLDR => todo!(),
            ArmInsn::ARM_INS_VLLDM => todo!(),
            ArmInsn::ARM_INS_VLSTM => todo!(),
            ArmInsn::ARM_INS_VMAX => todo!(),
            ArmInsn::ARM_INS_VMAXNM => todo!(),
            ArmInsn::ARM_INS_VMIN => todo!(),
            ArmInsn::ARM_INS_VMINNM => todo!(),
            ArmInsn::ARM_INS_VMLA => todo!(),
            ArmInsn::ARM_INS_VMLAL => todo!(),
            ArmInsn::ARM_INS_VMLS => todo!(),
            ArmInsn::ARM_INS_VMLSL => todo!(),
            ArmInsn::ARM_INS_VMOV => todo!(),
            ArmInsn::ARM_INS_VMOVL => todo!(),
            ArmInsn::ARM_INS_VMOVN => todo!(),
            ArmInsn::ARM_INS_VMOVX => todo!(),
            ArmInsn::ARM_INS_VMRS => todo!(),
            ArmInsn::ARM_INS_VMSR => todo!(),
            ArmInsn::ARM_INS_VMUL => todo!(),
            ArmInsn::ARM_INS_VMULL => todo!(),
            ArmInsn::ARM_INS_VMVN => todo!(),
            ArmInsn::ARM_INS_VNEG => todo!(),
            ArmInsn::ARM_INS_VNMLA => todo!(),
            ArmInsn::ARM_INS_VNMLS => todo!(),
            ArmInsn::ARM_INS_VNMUL => todo!(),
            ArmInsn::ARM_INS_VORN => todo!(),
            ArmInsn::ARM_INS_VORR => todo!(),
            ArmInsn::ARM_INS_VPADAL => todo!(),
            ArmInsn::ARM_INS_VPADD => todo!(),
            ArmInsn::ARM_INS_VPADDL => todo!(),
            ArmInsn::ARM_INS_VPMAX => todo!(),
            ArmInsn::ARM_INS_VPMIN => todo!(),
            ArmInsn::ARM_INS_VPOP => todo!(),
            ArmInsn::ARM_INS_VPUSH => todo!(),
            ArmInsn::ARM_INS_VQABS => todo!(),
            ArmInsn::ARM_INS_VQADD => todo!(),
            ArmInsn::ARM_INS_VQDMLAL => todo!(),
            ArmInsn::ARM_INS_VQDMLSL => todo!(),
            ArmInsn::ARM_INS_VQDMULH => todo!(),
            ArmInsn::ARM_INS_VQDMULL => todo!(),
            ArmInsn::ARM_INS_VQMOVN => todo!(),
            ArmInsn::ARM_INS_VQMOVUN => todo!(),
            ArmInsn::ARM_INS_VQNEG => todo!(),
            ArmInsn::ARM_INS_VQRDMLAH => todo!(),
            ArmInsn::ARM_INS_VQRDMLSH => todo!(),
            ArmInsn::ARM_INS_VQRDMULH => todo!(),
            ArmInsn::ARM_INS_VQRSHL => todo!(),
            ArmInsn::ARM_INS_VQRSHRN => todo!(),
            ArmInsn::ARM_INS_VQRSHRUN => todo!(),
            ArmInsn::ARM_INS_VQSHL => todo!(),
            ArmInsn::ARM_INS_VQSHLU => todo!(),
            ArmInsn::ARM_INS_VQSHRN => todo!(),
            ArmInsn::ARM_INS_VQSHRUN => todo!(),
            ArmInsn::ARM_INS_VQSUB => todo!(),
            ArmInsn::ARM_INS_VRADDHN => todo!(),
            ArmInsn::ARM_INS_VRECPE => todo!(),
            ArmInsn::ARM_INS_VRECPS => todo!(),
            ArmInsn::ARM_INS_VREV16 => todo!(),
            ArmInsn::ARM_INS_VREV32 => todo!(),
            ArmInsn::ARM_INS_VREV64 => todo!(),
            ArmInsn::ARM_INS_VRHADD => todo!(),
            ArmInsn::ARM_INS_VRINTA => todo!(),
            ArmInsn::ARM_INS_VRINTM => todo!(),
            ArmInsn::ARM_INS_VRINTN => todo!(),
            ArmInsn::ARM_INS_VRINTP => todo!(),
            ArmInsn::ARM_INS_VRINTR => todo!(),
            ArmInsn::ARM_INS_VRINTX => todo!(),
            ArmInsn::ARM_INS_VRINTZ => todo!(),
            ArmInsn::ARM_INS_VRSHL => todo!(),
            ArmInsn::ARM_INS_VRSHR => todo!(),
            ArmInsn::ARM_INS_VRSHRN => todo!(),
            ArmInsn::ARM_INS_VRSQRTE => todo!(),
            ArmInsn::ARM_INS_VRSQRTS => todo!(),
            ArmInsn::ARM_INS_VRSRA => todo!(),
            ArmInsn::ARM_INS_VRSUBHN => todo!(),
            ArmInsn::ARM_INS_VSDOT => todo!(),
            ArmInsn::ARM_INS_VSELEQ => todo!(),
            ArmInsn::ARM_INS_VSELGE => todo!(),
            ArmInsn::ARM_INS_VSELGT => todo!(),
            ArmInsn::ARM_INS_VSELVS => todo!(),
            ArmInsn::ARM_INS_VSHL => todo!(),
            ArmInsn::ARM_INS_VSHLL => todo!(),
            ArmInsn::ARM_INS_VSHR => todo!(),
            ArmInsn::ARM_INS_VSHRN => todo!(),
            ArmInsn::ARM_INS_VSLI => todo!(),
            ArmInsn::ARM_INS_VSQRT => todo!(),
            ArmInsn::ARM_INS_VSRA => todo!(),
            ArmInsn::ARM_INS_VSRI => todo!(),
            ArmInsn::ARM_INS_VST1 => todo!(),
            ArmInsn::ARM_INS_VST2 => todo!(),
            ArmInsn::ARM_INS_VST3 => todo!(),
            ArmInsn::ARM_INS_VST4 => todo!(),
            ArmInsn::ARM_INS_VSTMDB => todo!(),
            ArmInsn::ARM_INS_VSTMIA => todo!(),
            ArmInsn::ARM_INS_VSTR => todo!(),
            ArmInsn::ARM_INS_VSUB => todo!(),
            ArmInsn::ARM_INS_VSUBHN => todo!(),
            ArmInsn::ARM_INS_VSUBL => todo!(),
            ArmInsn::ARM_INS_VSUBW => todo!(),
            ArmInsn::ARM_INS_VSWP => todo!(),
            ArmInsn::ARM_INS_VTBL => todo!(),
            ArmInsn::ARM_INS_VTBX => todo!(),
            ArmInsn::ARM_INS_VTRN => todo!(),
            ArmInsn::ARM_INS_VTST => todo!(),
            ArmInsn::ARM_INS_VUDOT => todo!(),
            ArmInsn::ARM_INS_VUZP => todo!(),
            ArmInsn::ARM_INS_VZIP => todo!(),
            ArmInsn::ARM_INS_WFE => todo!(),
            ArmInsn::ARM_INS_WFI => todo!(),
            ArmInsn::ARM_INS_YIELD => todo!(),
            ArmInsn::ARM_INS_ENDING => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::arm::cpu::{REG_ITEMS, Reg};
    use crate::jit::Compiler;

    macro_rules! compile_and_run {
        ($compiler:ident, $func:ident, $state:ident) => {
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
            state.regs[i] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context);
        let func_cache = HashMap::new();
        let mut f = comp.new_function(0, &func_cache);

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        f.load_initial_reg_values(&all_regs).unwrap();

        let add_res = f
            .builder
            .build_int_add(f.reg_map.get(Reg::PC), f.reg_map.get(Reg::R9), "add_res")
            .unwrap();

        let interp_fn_type = f.void_t.fn_type(&[f.ptr_t.into(), f.i32_t.into()], false);

        let interp_fn_ptr = f
            .get_external_func_pointer(ArmState::jump_to as *const (&mut ArmState, u32) as usize)
            .unwrap();

        let call = f
            .builder
            .build_indirect_call(
                interp_fn_type,
                interp_fn_ptr,
                &[f.arm_state_ptr.into(), add_res.into()],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        f.builder.build_return(None).unwrap();

        println!("{:?}", state.regs);
        compile_and_run!(comp, f, state);
        println!("{:?}", state.regs);
        assert_eq!(state.pc(), 306);
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
            state.regs[i] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context);
        let mut cache = HashMap::new();

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        let mut f1 = comp.new_function(0, &cache);
        f1.load_initial_reg_values(&all_regs).unwrap();

        let r0 = f1.reg_map.get(Reg::R0);
        let r2 = f1.reg_map.get(Reg::R2);
        let r3 = f1.reg_map.get(Reg::R3);
        let v0 = f1.builder.build_int_add(r3, r2, "v0").unwrap();
        let v1 = f1.builder.build_int_sub(r0, v0, "v1").unwrap();

        // Perform context switch out before jumping to ArmState code
        f1.write_state_out().unwrap();

        let func_ptr_param = f1
            .get_external_func_pointer(ArmState::jump_to as *const (&mut ArmState, u32) as usize)
            .unwrap();

        let interp_fn_t = f1
            .void_t
            .fn_type(&[f1.ptr_t.into(), f1.i32_t.into()], false);

        let call = f1
            .builder
            .build_indirect_call(
                interp_fn_t,
                func_ptr_param,
                &[f1.arm_state_ptr.into(), v1.into()],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        f1.builder.build_return(None).unwrap();
        let compiled1 = f1.compile().unwrap();
        cache.insert(0, compiled1);

        let mut f2 = comp.new_function(1, &cache);
        f2.load_initial_reg_values(&all_regs).unwrap();
        f2.reg_map.update(Reg::R0, f2.i32_t.const_int(999, false));
        f2.write_state_out().unwrap();

        // Construct the function pointer using raw pointer obtained from function cache
        let func_ptr_param = f2.get_compiled_func_pointer(0).unwrap().unwrap();

        // Perform indirect call through pointer
        let call = f2
            .builder
            .build_indirect_call(f2.fn_t, func_ptr_param, &[f2.arm_state_ptr.into()], "call")
            .unwrap();
        call.set_tail_call(true);
        f2.builder.build_return(None).unwrap();
        let compiled2 = f2.compile().unwrap();
        cache.insert(1, compiled2);

        println!("{:?}", state.regs);
        unsafe {
            cache.get(&1).unwrap().call(&mut state);
        }
        println!("{:?}", state.regs);

        assert_eq!(
            state.regs,
            [
                999, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 986, 256
            ]
        );
    }

    #[test]
    fn test_call_intrinsic() {
        let mut state = ArmState::default();
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context);
        let mut f = comp.new_function(0, &cache);

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        f.load_initial_reg_values(&all_regs).unwrap();

        let bd = f.builder;
        let call = bd
            .build_call(
                f.sadd_with_overflow,
                &[
                    f.i32_t.const_int(0x7fffffff_u64, false).into(),
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
        f.write_state_out().unwrap();
        f.builder.build_return(None).unwrap();
        compile_and_run!(comp, f, state);

        println!("{:?}", state.regs);
        assert_eq!(state.regs[0], 0x800000fe);
        assert_eq!(state.regs[1], 1);
    }
}

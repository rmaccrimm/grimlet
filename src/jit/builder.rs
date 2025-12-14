macro_rules! imm {
    ($self:ident, $i:expr) => {
        $self.i32_t.const_int($i as u64, false)
    };
}

macro_rules! imm64 {
    ($self:ident, $i:expr) => {
        $self.llvm_ctx.i64_type().const_int($i as u64, false)
    };
}

macro_rules! call_intrinsic {
    ($builder:ident, $self:ident . $intrinsic:ident, $($args:expr),+) => {
        $builder
            .build_call(
                $self.$intrinsic,
                &[$($args.into()),+],
                &format!("{}_res", stringify!(intrinsic))
            )?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| anyhow!("failed to get {} return val", stringify!(intrinsic)))?
    };
}

mod alu;
mod branch;
mod flags;
mod load_store;
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
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};

use super::{CompiledFunction, FunctionCache};
use crate::arm::cpu::{ArmMode, ArmState, NUM_REGS, Reg};
use crate::arm::disasm::code_block::CodeBlock;
use crate::arm::disasm::instruction::ArmInstruction;
use crate::jit::builder::reg_map::RegMap;

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
    func_cache: Option<&'a FunctionCache<'ctx>>,

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
    fshr: FunctionValue<'a>,
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
        func_cache: Option<&'a FunctionCache<'ctx>>,
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
            let fshr = get_intrinsic("llvm.fshr", module)?;

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
                fshr,
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
        match self.func_cache {
            None => Ok(None),
            Some(cache) => {
                match cache.get(&key) {
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
            ArmInsn::ARM_INS_ADC => self.arm_adc(instr),
            ArmInsn::ARM_INS_ADD => self.arm_add(instr),
            ArmInsn::ARM_INS_AND => self.arm_and(instr),
            ArmInsn::ARM_INS_ASR => unimpl_instr!(instr, "ASR"),
            ArmInsn::ARM_INS_B => self.arm_b(instr),
            ArmInsn::ARM_INS_BIC => self.arm_bic(instr),
            ArmInsn::ARM_INS_BL => unimpl_instr!(instr, "BL"),
            ArmInsn::ARM_INS_BX => unimpl_instr!(instr, "BX"),
            ArmInsn::ARM_INS_CDP => unimpl_instr!(instr, "CDP"),
            ArmInsn::ARM_INS_CMN => self.arm_cmn(instr),
            ArmInsn::ARM_INS_CMP => self.arm_cmp(instr),
            ArmInsn::ARM_INS_EOR => self.arm_eor(instr),
            ArmInsn::ARM_INS_LDC => unimpl_instr!(instr, "LDC"),
            ArmInsn::ARM_INS_LDM => self.arm_ldmia(instr),
            ArmInsn::ARM_INS_LDMDA => unimpl_instr!(instr, "LDMDA"),
            ArmInsn::ARM_INS_LDMDB => unimpl_instr!(instr, "LDMDB"),
            ArmInsn::ARM_INS_LDMIB => unimpl_instr!(instr, "LDMIB"),
            ArmInsn::ARM_INS_LDR => unimpl_instr!(instr, "LDR"),
            ArmInsn::ARM_INS_LDRB => unimpl_instr!(instr, "LDRB"),
            ArmInsn::ARM_INS_LDRBT => unimpl_instr!(instr, "LDRBT"),
            ArmInsn::ARM_INS_LDRH => unimpl_instr!(instr, "LDRH"),
            ArmInsn::ARM_INS_LDRHT => unimpl_instr!(instr, "LDRHT"),
            ArmInsn::ARM_INS_LDRSB => unimpl_instr!(instr, "LDRSB"),
            ArmInsn::ARM_INS_LDRSBT => unimpl_instr!(instr, "LDRSBT"),
            ArmInsn::ARM_INS_LDRSH => unimpl_instr!(instr, "LDRSH"),
            ArmInsn::ARM_INS_LDRSHT => unimpl_instr!(instr, "LDRSHT"),
            ArmInsn::ARM_INS_LDRT => unimpl_instr!(instr, "LDRT"),
            ArmInsn::ARM_INS_LSL => unimpl_instr!(instr, "LSL"),
            ArmInsn::ARM_INS_LSR => unimpl_instr!(instr, "LSR"),
            ArmInsn::ARM_INS_MCR => unimpl_instr!(instr, "MCR"),
            ArmInsn::ARM_INS_MLA => self.arm_mla(instr),
            ArmInsn::ARM_INS_MOV => self.arm_mov(instr),
            ArmInsn::ARM_INS_MRC => unimpl_instr!(instr, "MRC"),
            ArmInsn::ARM_INS_MRS => self.arm_msr(instr),
            ArmInsn::ARM_INS_MSR => self.arm_mrs(instr),
            ArmInsn::ARM_INS_MUL => self.arm_mul(instr),
            ArmInsn::ARM_INS_MVN => self.arm_mvn(instr),
            ArmInsn::ARM_INS_ORR => self.arm_orr(instr),
            // In ARM mode, ldmia rd! sometimes gets ddecoded with POP mnemonic
            ArmInsn::ARM_INS_POP => self.arm_ldmia(instr),
            // In ARM mode, stmdb rd! sometimes gets decoded with PUSH mnemonic
            ArmInsn::ARM_INS_PUSH => self.arm_stmdb(instr),
            ArmInsn::ARM_INS_ROR => unimpl_instr!(instr, "ROR"),
            ArmInsn::ARM_INS_RRX => unimpl_instr!(instr, "RRX"),
            ArmInsn::ARM_INS_RSB => self.arm_rsb(instr),
            ArmInsn::ARM_INS_RSC => self.arm_rsc(instr),
            ArmInsn::ARM_INS_SBC => self.arm_sbc(instr),
            ArmInsn::ARM_INS_SMLAL => self.arm_smlal(instr),
            ArmInsn::ARM_INS_SMULL => self.arm_smull(instr),
            ArmInsn::ARM_INS_STC => unimpl_instr!(instr, "STC"),
            ArmInsn::ARM_INS_STM => self.arm_stmia(instr),
            ArmInsn::ARM_INS_STMIB => unimpl_instr!(instr, "STMIB"),
            ArmInsn::ARM_INS_STMDA => unimpl_instr!(instr, "STMDA"),
            // Possibly decoded as a PUSH? When writeback enabled
            ArmInsn::ARM_INS_STMDB => unimpl_instr!(instr, "STMDB"),

            ArmInsn::ARM_INS_STR => unimpl_instr!(instr, "STR"),
            ArmInsn::ARM_INS_STRB => unimpl_instr!(instr, "STRB"),
            ArmInsn::ARM_INS_STRBT => unimpl_instr!(instr, "STRBT"),
            ArmInsn::ARM_INS_STRH => unimpl_instr!(instr, "STRH"),
            ArmInsn::ARM_INS_STRT => unimpl_instr!(instr, "STRT"),
            ArmInsn::ARM_INS_SUB => self.arm_sub(instr),
            // SWI?
            ArmInsn::ARM_INS_SWP => unimpl_instr!(instr, "SWP"),
            ArmInsn::ARM_INS_SWPB => unimpl_instr!(instr, "SWPB"),
            ArmInsn::ARM_INS_TEQ => self.arm_teq(instr),
            ArmInsn::ARM_INS_TST => self.arm_tst(instr),
            ArmInsn::ARM_INS_UMLAL => self.arm_umlal(instr),
            ArmInsn::ARM_INS_UMULL => self.arm_umull(instr),
            _ => panic!("unsupported instruction {:?}", instr.opcode),
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
        let mut f = comp.new_function(0, None);

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        f.load_initial_reg_values(&all_regs).unwrap();

        let add_res = f
            .builder
            .build_int_add(f.reg_map.get(Reg::PC), f.reg_map.get(Reg::R9), "add_res")
            .unwrap();

        let interp_fn_type = f.void_t.fn_type(&[f.ptr_t.into(), f.i32_t.into()], false);

        let interp_fn_ptr = f
            .get_external_func_pointer(ArmState::jump_to as *const fn(&mut ArmState, u32) as usize)
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
            state.regs[i] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context);
        let mut cache = HashMap::new();

        let all_regs: HashSet<Reg> = REG_ITEMS.into_iter().collect();
        let mut f1 = comp.new_function(0, Some(&cache));
        f1.load_initial_reg_values(&all_regs).unwrap();

        let r0 = f1.reg_map.get(Reg::R0);
        let r2 = f1.reg_map.get(Reg::R2);
        let r3 = f1.reg_map.get(Reg::R3);
        let v0 = f1.builder.build_int_add(r3, r2, "v0").unwrap();
        let v1 = f1.builder.build_int_sub(r0, v0, "v1").unwrap();

        // Perform context switch out before jumping to ArmState code
        f1.write_state_out().unwrap();

        let func_ptr_param = f1
            .get_external_func_pointer(ArmState::jump_to as fn(&mut ArmState, u32) as usize)
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

        let mut f2 = comp.new_function(1, Some(&cache));
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
        let mut comp = Compiler::new(&context);
        let mut f = comp.new_function(0, None);

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

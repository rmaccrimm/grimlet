#[macro_export]
/// Convenience macro for testing single functions
macro_rules! compile_and_run {
    ($compiler:ident, $func:ident, $state:ident) => {
        unsafe {
            let fptr = $func.compile().unwrap().as_raw();
            $compiler
                .compile_entry_point()
                .unwrap()
                .call(&mut $state, fptr);
        }
    };
}

mod alu;
mod branch;
mod flags;
mod instr;

use crate::arm::cpu::{ArmState, NUM_REGS, Reg};
use crate::arm::disasm::ArmDisasm;
use anyhow::{Result, anyhow};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::types::{ArrayType, FunctionType, IntType, PointerType, StructType, VoidType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::collections::HashMap;
use std::fs;

type JumpTarget = unsafe extern "C" fn(*mut ArmState, *const i32);

pub type CompiledFunction<'a> = JitFunction<'a, JumpTarget>;

pub type EntryPoint<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState, JumpTarget)>;

pub type FunctionCache<'ctx> = HashMap<u64, CompiledFunction<'ctx>>;

pub struct RegMap<'a> {
    llvm_values: Vec<IntValue<'a>>,
}

impl<'a> RegMap<'a> {
    pub fn new(llvm_values: Vec<IntValue<'a>>) -> Self {
        if llvm_values.len() != NUM_REGS {
            panic!("Expected exactly {} values", llvm_values.len());
        }
        Self { llvm_values }
    }

    pub fn update(&mut self, reg: Reg, value: IntValue<'a>) {
        self.llvm_values[reg as usize] = value;
    }

    pub fn get(&self, reg: Reg) -> IntValue<'a> {
        self.llvm_values[reg as usize]
    }

    pub fn r0(&self) -> IntValue<'a> {
        self.get(Reg::R0)
    }

    pub fn r1(&self) -> IntValue<'a> {
        self.get(Reg::R1)
    }

    pub fn r2(&self) -> IntValue<'a> {
        self.get(Reg::R2)
    }

    pub fn r3(&self) -> IntValue<'a> {
        self.get(Reg::R3)
    }

    pub fn r4(&self) -> IntValue<'a> {
        self.get(Reg::R4)
    }

    pub fn r5(&self) -> IntValue<'a> {
        self.get(Reg::R5)
    }

    pub fn r6(&self) -> IntValue<'a> {
        self.get(Reg::R6)
    }

    pub fn r7(&self) -> IntValue<'a> {
        self.get(Reg::R7)
    }

    pub fn r8(&self) -> IntValue<'a> {
        self.get(Reg::R8)
    }

    pub fn r9(&self) -> IntValue<'a> {
        self.get(Reg::R9)
    }

    pub fn r10(&self) -> IntValue<'a> {
        self.get(Reg::R10)
    }

    pub fn r11(&self) -> IntValue<'a> {
        self.get(Reg::R11)
    }

    pub fn r12(&self) -> IntValue<'a> {
        self.get(Reg::R12)
    }

    pub fn sp(&self) -> IntValue<'a> {
        self.get(Reg::SP)
    }

    pub fn lr(&self) -> IntValue<'a> {
        self.get(Reg::LR)
    }

    pub fn pc(&self) -> IntValue<'a> {
        self.get(Reg::PC)
    }

    pub fn cpsr(&self) -> IntValue<'a> {
        self.get(Reg::CPSR)
    }
}

/// Helper that converts the LLVMString error message into an anyhow error
fn get_ptr_param<'a>(func: &FunctionValue<'a>, i: usize) -> Result<PointerValue<'a>> {
    Ok(func
        .get_nth_param(i as u32)
        .ok_or(anyhow!(
            "{} signature has no parameter {}",
            func.get_name().to_str()?,
            i
        ))?
        .into_pointer_value())
}

fn func_name(addr: u64) -> String {
    format!("fn_{:#010x}", addr)
}

/// Manages LLVM compilation state and constructs new LlvmFunctions
pub struct Compiler<'ctx> {
    llvm_ctx: &'ctx Context,
    builder: Builder<'ctx>,
    modules: Vec<Module<'ctx>>,
    engines: Vec<ExecutionEngine<'ctx>>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let modules = Vec::new();
        let engines = Vec::new();
        let builder = context.create_builder();

        Ok(Self {
            llvm_ctx: context,
            modules,
            engines,
            builder,
        })
    }

    pub fn new_function<'a>(
        &'a mut self,
        addr: u64,
        func_cache: &'a FunctionCache<'ctx>,
    ) -> Result<LlvmFunction<'ctx, 'a>>
    where
        'ctx: 'a,
    {
        let i = self.create_module(&format!("m_{}", addr))?;
        let lf = LlvmFunction::new(
            addr,
            self.llvm_ctx,
            &self.builder,
            &self.modules[i],
            &self.engines[i],
            func_cache,
        )?;
        Ok(lf)
    }

    pub fn dump(&self) -> Result<()> {
        if fs::exists("llvm")? && fs::metadata("llvm")?.is_dir() {
            fs::remove_dir_all("llvm")?
        }
        fs::create_dir("llvm")?;
        for (i, m) in self.modules.iter().enumerate() {
            m.print_to_file(format!("llvm/mod_{}.ll", i)).unwrap();
        }
        Ok(())
    }

    fn create_module(&mut self, name: &str) -> Result<usize> {
        self.modules.push(self.llvm_ctx.create_module(name));
        let module = self.modules.last().unwrap();
        self.engines.push(
            module
                .create_jit_execution_engine(OptimizationLevel::None)
                .map_err(|s| {
                    let er = s
                        .to_str()
                        .map(String::from)
                        .unwrap_or("Failed to create execution engine".into());
                    anyhow!(er)
                })?,
        );
        Ok(self.modules.len() - 1)
    }

    // Performs context switch from guest machine to LLVM code and jumps to provided function
    pub fn compile_entry_point(&mut self) -> Result<EntryPoint<'ctx>> {
        let i = self.create_module("m_entrypoint")?;
        let ctx = self.llvm_ctx;
        let bd = &self.builder;
        let module = &self.modules[i];
        let ee = &self.engines[i];

        let i32_t = ctx.i32_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();
        let regs_t = i32_t.array_type(NUM_REGS as u32);
        let arm_state_t = ArmState::get_llvm_type(ctx);
        let fn_t = void_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        let entry_type = self
            .llvm_ctx
            .void_type()
            .fn_type(&[ptr_t.into(), ptr_t.into()], false);
        let f = module.add_function("entry_point", entry_type, None);
        let basic_block = self.llvm_ctx.append_basic_block(f, "start");
        bd.position_at_end(basic_block);

        let arm_state_ptr = get_ptr_param(&f, 0)?;
        let fn_ptr_arg = get_ptr_param(&f, 1)?;

        let regs_ptr = bd.build_alloca(regs_t, "regs_ptr")?;

        // Perform context switch in, i.e. copy guest machine state into an array
        let zero = i32_t.const_zero();
        let one = i32_t.const_int(1, false);
        for r in 0..NUM_REGS {
            let reg_ind = i32_t.const_int(r as u64, false);
            let gep_inds = [zero, one, reg_ind];
            let name = format!("arm_state_r{}_ptr", r);
            // Pointer to the register in the guest machine (ArmState object)
            let arm_state_elem_ptr =
                unsafe { bd.build_gep(arm_state_t, arm_state_ptr, &gep_inds, &name)? };
            let value = bd
                .build_load(i32_t, arm_state_elem_ptr, &format!("r{}", r))?
                .into_int_value();

            let gep_inds = [zero, reg_ind];
            let name = format!("reg_arr_r{}_ptr", r);
            // Pointer to the local register (i32 array)
            let reg_arr_elem_ptr = unsafe { bd.build_gep(regs_t, regs_ptr, &gep_inds, &name)? };
            bd.build_store(reg_arr_elem_ptr, value)?;
        }

        // Call the actual processing func. Note this is not a tail call as the stack becomes
        // corrupted (presumably because the regs_ptr array gets freed)
        bd.build_indirect_call(
            fn_t,
            fn_ptr_arg,
            &[arm_state_ptr.into(), regs_ptr.into()],
            "call",
        )?;
        bd.build_return(None)?;
        assert!(f.verify(true));

        let entry_point = unsafe { ee.get_function("entry_point")? };
        Ok(entry_point)
    }
}

/// Builder for creating & compiling LLVM functions
pub struct LlvmFunction<'ctx, 'a>
where
    'ctx: 'a,
{
    addr: u64,
    name: String,
    func: FunctionValue<'a>,
    // Latest value for each register
    reg_map: RegMap<'a>,
    // References to parent compiler LLVM state
    llvm_ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    execution_engine: &'a ExecutionEngine<'ctx>,
    // Read only ref to already-compiled functions
    func_cache: &'a FunctionCache<'ctx>,
    // Last instruction executed. Used to lazily evaluate status flags
    last_result: IntValue<'a>,
    last_instr: ArmDisasm,
    // Function arguments
    arm_state_ptr: PointerValue<'a>,
    reg_array_ptr: PointerValue<'a>,
    // Frequently used LLVM types
    arm_state_t: StructType<'a>,
    reg_array_t: ArrayType<'a>,
    fn_t: FunctionType<'a>,
    i32_t: IntType<'a>,
    ptr_t: PointerType<'a>,
    void_t: VoidType<'a>,
    // Return type of add/sub with overflow intrinsics
    intrinsic_t: StructType<'a>,
    // Overflow arithmetic intrinsics
    sadd_with_overflow: FunctionValue<'a>,
    ssub_with_overflow: FunctionValue<'a>,
    uadd_with_overflow: FunctionValue<'a>,
    usub_with_overflow: FunctionValue<'a>,
}

impl<'ctx, 'a> LlvmFunction<'ctx, 'a> {
    fn new(
        addr: u64,
        llvm_ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        execution_engine: &'a ExecutionEngine<'ctx>,
        func_cache: &'a FunctionCache<'ctx>,
    ) -> Result<Self> {
        let name = func_name(addr);
        let ctx = llvm_ctx;
        let i32_t = ctx.i32_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();
        let regs_t = i32_t.array_type(NUM_REGS as u32);
        let arm_state_t = ArmState::get_llvm_type(ctx);
        let fn_t = void_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        let bd = builder;
        let func = module.add_function(&name, fn_t, None);
        let basic_block = ctx.append_basic_block(func, "start");
        bd.position_at_end(basic_block);

        let arm_state_ptr = get_ptr_param(&func, 0)?;
        let reg_array_ptr = get_ptr_param(&func, 1)?;

        let mut reg_map = Vec::new();
        for i in 0..NUM_REGS {
            let name = format!("r{}_elem_ptr", i);
            let gep_inds = [i32_t.const_zero(), i32_t.const_int(i as u64, false)];
            let ptr = unsafe {
                bd.build_gep(regs_t, reg_array_ptr, &gep_inds, &name)
                    .unwrap()
            };
            let name = format!("r{}", i);
            let v = bd.build_load(i32_t, ptr, &name)?.into_int_value();
            reg_map.push(v);
        }

        // Declare intrinsics
        let intrinsic_t = ctx.struct_type(&[i32_t.into(), ctx.bool_type().into()], false);

        let sadd_intrinsic = Intrinsic::find("llvm.sadd.with.overflow").unwrap();
        let sadd_with_overflow = sadd_intrinsic
            .get_declaration(module, &[i32_t.into()])
            .unwrap();
        let ssub_intrinsic = Intrinsic::find("llvm.ssub.with.overflow").unwrap();
        let ssub_with_overflow = ssub_intrinsic
            .get_declaration(module, &[i32_t.into()])
            .unwrap();
        let uadd_intrinsic = Intrinsic::find("llvm.uadd.with.overflow").unwrap();
        let uadd_with_overflow = uadd_intrinsic
            .get_declaration(module, &[i32_t.into()])
            .unwrap();
        let usub_intrinsic = Intrinsic::find("llvm.usub.with.overflow").unwrap();
        let usub_with_overflow = usub_intrinsic
            .get_declaration(module, &[i32_t.into()])
            .unwrap();

        Ok(LlvmFunction {
            addr,
            name,
            reg_map: RegMap::new(reg_map),
            func,
            arm_state_ptr,
            reg_array_ptr,
            llvm_ctx: ctx,
            builder,
            module,
            execution_engine,
            func_cache,
            arm_state_t,
            fn_t,
            i32_t,
            ptr_t,
            reg_array_t: regs_t,
            void_t,
            last_result: i32_t.const_zero(),
            last_instr: ArmDisasm::default(),
            intrinsic_t,
            sadd_with_overflow,
            ssub_with_overflow,
            uadd_with_overflow,
            usub_with_overflow,
        })
    }

    pub fn compile(self) -> Result<CompiledFunction<'ctx>> {
        self.builder.build_return(None)?;
        if self.func.verify(true) {
            let jit_func = unsafe { self.execution_engine.get_function(&self.name)? };
            Ok(jit_func)
        } else {
            Err(anyhow!("Compilation failed"))
        }
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

    fn get_compiled_func_pointer(&self, key: u64) -> Result<Option<PointerValue<'a>>> {
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
        let zero = self.i32_t.const_zero();
        let one = self.i32_t.const_int(1, false);
        for (i, rval) in self.reg_map.llvm_values.iter().enumerate() {
            let reg_ind = self.i32_t.const_int(i as u64, false);
            let gep_inds = [zero, one, reg_ind];
            let name = format!("arm_state_r{}_ptr", i);
            // Pointer to the register in the guest machine (ArmState object)
            let arm_state_elem_ptr =
                unsafe { bd.build_gep(self.arm_state_t, self.arm_state_ptr, &gep_inds, &name)? };

            bd.build_store(arm_state_elem_ptr, *rval)?;
        }
        Ok(())
    }

    // For jumping without context switching. Updates the reg array allocated in entry point with
    // latest values in reg_map
    fn update_reg_array(&self) -> Result<()> {
        let bd = &self.builder;
        let zero = self.i32_t.const_zero();
        for (i, rval) in self.reg_map.llvm_values.iter().enumerate() {
            let reg_ind = self.i32_t.const_int(i as u64, false);
            let gep_inds = [zero, reg_ind];
            let name = format!("reg_arr_r{}_ptr", i);
            // Pointer to the local register (i32 array)
            let reg_arr_elem_ptr =
                unsafe { bd.build_gep(self.reg_array_t, self.reg_array_ptr, &gep_inds, &name)? };
            bd.build_store(reg_arr_elem_ptr, *rval)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_to_external() {
        // End result is:
        // pc <- r15 + r9
        let mut state = ArmState::default();
        for i in 0..NUM_REGS {
            state.regs[i] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context).unwrap();
        let func_cache = HashMap::new();
        let f = comp.new_function(0, &func_cache).unwrap();

        let add_res = f
            .builder
            .build_int_add(f.reg_map.get(Reg::PC), f.reg_map.get(Reg::R9), "add_res")
            .unwrap();

        let interp_fn_type = f.void_t.fn_type(&[f.ptr_t.into(), f.i32_t.into()], false);

        let interp_fn_ptr = f
            .get_external_func_pointer(ArmState::jump_to as usize)
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
        let mut comp = Compiler::new(&context).unwrap();
        let entry_point = comp.compile_entry_point().unwrap();
        let mut cache = HashMap::new();

        let f1 = comp.new_function(0, &cache).unwrap();
        let r0 = f1.reg_map.get(Reg::R0);
        let r2 = f1.reg_map.get(Reg::R2);
        let r3 = f1.reg_map.get(Reg::R3);
        let v0 = f1.builder.build_int_add(r3, r2, "v0").unwrap();
        let v1 = f1
            .builder
            .build_int_sub(
                // r0,
                r0, v0, "v1",
            )
            .unwrap();

        // Perform context switch out before jumping to ArmState code
        f1.write_state_out().unwrap();

        let func_ptr_param = f1
            .get_external_func_pointer(ArmState::jump_to as usize)
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
                // &[
                //     f1.arm_state_ptr.into(),
                //     f1.i32_t.const_int(843, false).into(),
                // ],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        let compiled1 = f1.compile().unwrap();
        cache.insert(0, compiled1);

        let mut f2 = comp.new_function(1, &cache).unwrap();
        f2.reg_map.update(Reg::R0, f2.i32_t.const_int(999, false));
        f2.update_reg_array().unwrap();

        // Construct the function pointer using raw pointer obtained from function cache
        let func_ptr_param = f2.get_compiled_func_pointer(0).unwrap().unwrap();

        // Perform indirect call through pointer
        let call = f2
            .builder
            .build_indirect_call(
                f2.fn_t,
                func_ptr_param,
                &[f2.arm_state_ptr.into(), f2.reg_array_ptr.into()],
                "call",
            )
            .unwrap();
        call.set_tail_call(true);
        let compiled2 = f2.compile().unwrap();
        cache.insert(1, compiled2);

        // comp.dump();
        println!("{:?}", state.regs);
        unsafe {
            entry_point.call(&mut state, cache.get(&1).unwrap().as_raw());
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
    fn test_call_intrinsic() -> Result<()> {
        let mut state = ArmState::default();
        let context = Context::create();
        let cache = HashMap::new();
        let mut comp = Compiler::new(&context)?;
        let mut f = comp.new_function(0, &cache)?;
        let bd = f.builder;
        let call = bd.build_call(
            f.sadd_with_overflow,
            &[
                f.i32_t.const_int(0x7fffffff_u64, false).into(),
                f.i32_t.const_int(0xff, false).into(),
            ],
            "res",
        )?;
        let res = call
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_struct_value();

        let val = bd.build_extract_value(res, 0, "val")?.into_int_value();
        let overflowed = bd
            .build_extract_value(res, 1, "overflowed")?
            .into_int_value();

        f.reg_map.update(Reg::R0, val);
        f.reg_map.update(Reg::R1, overflowed);
        f.write_state_out()?;
        compile_and_run!(comp, f, state);

        comp.dump()?;
        println!("{:?}", state.regs);
        assert_eq!(state.regs[0], 0x800000fe);
        assert_eq!(state.regs[1], 1);
        Ok(())
    }
}

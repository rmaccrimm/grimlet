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
mod instr;
mod ldstr;
mod tests;

use crate::arm::cpu::{ArmState, NUM_REGS, Reg};
use anyhow::{Result, anyhow};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::{ArrayType, FunctionType, IntType, PointerType, StructType, VoidType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::collections::HashMap;

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

/// Core struct responsible for translating ARM instructions to LLVM and managing the compiled
/// executable code.
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

    pub fn dump(&self) {
        for (i, m) in self.modules.iter().enumerate() {
            m.print_to_file(&format!("mod_{}.ll", i)).unwrap();
        }
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

/// Builder struct for creating & compiling LLVM functions
pub struct LlvmFunction<'ctx, 'a>
where
    'ctx: 'a,
{
    llvm_ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    execution_engine: &'a ExecutionEngine<'ctx>,
    func_cache: &'a FunctionCache<'ctx>,
    addr: u64,
    name: String,
    func: FunctionValue<'a>,
    reg_map: RegMap<'a>,
    arm_state_ptr: PointerValue<'a>,
    reg_array_ptr: PointerValue<'a>,
    arm_state_t: StructType<'a>,
    reg_array_t: ArrayType<'a>,
    fn_t: FunctionType<'a>,
    i32_t: IntType<'a>,
    ptr_t: PointerType<'a>,
    void_t: VoidType<'a>,
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

        Ok(LlvmFunction {
            addr: addr as u64,
            name: name,
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

    fn get_external_func_pointer(&self, func_addr: u64) -> Result<PointerValue<'a>> {
        let ee = &self.execution_engine;
        let func_ptr = self.builder.build_int_to_ptr(
            self.llvm_ctx
                .ptr_sized_int_type(ee.get_target_data(), None)
                .const_int(func_addr as u64, false),
            self.ptr_t,
            &format!("extern_ptr"),
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
                            .const_int(f.as_raw() as u64, false),
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

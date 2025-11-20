mod alu;
mod branch;
mod compile;
mod ldstr;

use crate::arm::cpu::ArmState;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::ops::Add;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{self, ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::types::{ArrayType, FunctionType, IntType, PointerType, StructType, VoidType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, OptimizationLevel};

type JumpTarget = unsafe extern "C" fn(*mut ArmState, *const i32);

type CompiledFunc<'a> = JitFunction<'a, JumpTarget>;

type EntryPoint<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState, JumpTarget)>;

type Trampoline<'a> = JitFunction<'a, unsafe extern "C" fn(JumpTarget, *mut ArmState, *const i32)>;

pub struct LlvmFunction<'a> {
    addr: u64,
    name: String,
    reg_map: Vec<IntValue<'a>>,
    module_ind: usize,
    func: FunctionValue<'a>,
    state_ptr: PointerValue<'a>,
}

/// Helper that converts the LLVMString into an anyhow error
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

pub struct Compiler<'ctx> {
    pub func_cache: HashMap<u32, CompiledFunc<'ctx>>,
    llvm_ctx: &'ctx Context,
    builder: Builder<'ctx>,
    modules: Vec<Module<'ctx>>,
    engines: Vec<ExecutionEngine<'ctx>>,
    arm_state_t: StructType<'ctx>,
    fn_t: FunctionType<'ctx>,
    i32_t: IntType<'ctx>,
    ptr_t: PointerType<'ctx>,
    regs_t: ArrayType<'ctx>,
    void_t: VoidType<'ctx>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let modules = Vec::new();
        let engines = Vec::new();
        let builder = context.create_builder();
        let i32_t = context.i32_type();
        let ptr_t = context.ptr_type(AddressSpace::default());
        let void_t = context.void_type();
        let regs_t = i32_t.array_type(17);
        let arm_state_t = context.struct_type(&[regs_t.into(), ptr_t.into()], false);
        let fn_t = void_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        Self {
            func_cache: HashMap::new(),
            llvm_ctx: context,
            modules,
            engines,
            builder,
            arm_state_t,
            i32_t,
            fn_t,
            ptr_t,
            regs_t,
            void_t,
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

    pub fn build_entry_point(&mut self) -> Result<EntryPoint<'ctx>> {
        let i = self.create_module("m_entrypoint")?;
        let module = &self.modules[i];
        let ee = &self.engines[i];

        // First global function - trampoline()
        // Simply performs an indirect call to the provided function pointer, passing array arg
        // forward.
        let trampoline_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.ptr_t.into(), self.ptr_t.into()],
            false,
        );

        let trampoline = module.add_function("trampoline", trampoline_t, None);
        let basic_block = self.llvm_ctx.append_basic_block(trampoline, "start");
        self.builder.position_at_end(basic_block);

        let fn_ptr_arg = get_ptr_param(&trampoline, 0)?;
        let state_ptr_arg = get_ptr_param(&trampoline, 1)?;
        let array_ptr_arg = get_ptr_param(&trampoline, 2)?;

        self.builder.build_indirect_call(
            self.fn_t,
            fn_ptr_arg,
            &[state_ptr_arg.into(), array_ptr_arg.into()],
            "call",
        )?;
        self.builder.build_return(None)?;

        // Second global function - entry_point
        // Performs context switch from guest machine to LLVM code and jumps to provided function
        let entry_type = self
            .llvm_ctx
            .void_type()
            .fn_type(&[self.ptr_t.into(), self.ptr_t.into()], false);
        let f = module.add_function("entry_point", entry_type, None);
        let basic_block = self.llvm_ctx.append_basic_block(f, "start");
        self.builder.position_at_end(basic_block);

        let state_ptr_arg = get_ptr_param(&f, 0)?;
        let fn_ptr_arg = get_ptr_param(&f, 1)?;

        let call_arg = self.builder.build_alloca(self.regs_t, "reg_values_ptr")?;

        for r in 0..17usize {
            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_zero(),
                self.i32_t.const_int(r as u64, false),
            ];
            let name = format!("r{}_guest_ptr", r);
            let guest_ptr = unsafe {
                self.builder
                    .build_gep(self.arm_state_t, state_ptr_arg, &gep_inds, &name)?
            };
            let value = self
                .builder
                .build_load(self.i32_t, guest_ptr, &format!("r{}", r))?
                .into_int_value();

            let gep_inds = [
                self.i32_t.const_zero(),
                self.i32_t.const_int(r as u64, false),
            ];
            let name = format!("r{}_ptr", r);
            let arr_ptr = unsafe {
                self.builder
                    .build_gep(self.regs_t, call_arg, &gep_inds, &name)?
            };
            self.builder.build_store(arr_ptr, value)?;
        }

        self.builder.build_call(
            trampoline,
            &[fn_ptr_arg.into(), state_ptr_arg.into(), call_arg.into()],
            "call",
        )?;
        self.builder.build_return(None)?;
        assert!(f.verify(true));

        let entry_point = unsafe { ee.get_function("entry_point")? };
        Ok(entry_point)
    }

    pub fn new_function<'a>(&mut self, addr: u32) -> Result<LlvmFunction<'a>>
    where
        'ctx: 'a,
    {
        let func_name = format!("fn_{:#010x}", addr);
        let i = self.create_module(&format!("m_{}", &func_name))?;

        let module = &self.modules[i];
        let ee = &self.engines[i];

        let interp_fn_type = self
            .void_t
            .fn_type(&[self.ptr_t.into(), self.i32_t.into()], false);
        let interp_fn =
            module.add_function("ArmState::jump_to", interp_fn_type, Some(Linkage::External));

        ee.add_global_mapping(&interp_fn, ArmState::jump_to as usize);

        let trampoline_fn_t = self.void_t.fn_type(
            &[self.ptr_t.into(), self.ptr_t.into(), self.ptr_t.into()],
            false,
        );
        let trampoline_fn =
            module.add_function("trampoline", trampoline_fn_t, Some(Linkage::External));

        let globals_ee = &self.engines[0];
        let trampoline_compiled: Trampoline<'ctx> =
            unsafe { globals_ee.get_function("trampoline")? };

        unsafe {
            ee.add_global_mapping(&trampoline_fn, trampoline_compiled.as_raw() as usize);
        }

        let func = module.add_function(&func_name, self.fn_t, None);
        let basic_block = self.llvm_ctx.append_basic_block(func, "start");
        self.builder.position_at_end(basic_block);

        let state_ptr = get_ptr_param(&func, 0)?;
        let base_ptr = get_ptr_param(&func, 1)?;

        let reg_map = (0..17)
            .map(|i| {
                let name = format!("r{}_elem_ptr", i);
                let gep_inds = [self.i32_t.const_zero(), self.i32_t.const_int(i, false)];
                let ptr = unsafe {
                    self.builder
                        .build_gep(self.regs_t, base_ptr, &gep_inds, &name)
                        .unwrap()
                };
                let name = format!("r{}", i);
                self.builder
                    .build_load(self.i32_t, ptr, &name)
                    .unwrap()
                    .into_int_value()
            })
            .collect();

        Ok(LlvmFunction {
            addr: addr as u64,
            name: func_name,
            reg_map,
            module_ind: self.modules.len() - 1,
            func,
            state_ptr,
        })
    }

    pub fn append_insn(&mut self, func: &LlvmFunction, addr: u32) {
        todo!();
    }

    pub fn compile(&mut self, func: LlvmFunction) -> Result<&CompiledFunc<'ctx>> {
        self.builder.build_return(None).unwrap();
        if func.func.verify(true) {
            let jit_func = unsafe {
                self.engines[func.module_ind]
                    .get_function(&func.name)
                    .unwrap()
            };
            let k = func.addr as u32;
            self.func_cache.insert(k, jit_func);
            Ok(&self.func_cache.get(&k).unwrap())
        } else {
            Err(anyhow!("Compilation failed"))
        }
    }

    pub fn dump(&self) {
        for (i, m) in self.modules.iter().enumerate() {
            m.print_to_file(&format!("mod_{}.ll", i)).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_to_external() {
        let mut state = ArmState::new();
        let context = Context::create();
        let mut comp = Compiler::new(&context);

        let entry = comp.build_entry_point().unwrap();

        let func = comp.new_function(0).unwrap();
        // Run a single instruction (call `jump_to` func)
        let module = &comp.modules[func.module_ind];
        let interp_fn = module.get_function("ArmState::jump_to").unwrap();

        comp.builder
            .build_call(
                interp_fn,
                &[
                    func.state_ptr.into(),
                    comp.i32_t.const_int(99, false).into(),
                ],
                "fn_result",
            )
            .unwrap();

        let compiled = comp.compile(func).unwrap();

        // comp.dump();
        println!("{:?}", state.regs);
        unsafe {
            entry.call(&mut state, compiled.as_raw());
        }
        println!("{:?}", state.regs);
        assert_eq!(state.pc(), 99);
    }

    #[test]
    fn test_cross_module_calls() {
        let mut state = ArmState::new();
        let context = Context::create();
        let mut comp = Compiler::new(&context);

        let entry = comp.build_entry_point().unwrap();

        // 1st Func - Run a single instruction (call `jump_to` func)
        let func = comp.new_function(0).unwrap();
        let module = &comp.modules[func.module_ind];
        let interp_fn = module.get_function("ArmState::jump_to").unwrap();
        comp.builder
            .build_call(
                interp_fn,
                &[
                    func.state_ptr.into(),
                    comp.i32_t.const_int(99, false).into(),
                ],
                "fn_result",
            )
            .unwrap();
        comp.compile(func).unwrap();

        // 2nd Func - Just calls 1st
        let func = comp.new_function(0).unwrap();
        let compiled_1 = comp.func_cache.get(&0).unwrap();
        let module = &comp.modules[func.module_ind];
        let ee = &comp.engines[func.module_ind];

        let state_param = get_ptr_param(&func.func, 0).unwrap();
        let regs_param = get_ptr_param(&func.func, 1).unwrap();

        let trampoline = module.get_function("trampoline").unwrap();

        unsafe {
            let func_ptr_param = comp
                .builder
                .build_int_to_ptr(
                    context
                        .ptr_sized_int_type(ee.get_target_data(), None)
                        .const_int(compiled_1.as_raw() as u64, false),
                    comp.ptr_t,
                    "raw_fn_pointer",
                )
                .unwrap();
            comp.builder
                .build_call(
                    trampoline,
                    &[func_ptr_param.into(), state_param.into(), regs_param.into()],
                    "call",
                )
                .unwrap();
        }
        let compiled_2 = comp.compile(func).unwrap();

        // comp.dump();
        println!("{:?}", state.regs);
        unsafe {
            entry.call(&mut state, compiled_2.as_raw());
        }
        println!("{:?}", state.regs);
    }
}

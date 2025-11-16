use std::collections::HashMap;

use super::state::GuestState;
use anyhow::{Result, anyhow};
use inkwell::builder::Builder;
use inkwell::context::{Context, ContextRef};
use inkwell::execution_engine::{self, ExecutionEngine, JitFunction};
use inkwell::llvm_sys::LLVMValue;
use inkwell::llvm_sys::prelude::LLVMValueRef;
use inkwell::module::Module;
use inkwell::types::{FunctionType, IntType, PointerType, StructType};
use inkwell::values::{
    AsValueRef, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{AddressSpace, OptimizationLevel};
use std::sync::Arc;

pub type CompiledBlock = unsafe extern "C" fn(*mut GuestState);

struct GuestRegRef<'a> {
    ptr: PointerValue<'a>,
    value: IntValue<'a>,
}

pub struct LlvmComponents<'ctx> {
    pub function: Option<JitFunction<'ctx, CompiledBlock>>,
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    i32_type: IntType<'ctx>,
    ptr_type: PointerType<'ctx>,
    fn_type: FunctionType<'ctx>,
    state_type: StructType<'ctx>,
}

impl<'ctx> LlvmComponents<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("main");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        let builder = context.create_builder();
        let i32_type = context.i32_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context.void_type().fn_type(&[ptr_type.into()], false);

        let field_types = [i32_type.array_type(17).into(), ptr_type.into()];
        let state_type = context.struct_type(&field_types, false);

        Self {
            function: None,
            context,
            module,
            builder,
            execution_engine,
            i32_type,
            ptr_type,
            fn_type,
            state_type,
        }
    }
}

pub struct Compiler {
    pub func_count: u32,
}

impl Compiler {
    pub fn new() -> Self {
        Self { func_count: 0 }
    }

    /*
       Helpers for accessing frequently used LLVM types (_ll)
    */

    /// Loads the guest machine register into an LLVM register. Both the pointer into the guest
    /// state and current value are maintained so it can be transfered back at the end
    // fn load_registers<'a>(&self, state_ptr: &'a PointerValue, regs: &'a mut Vec<GuestRegRef<'a>>)
    // where
    //     'ctx: 'a,
    // {
    //     todo!();
    // }

    pub fn compile_test<'ctx>(&mut self, ll: &mut LlvmComponents<'ctx>) {
        // ) -> Option<JitFunction<'a, CompiledBlock>> {
        let name = format!("block_{}", self.func_count);
        self.func_count += 1;
        let function = ll.module.add_function(&name, ll.fn_type, None);
        let basic_block = ll.context.append_basic_block(function, "start");
        ll.builder.position_at_end(basic_block);

        let state_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let mut regs = Vec::new();
        for r in 0..17usize {
            let gep_inds = [
                ll.i32_type.const_zero(),
                ll.i32_type.const_zero(),
                ll.i32_type.const_int(r as u64, false),
            ];
            let ptr = unsafe {
                ll.builder
                    .build_gep(ll.state_type, state_ptr, &gep_inds, &format!("r{}_ptr", r))
                    .unwrap()
            };
            let r = ll
                .builder
                .build_load(ll.context.i32_type(), ptr, &format!("r{}", r))
                .unwrap()
                .into_int_value();

            regs.push(GuestRegRef { ptr, value: r });
        }

        // TODO - emit code here, stop at some point. Update regs as we go.

        for r in regs {
            ll.builder.build_store(r.ptr, r.value).unwrap();
        }
        ll.builder.build_return(None).unwrap();

        if function.verify(true) {
            ll.function = unsafe { Some(ll.execution_engine.get_function(&name).unwrap()) };
        }
    }
}

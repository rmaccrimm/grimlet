use std::collections::HashMap;

use super::state::GuestState;
use anyhow::{Result, anyhow};
use inkwell::builder::Builder;
use inkwell::context::{Context, ContextRef};
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::llvm_sys::LLVMValue;
use inkwell::llvm_sys::prelude::LLVMValueRef;
use inkwell::module::Module;
use inkwell::types::{FunctionType, IntType, PointerType, StructType};
use inkwell::values::{AsValueRef, BasicValue, BasicValueEnum, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};
use std::sync::Arc;

pub type CompiledBlock = unsafe extern "C" fn(*mut GuestState);

struct GuestRegRef<'a> {
    ptr: PointerValue<'a>,
    value: IntValue<'a>,
}

pub struct Compiler<'ctx> {
    context: ContextRef<'ctx>,
    module: Module<'ctx>,
    func_count: u32,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let module = context.create_module("main");
        Ok(Self {
            context: module.get_context(),
            module: module,
            func_count: 0,
        })
    }

    /*
       Helpers for accessing frequently used LLVM types (_ll)
    */

    fn i32_ll(&self) -> IntType {
        self.context.i32_type()
    }

    fn state_ll(&self) -> StructType {
        let types = [
            self.context.i32_type().array_type(16).into(),
            self.context.ptr_type(AddressSpace::default()).into(),
        ];
        self.context.struct_type(&types, false)
    }

    fn ptr_ll(&self) -> PointerType {
        self.context.ptr_type(AddressSpace::default())
    }

    fn fn_ll(&self) -> FunctionType {
        self.context
            .void_type()
            .fn_type(&[self.ptr_ll().into()], false)
    }

    /// Loads the guest machine register into an LLVM register. Both the pointer into the guest
    /// state and current value are maintained so it can be transfered back at the end
    fn load_reg<'a>(
        &'ctx self,
        r: usize,
        builder: &'a Builder,
        state_ptr: &'a PointerValue,
    ) -> GuestRegRef<'a>
    where
        'ctx: 'a,
    {
        let gep_inds = [
            self.i32_ll().const_int(0, false),
            self.i32_ll().const_int(r as u64, false),
        ];
        let ptr = unsafe {
            builder
                .build_gep(
                    self.state_ll(),
                    *state_ptr,
                    &gep_inds,
                    &format!("r{}_ptr", r),
                )
                .unwrap()
        };
        let r = builder
            .build_load(self.context.i32_type(), ptr, &format!("r{}", r))
            .unwrap()
            .into_int_value();

        GuestRegRef { ptr, value: r }
    }

    pub fn compile_test(&'ctx mut self) -> Option<JitFunction<CompiledBlock>> {
        let name = format!("block_{}", self.func_count);
        self.func_count += 1;
        let function = self.module.add_function(&name, self.fn_ll(), None);
        let basic_block = self.context.append_basic_block(function, "start");
        let builder = self.context.create_builder();
        builder.position_at_end(basic_block);

        let state_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let mut regs = Vec::new();
        for r in 0..17usize {
            regs.push(self.load_reg(r, &builder, &state_ptr));
        }

        // TODO - emit code here, stop at some point. Update regs as we go.

        for r in regs {
            builder.build_store(r.ptr, r.value).unwrap();
        }

        builder.build_return(None).unwrap();

        if function.verify(true) {
            return unsafe {
                self.module
                    .create_execution_engine()
                    .unwrap()
                    .get_function(&name)
                    .ok()
            };
        }
        None
    }
}

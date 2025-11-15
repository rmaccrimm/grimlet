use std::collections::HashMap;

use anyhow::{Result, anyhow};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::llvm_sys::LLVMValue;
use inkwell::llvm_sys::prelude::LLVMValueRef;
use inkwell::module::Module;
use inkwell::values::{BasicValue, BasicValueEnum, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use super::state::GuestState;

pub type CompiledBlock = unsafe extern "C" fn(*mut GuestState);

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    reg_map: HashMap<String, LLVMValueRef>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let module = context.create_module("sum");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|s| anyhow!(s.to_string()))?;
        let builder = context.create_builder();

        Ok(Self {
            context,
            module,
            builder,
            execution_engine,
            reg_map: HashMap::new(),
        })
    }

    pub fn compile_test(&self) -> Option<JitFunction<'_, CompiledBlock>> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[ptr_type.into()], false);
        let function = self.module.add_function("load_state", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "start");
        self.builder.position_at_end(basic_block);

        let types = [
            self.context.i32_type().array_type(16).into(),
            self.context.ptr_type(AddressSpace::default()).into(),
        ];
        let state_type = self.context.struct_type(&types, false);
        let state_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        let r0_ptr = unsafe {
            self.builder
                .build_gep(
                    state_type,
                    state_ptr,
                    &[
                        self.context.i32_type().const_int(0, false),
                        self.context.i32_type().const_int(0, false),
                    ],
                    "r0_ptr",
                )
                .unwrap()
        };
        let v0 = self
            .builder
            .build_load(self.context.i32_type(), r0_ptr, "r0")
            .unwrap()
            .into_int_value();

        let v1 = self
            .builder
            .build_int_mul(
                v0,
                self.context.i32_type().const_int(2, false).into(),
                "r0_2",
            )
            .unwrap();

        self.builder.build_store(r0_ptr, v1).unwrap();
        self.builder.build_return(None).unwrap();

        if function.verify(true) {
            return unsafe { self.execution_engine.get_function("load_state").ok() };
        }
        None
    }

    pub fn jit_compile_load(&self) -> Option<JitFunction<'_, CompiledBlock>> {
        todo!();
        // self.builder.position_at_end(basic_block);
        // let addr = function.get_nth_param(0)?.into_pointer_value();
        // let r0 = self
        //     .builder
        //     .build_load(self.context.i32_type(), addr, "r0")
        //     .ok()?
        //     .into_int_value();
        // let c = self.context.i32_type().const_int(5, false);
        // let res = self.builder.build_int_add(r0, c, "res").unwrap();
        // self.builder.build_store(addr, res).ok()?;
        // self.builder.build_return(None).ok()?;
    }
}

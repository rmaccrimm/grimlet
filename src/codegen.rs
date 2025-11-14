use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, BasicValueEnum, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use super::state::GuestState;

pub type CompiledBlock = unsafe extern "C" fn(*mut i32);

pub struct CodeGen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn jit_compile_load(&self) -> Option<JitFunction<'_, CompiledBlock>> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[ptr_type.into()], false);
        let function = self.module.add_function("load_state", fn_type, None);
        let basic_block = self
            .context
            .append_basic_block(function, "load_state_block");

        self.builder.position_at_end(basic_block);
        let addr = function.get_nth_param(0)?.into_pointer_value();
        let r0 = self
            .builder
            .build_load(self.context.i32_type(), addr, "r0")
            .ok()?
            .into_int_value();
        let c = self.context.i32_type().const_int(5, false);
        let res = self.builder.build_int_add(r0, c, "res").unwrap();
        self.builder.build_store(addr, res).ok()?;
        self.builder.build_return(None).ok()?;

        if function.verify(true) {
            return unsafe { self.execution_engine.get_function("load_state").ok() };
        }
        None
    }
}

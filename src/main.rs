use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, BasicValueEnum, IntValue};
use inkwell::{AddressSpace, OptimizationLevel};

use std::error::Error;

/// Convenience type alias for the `sum` function.
///
/// Calling this is innately `unsafe` because there's no guarantee it doesn't
/// do `unsafe` operations internally.
type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

type CompiledBlock = unsafe extern "C" fn(*mut CpuState);

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_load(&self) -> Option<JitFunction<'_, CompiledBlock>> {
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

#[repr(C)]
pub struct CpuState {
    pub regs: [u32; 16],
    pub mem: Box<[u8; 0xe010000]>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let load = codegen
        .jit_compile_load()
        .ok_or("Unable to JIT compile `load_state`")?;

    let mut x: i32 = 10;

    println!("Calling func...");
    unsafe {
        load.call(&mut x);
    }
    println!("x: {}", x);

    Ok(())
}

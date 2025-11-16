use crate::state::GuestState;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};

use inkwell::module::Module;
use inkwell::types::{FunctionType, IntType, PointerType, StructType};
use inkwell::values::{IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

pub type CompiledBlock = unsafe extern "C" fn(*mut GuestState);

struct RegMap<'a> {
    ptr: PointerValue<'a>,
    value: IntValue<'a>,
}

pub struct Compiler<'ctx> {
    pub function: Option<JitFunction<'ctx, CompiledBlock>>,
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    i32_type: IntType<'ctx>,
    ptr_type: PointerType<'ctx>,
    fn_type: FunctionType<'ctx>,
    state_type: StructType<'ctx>,
    func_count: u32,
}

impl<'ctx> Compiler<'ctx> {
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
            func_count: 0,
        }
    }

    /// Loads the guest machine registers into LLVM values and maintains a mapping from the pointer
    /// into the guest state and current value
    fn load_registers<'a>(&self, state_ptr: PointerValue<'a>) -> Vec<RegMap<'a>>
    where
        'ctx: 'a,
    {
        let mut regs = Vec::new();
        for r in 0..17usize {
            let gep_inds = [
                self.i32_type.const_zero(),
                self.i32_type.const_zero(),
                self.i32_type.const_int(r as u64, false),
            ];
            let ptr = unsafe {
                self.builder
                    .build_gep(
                        self.state_type,
                        state_ptr,
                        &gep_inds,
                        &format!("r{}_ptr", r),
                    )
                    .unwrap()
            };
            let r = self
                .builder
                .build_load(self.context.i32_type(), ptr, &format!("r{}", r))
                .unwrap()
                .into_int_value();

            regs.push(RegMap { ptr, value: r });
        }
        regs
    }

    pub fn compile<'a>(&mut self) {
        let name = format!("block_{}", self.func_count);
        self.func_count += 1;
        let function = self.module.add_function(&name, self.fn_type, None);
        let basic_block = self.context.append_basic_block(function, "start");
        self.builder.position_at_end(basic_block);

        let state_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let mut regs = self.load_registers(state_ptr.clone());

        // TODO - emit code here, stop at some point. Update regs as we go.
        regs[0].value = self
            .builder
            .build_int_add(regs[0].value, regs[1].value, "v0")
            .unwrap();
        regs[0].value = self
            .builder
            .build_int_add(regs[0].value, regs[2].value, "v1")
            .unwrap();
        regs[0].value = self
            .builder
            .build_int_mul(regs[0].value, regs[3].value, "v2")
            .unwrap();

        for r in regs {
            self.builder.build_store(r.ptr, r.value).unwrap();
        }
        self.builder.build_return(None).unwrap();

        if function.verify(true) {
            self.function = unsafe { Some(self.execution_engine.get_function(&name).unwrap()) };
        }
    }
}

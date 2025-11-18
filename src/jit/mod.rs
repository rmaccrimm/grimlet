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
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::{Linkage, Module};
use inkwell::types::{FunctionType, IntType, PointerType, StructType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

type CompiledFunc<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState)>;

struct RegMap<'a> {
    ptr: PointerValue<'a>,
    value: IntValue<'a>,
}

pub struct LlvmFunction<'a> {
    context: &'a Context,
    addr: u64,
    regs: Vec<RegMap<'a>>,
    f: FunctionValue<'a>,
    state_ptr: PointerValue<'a>,
}

pub struct Compiler<'ctx> {
    pub func_cache: HashMap<u32, CompiledFunc<'ctx>>,
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    i32_type: IntType<'ctx>,
    ptr_type: PointerType<'ctx>,
    fn_type: FunctionType<'ctx>,
    state_type: StructType<'ctx>,
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

        // Link to external ARM interpreter functions
        let args = [
            context.ptr_type(AddressSpace::default()).into(),
            i32_type.into(),
        ];
        let interp_fn_type = context.void_type().fn_type(&args, false);
        let interp_fn =
            module.add_function("ArmState::jump_to", interp_fn_type, Some(Linkage::External));
        execution_engine.add_global_mapping(&interp_fn, ArmState::jump_to as usize);

        Self {
            func_cache: HashMap::new(),
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

    /// Loads the guest machine registers into LLVM values and maintains a mapping from the pointer
    /// into the guest state to it's current LLVM value
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

    fn store_registers(&self, regs: &Vec<RegMap>) {
        for r in regs {
            self.builder.build_store(r.ptr, r.value).unwrap();
        }
    }

    pub fn new_function<'a>(&mut self, addr: u32) -> LlvmFunction<'a>
    where
        'ctx: 'a,
    {
        let name = format!("fn_{:#010x}", addr);
        let f = self.module.add_function(&name, self.fn_type, None);
        let basic_block = self.context.append_basic_block(f, "start");
        self.builder.position_at_end(basic_block);

        let state_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
        let regs = self.load_registers(state_ptr.clone());

        LlvmFunction {
            context: self.context,
            addr: addr as u64,
            regs,
            f,
            state_ptr,
        }
    }

    pub fn append_insn(&mut self, lctx: &LlvmFunction, addr: u32) {
        // TODO - emit code here, stop at some point. Update regs as we go.
        // let interp = &ArmState::jump_to;
        self.store_registers(&lctx.regs);
        let interp_fn = self.module.get_function("ArmState::jump_to").unwrap();
        self.builder
            .build_call(
                interp_fn,
                &[
                    lctx.state_ptr.into(),
                    self.i32_type.const_int(99, false).into(),
                ],
                "fn_result",
            )
            .unwrap();

        self.builder.build_return(None).unwrap();
    }

    pub fn compile(&mut self, func: LlvmFunction) {
        if func.f.verify(true) {
            let jit_func = unsafe {
                self.execution_engine
                    .get_function(&func.f.get_name().to_str().unwrap())
                    .unwrap()
            };
            self.func_cache.insert(func.addr as u32, jit_func);
        }
    }

    pub fn dump(&self, path: &str) {
        self.module.print_to_file(path).unwrap()
    }
}

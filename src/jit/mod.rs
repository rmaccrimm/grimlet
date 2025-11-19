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
use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

type JumpTarget = unsafe extern "C" fn(*const i32);

type CompiledFunc<'a> = JitFunction<'a, JumpTarget>;

type EntryPoint<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState, *const JumpTarget)>;

pub struct LlvmFunction<'a> {
    addr: u64,
    name: String,
    reg_map: Vec<IntValue<'a>>,
    module_ind: usize,
    ee_ind: usize,
    func: FunctionValue<'a>,
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
        let mut modules = Vec::new();
        modules.push(context.create_module("m_global"));
        let module = modules.last().unwrap();

        let mut engines = Vec::new();
        engines.push(module.create_execution_engine().unwrap());
        let ee = engines.last().unwrap();

        let builder = context.create_builder();
        let i32_t = context.i32_type();
        let ptr_t = context.ptr_type(AddressSpace::default());
        let void_t = context.void_type();
        let regs_t = i32_t.array_type(17);
        let arm_state_t = context.struct_type(&[regs_t.into(), ptr_t.into()], false);
        let fn_t = void_t.fn_type(&[ptr_t.into()], false);

        // Link to external ARM interpreter functions
        let args = [
            context.ptr_type(AddressSpace::default()).into(),
            i32_t.into(),
        ];
        let interp_fn_type = context.void_type().fn_type(&args, false);
        let interp_fn =
            modules[0].add_function("ArmState::jump_to", interp_fn_type, Some(Linkage::External));

        ee.add_global_mapping(&interp_fn, ArmState::jump_to as usize);

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

    pub fn build_entry_point(&mut self) -> EntryPoint<'ctx> {
        let ctx = self.llvm_ctx;
        let builder = &self.builder;

        self.modules.push(ctx.create_module("m_entrypoint"));
        let module = self.modules.last().unwrap();

        self.engines.push(
            module
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap(),
        );
        let ee = self.engines.last().unwrap();

        let entry_type = ctx
            .void_type()
            .fn_type(&[self.ptr_t.into(), self.ptr_t.into()], false);
        let f = module.add_function("fn_entry_point", entry_type, None);
        let basic_block = ctx.append_basic_block(f, "start");
        builder.position_at_end(basic_block);

        // First arg - pointer to ARM guest state, 2nd arg - function to jump to
        let state_ptr = f.get_nth_param(0).unwrap().into_pointer_value();
        let fn_ptr = f.get_nth_param(1).unwrap().into_pointer_value();

        let call_args = builder
            .build_alloca(self.i32_t.array_type(17), "reg_values")
            .unwrap();

        for r in 0..17usize {
            let gep_inds = [
                ctx.i32_type().const_zero(),
                ctx.i32_type().const_zero(),
                ctx.i32_type().const_int(r as u64, false),
            ];
            let name = format!("r{}_guest_ptr", r);
            let guest_ptr = unsafe {
                builder
                    .build_gep(self.arm_state_t, state_ptr, &gep_inds, &name)
                    .unwrap()
            };
            let value = builder
                .build_load(ctx.i32_type(), guest_ptr, &format!("r{}", r))
                .unwrap()
                .into_int_value();

            let gep_inds = [
                // self.i32_t.const_zero(),
                self.i32_t.const_int(r as u64, false),
            ];
            let name = format!("r{}_array_ptr", r);
            let arr_ptr = unsafe {
                builder
                    .build_gep(self.i32_t, call_args, &gep_inds, &name)
                    .unwrap()
            };
            builder.build_store(arr_ptr, value).unwrap();
        }
        builder
            .build_indirect_call(self.fn_t, fn_ptr, &[call_args.into()], "call")
            .unwrap();
        builder.build_return(None).unwrap();
        assert!(f.verify(true));

        let entry_point = unsafe { ee.get_function("fn_entry_point").unwrap() };
        entry_point
    }

    pub fn new_function<'a>(&mut self, addr: u32) -> LlvmFunction<'a>
    where
        'ctx: 'a,
    {
        let name = format!("fn_{:#010x}", addr);
        self.modules
            .push(self.llvm_ctx.create_module(&format!("m_{}", &name)));
        let module = self.modules.last().unwrap();

        self.engines.push(
            module
                .create_jit_execution_engine(OptimizationLevel::None)
                .unwrap(),
        );
        // let ee = self.engines.last().unwrap();

        let func = module.add_function(&name, self.fn_t, None);
        let basic_block = self.llvm_ctx.append_basic_block(func, "start");
        self.builder.position_at_end(basic_block);

        let reg_map = (0..17)
            .map(|i| {
                let arr = func.get_nth_param(0).unwrap().into_pointer_value();
                let name = format!("r{}_elem_ptr", i);
                let ptr = unsafe {
                    self.builder
                        .build_gep(
                            self.regs_t,
                            arr,
                            &[self.i32_t.const_zero(), self.i32_t.const_int(i, false)],
                            &name,
                        )
                        .unwrap()
                };
                let name = format!("r{}", i);
                self.builder
                    .build_load(self.i32_t, ptr, &name)
                    .unwrap()
                    .into_int_value()
            })
            .collect();

        LlvmFunction {
            addr: addr as u64,
            name,
            reg_map,
            module_ind: self.modules.len() - 1,
            ee_ind: self.engines.len() - 1,
            func,
        }
    }

    pub fn append_insn(&mut self, func: &LlvmFunction, addr: u32) {
        todo!();
        // TODO - emit code here, stop at some point. Update regs as we go.
        // let interp = &ArmState::jump_to;
        // let regs_ptr = func.module.get_global("regs").unwrap().as_pointer_value();
        // self.store_registers(&func.regs);
        // let module = &self.modules[func.module_ind];

        // let interp_fn = module.get_function("ArmState::jump_to").unwrap();
        // self.load_registers(&module.get_global("regs").unwrap().as_pointer_value());
        //     self.builder
        //         .build_call(
        //             interp_fn,
        //             &[
        //                 func.state_ptr.into(),
        //                 self.i32_t.const_int(99, false).into(),
        //             ],
        //             "fn_result",
        //         )
        //         .unwrap();

        //     self.builder.build_return(None).unwrap();
        // }
    }

    pub fn compile(&mut self, func: LlvmFunction) -> Result<&CompiledFunc<'ctx>> {
        self.builder.build_return(None).unwrap();
        if func.func.verify(true) {
            let jit_func = unsafe { self.engines[func.ee_ind].get_function(&func.name).unwrap() };
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
    fn test_compile() {
        let mut state = ArmState::new();
        let context = Context::create();
        let mut comp = Compiler::new(&context);

        let entry = comp.build_entry_point();
        let func = comp.new_function(0);
        let compiled = comp.compile(func).unwrap();
        comp.dump();

        unsafe {
            // compiled.call(&mut state);
            // entry.call(&mut state, &mut compiled.as_raw());
        }
    }
}

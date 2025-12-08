mod builder;

use std::collections::HashMap;
use std::fs;

use anyhow::Result;
use builder::FunctionBuilder;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;

use crate::arm::cpu::ArmState;

type JumpTarget = unsafe extern "C" fn(*mut ArmState);

pub type CompiledFunction<'a> = JitFunction<'a, JumpTarget>;

pub type FunctionCache<'ctx> = HashMap<usize, CompiledFunction<'ctx>>;

/// Manages LLVM compilation state and constructs new LlvmFunctions
pub struct Compiler<'ctx> {
    llvm_ctx: &'ctx Context,
    builder: Builder<'ctx>,
    modules: Vec<Module<'ctx>>,
    engines: Vec<ExecutionEngine<'ctx>>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let modules = Vec::new();
        let engines = Vec::new();
        let builder = context.create_builder();

        Self {
            llvm_ctx: context,
            modules,
            engines,
            builder,
        }
    }

    pub fn new_function<'a>(
        &'a mut self,
        addr: usize,
        func_cache: &'a FunctionCache<'ctx>,
    ) -> FunctionBuilder<'ctx, 'a>
    where
        'ctx: 'a,
    {
        let i = self.create_module(&format!("m_{}", addr));
        FunctionBuilder::new(
            addr,
            self.llvm_ctx,
            &self.builder,
            &self.modules[i],
            &self.engines[i],
            func_cache,
        )
    }

    pub fn dump(&self) -> Result<()> {
        if fs::exists("llvm")? && fs::metadata("llvm")?.is_dir() {
            fs::remove_dir_all("llvm")?
        }
        fs::create_dir("llvm")?;
        for (i, m) in self.modules.iter().enumerate() {
            m.print_to_file(format!("llvm/mod_{}.ll", i)).unwrap();
        }
        Ok(())
    }

    fn create_module(&mut self, name: &str) -> usize {
        self.modules.push(self.llvm_ctx.create_module(name));
        let module = self.modules.last().unwrap();
        self.engines.push(
            module
                .create_jit_execution_engine(OptimizationLevel::None)
                .expect("failed to create LLVM execution engine"),
        );
        self.modules.len() - 1
    }
}

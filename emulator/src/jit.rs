mod builder;

use crate::arm::cpu::{ArmState, NUM_REGS};
use builder::{FunctionBuilder, get_ptr_param};

use anyhow::Result;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::{AddressSpace, OptimizationLevel};
use std::collections::HashMap;
use std::fs;

type JumpTarget = unsafe extern "C" fn(*mut ArmState, *const i32);

pub type CompiledFunction<'a> = JitFunction<'a, JumpTarget>;

pub type EntryPoint<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState, JumpTarget)>;

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
    ) -> Result<FunctionBuilder<'ctx, 'a>>
    where
        'ctx: 'a,
    {
        let i = self.create_module(&format!("m_{}", addr));
        let lf = FunctionBuilder::new(
            addr,
            self.llvm_ctx,
            &self.builder,
            &self.modules[i],
            &self.engines[i],
            func_cache,
        );
        Ok(lf)
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

    // Performs context switch from guest machine to LLVM code and jumps to provided function
    pub fn compile_entry_point(&mut self) -> EntryPoint<'ctx> {
        let i = self.create_module("m_entrypoint");
        let ctx = self.llvm_ctx;
        let bd = &self.builder;
        let module = &self.modules[i];
        let ee = &self.engines[i];

        let i32_t = ctx.i32_type();
        let ptr_t = ctx.ptr_type(AddressSpace::default());
        let void_t = ctx.void_type();
        let regs_t = i32_t.array_type(NUM_REGS as u32);
        let arm_state_t = ArmState::get_llvm_type(ctx);
        let fn_t = void_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        let entry_type = self
            .llvm_ctx
            .void_type()
            .fn_type(&[ptr_t.into(), ptr_t.into()], false);
        let f = module.add_function("entry_point", entry_type, None);
        let basic_block = self.llvm_ctx.append_basic_block(f, "start");
        bd.position_at_end(basic_block);

        let arm_state_ptr = get_ptr_param(&f, 0);
        let fn_ptr_arg = get_ptr_param(&f, 1);

        let build = || -> Result<()> {
            let regs_ptr = bd.build_alloca(regs_t, "regs_ptr")?;

            // Perform context switch in, i.e. copy guest machine state into an array
            let zero = i32_t.const_zero();
            let one = i32_t.const_int(1, false);
            for r in 0..NUM_REGS {
                let reg_ind = i32_t.const_int(r as u64, false);
                let gep_inds = [zero, one, reg_ind];
                let name = format!("arm_state_r{}_ptr", r);
                // Pointer to the register in the guest machine (ArmState object)
                let arm_state_elem_ptr =
                    unsafe { bd.build_gep(arm_state_t, arm_state_ptr, &gep_inds, &name)? };
                let value = bd
                    .build_load(i32_t, arm_state_elem_ptr, &format!("r{}", r))?
                    .into_int_value();

                let gep_inds = [zero, reg_ind];
                let name = format!("reg_arr_r{}_ptr", r);
                // Pointer to the local register (i32 array)
                let reg_arr_elem_ptr = unsafe { bd.build_gep(regs_t, regs_ptr, &gep_inds, &name)? };
                bd.build_store(reg_arr_elem_ptr, value)?;
            }

            // Call the actual processing func. Note this is not a tail call as the stack becomes
            // corrupted (presumably because the regs_ptr array gets freed)
            bd.build_indirect_call(
                fn_t,
                fn_ptr_arg,
                &[arm_state_ptr.into(), regs_ptr.into()],
                "call",
            )?;
            bd.build_return(None)?;
            Ok(())
        };
        build().expect("LLVM codegen failed");
        assert!(f.verify(true));
        unsafe {
            ee.get_function("entry_point")
                .expect("failed to compile entry_point")
        }
    }
}

#[cfg(test)]
mod tests {

    use capstone::arch::arm::ArmInsn;

    use crate::arm::disasm::cons::*;

    #[test]
    fn test_next_code_block() {
        let program = [
            op_reg_imm(ArmInsn::ARM_INS_CMP, 0, 1, None),
            op_imm(ArmInsn::ARM_INS_B, 36, None),
            op_reg_reg(ArmInsn::ARM_INS_MOV, 1, 0, None),
            op_reg_imm(ArmInsn::ARM_INS_MOV, 0, 1, None),
            op_reg_reg_reg(ArmInsn::ARM_INS_MUL, 0, 0, 1, None),
            op_reg_reg_imm(ArmInsn::ARM_INS_SUBS, 1, 1, 1, None),
            op_imm(ArmInsn::ARM_INS_B, 16, None),
        ];
    }
}

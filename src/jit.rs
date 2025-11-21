mod alu;
mod branch;
mod compile;
mod ldstr;

use crate::arm::cpu::{ArmState, NUM_REGS, Reg};
use anyhow::{Context as _, Result, anyhow};
use std::collections::HashMap;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::{ArrayType, FunctionType, IntType, PointerType, StructType, VoidType};
use inkwell::values::{FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, OptimizationLevel};

type JumpTarget = unsafe extern "C" fn(*mut ArmState, *const i32);

type CompiledFunc<'a> = JitFunction<'a, JumpTarget>;

type EntryPoint<'a> = JitFunction<'a, unsafe extern "C" fn(*mut ArmState, JumpTarget)>;

/// State needed to build a new function. Returned by Compiler so you cannot attempt to compile
/// something that has not been initialized.
pub struct LlvmFunction<'a> {
    addr: u64,
    name: String,
    reg_map: Vec<IntValue<'a>>,
    module_ind: usize,
    func: FunctionValue<'a>,
    state_ptr: PointerValue<'a>,
    regs_ptr: PointerValue<'a>,
}

/// Helper that converts the LLVMString error message into an anyhow error
fn get_ptr_param<'a>(func: &FunctionValue<'a>, i: usize) -> Result<PointerValue<'a>> {
    Ok(func
        .get_nth_param(i as u32)
        .ok_or(anyhow!(
            "{} signature has no parameter {}",
            func.get_name().to_str()?,
            i
        ))?
        .into_pointer_value())
}

fn func_name(addr: i32) -> String {
    format!("fn_{:#010x}", addr)
}

/// Maps a guest machine address to a compiled LLVM function
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct FuncCacheKey(u32);

/// Core struct responsible for translating ARM instructions to LLVM and managing the compiled
/// executable code.
pub struct Compiler<'ctx> {
    func_cache: HashMap<FuncCacheKey, CompiledFunc<'ctx>>,
    entry_point: Option<EntryPoint<'ctx>>,
    active_func: bool,
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
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let modules = Vec::new();
        let engines = Vec::new();
        let builder = context.create_builder();
        let i32_t = context.i32_type();
        let ptr_t = context.ptr_type(AddressSpace::default());
        let void_t = context.void_type();
        let regs_t = i32_t.array_type(17);
        let arm_state_t = ArmState::get_llvm_type(context);
        let fn_t = void_t.fn_type(&[ptr_t.into(), ptr_t.into()], false);

        let mut comp = Self {
            func_cache: HashMap::new(),
            llvm_ctx: context,
            entry_point: None,
            active_func: false,
            modules,
            engines,
            builder,
            arm_state_t,
            i32_t,
            fn_t,
            ptr_t,
            regs_t,
            void_t,
        };
        let i = comp.create_module("m_entrypoint")?;
        comp.build_entry_point()
            .context("Failed to compile entry point")?;

        let ee = &comp.engines[i];
        comp.entry_point = Some(unsafe { ee.get_function("entry_point")? });
        Ok(comp)
    }

    pub fn new_function<'a>(&mut self, addr: u32) -> Result<LlvmFunction<'a>>
    where
        'ctx: 'a,
    {
        if self.active_func {
            return Err(anyhow!(
                "Compile current function before beginning a new one"
            ));
        }
        let fname = func_name(addr as i32);
        let i = self.create_module(&format!("m_{}", &fname))?;
        let bd = &self.builder;

        let module = &self.modules[i];
        let func = module.add_function(&fname, self.fn_t, None);
        let basic_block = self.llvm_ctx.append_basic_block(func, "start");
        bd.position_at_end(basic_block);

        let state_ptr = get_ptr_param(&func, 0)?;
        let regs_ptr = get_ptr_param(&func, 1)?;

        let reg_map = (0..17)
            .map(|i| {
                let name = format!("r{}_elem_ptr", i);
                let gep_inds = [self.i32_t.const_zero(), self.i32_t.const_int(i, false)];
                let ptr = unsafe {
                    bd.build_gep(self.regs_t, regs_ptr, &gep_inds, &name)
                        .unwrap()
                };
                let name = format!("r{}", i);
                bd.build_load(self.i32_t, ptr, &name)
                    .unwrap()
                    .into_int_value()
            })
            .collect();

        self.active_func = true;
        Ok(LlvmFunction {
            addr: addr as u64,
            name: fname,
            reg_map,
            module_ind: self.modules.len() - 1,
            func,
            state_ptr,
            regs_ptr,
        })
    }

    pub fn lookup_function(&self, addr: u32) -> Option<FuncCacheKey> {
        let k = FuncCacheKey(addr);
        match self.func_cache.get(&k) {
            Some(_) => Some(k),
            None => None,
        }
    }

    pub fn call_function(&self, k: FuncCacheKey, state: &mut ArmState) -> Result<()> {
        let func = self.func_cache.get(&k).expect("Nonexistent function key!");
        unsafe {
            self.entry_point
                .as_ref()
                .expect("Entry point is missing")
                .call(state, func.as_raw());
        }
        Ok(())
    }

    pub fn append_insn(&mut self, _func: &LlvmFunction, _addr: u32) {
        todo!();
    }

    pub fn compile(&mut self, func: LlvmFunction) -> Result<FuncCacheKey> {
        if func.func.verify(true) {
            let jit_func = unsafe {
                self.engines[func.module_ind]
                    .get_function(&func.name)
                    .unwrap()
            };
            let k = FuncCacheKey(func.addr as u32);
            // TODO - notify if this is a replacement?
            self.func_cache.insert(k, jit_func);
            self.active_func = false;
            Ok(k)
        } else {
            Err(anyhow!("Compilation failed"))
        }
    }

    pub fn dump(&self) {
        for (i, m) in self.modules.iter().enumerate() {
            m.print_to_file(&format!("mod_{}.ll", i)).unwrap();
        }
    }

    fn create_module(&mut self, name: &str) -> Result<usize> {
        self.modules.push(self.llvm_ctx.create_module(name));
        let module = self.modules.last().unwrap();
        self.engines.push(
            module
                .create_jit_execution_engine(OptimizationLevel::None)
                .map_err(|s| {
                    let er = s
                        .to_str()
                        .map(String::from)
                        .unwrap_or("Failed to create execution engine".into());
                    anyhow!(er)
                })?,
        );
        Ok(self.modules.len() - 1)
    }

    fn context_switch_in<'a>(
        &self,
        arm_state_ptr: PointerValue<'a>,
        regs_ptr: PointerValue<'a>,
    ) -> Result<()> {
        let bd = &self.builder;
        let zero = self.i32_t.const_zero();
        let one = self.i32_t.const_int(1, false);
        for r in 0..17usize {
            let reg_ind = self.i32_t.const_int(r as u64, false);
            let gep_inds = [zero, one, reg_ind];
            let name = format!("arm_state_r{}_ptr", r);
            // Pointer to the register in the guest machine (ArmState object)
            let arm_state_elem_ptr =
                unsafe { bd.build_gep(self.arm_state_t, arm_state_ptr, &gep_inds, &name)? };
            let value = bd
                .build_load(self.i32_t, arm_state_elem_ptr, &format!("r{}", r))?
                .into_int_value();

            let gep_inds = [zero, reg_ind];
            let name = format!("reg_arr_r{}_ptr", r);
            // Pointer to the local register (i32 array)
            let reg_arr_elem_ptr =
                unsafe { bd.build_gep(self.regs_t, regs_ptr, &gep_inds, &name)? };
            bd.build_store(reg_arr_elem_ptr, value)?;
        }
        Ok(())
    }

    fn context_switch_out<'a>(
        &self,
        arm_state_ptr: PointerValue<'a>,
        regs_ptr: PointerValue<'a>,
    ) -> Result<()> {
        let bd = &self.builder;
        let zero = self.i32_t.const_zero();
        let one = self.i32_t.const_int(1, false);
        for r in 0..NUM_REGS {
            let reg_ind = self.i32_t.const_int(r as u64, false);
            let gep_inds = [zero, reg_ind];
            let name = format!("reg_arr_r{}_ptr", r);
            // Pointer to the local register (i32 array)
            let reg_arr_elem_ptr =
                unsafe { bd.build_gep(self.regs_t, regs_ptr, &gep_inds, &name)? };
            let value = bd
                .build_load(self.i32_t, reg_arr_elem_ptr, &format!("r{}", r))?
                .into_int_value();

            let gep_inds = [zero, one, reg_ind];
            let name = format!("arm_state_r{}_ptr", r);
            // Pointer to the register in the guest machine (ArmState object)
            let arm_state_elem_ptr =
                unsafe { bd.build_gep(self.arm_state_t, arm_state_ptr, &gep_inds, &name)? };
            bd.build_store(arm_state_elem_ptr, value)?;
        }
        Ok(())
    }

    // Performs context switch from guest machine to LLVM code and jumps to provided function
    fn build_entry_point(&mut self) -> Result<EntryPoint<'ctx>> {
        let bd = &self.builder;
        let module = &self.modules[0];
        let ee = &self.engines[0];

        let entry_type = self
            .llvm_ctx
            .void_type()
            .fn_type(&[self.ptr_t.into(), self.ptr_t.into()], false);
        let f = module.add_function("entry_point", entry_type, None);
        let basic_block = self.llvm_ctx.append_basic_block(f, "start");
        bd.position_at_end(basic_block);

        let arm_state_ptr = get_ptr_param(&f, 0)?;
        let fn_ptr_arg = get_ptr_param(&f, 1)?;

        let regs_ptr = bd.build_alloca(self.regs_t, "regs_ptr")?;
        self.context_switch_in(arm_state_ptr, regs_ptr)?;

        let call = bd.build_indirect_call(
            self.fn_t,
            fn_ptr_arg,
            &[arm_state_ptr.into(), regs_ptr.into()],
            "call",
        )?;
        call.set_tail_call(true);
        bd.build_return(None)?;
        assert!(f.verify(true));

        let entry_point = unsafe { ee.get_function("entry_point")? };
        Ok(entry_point)
    }

    fn get_external_func_pointer(&self, func_addr: u64) -> Result<PointerValue<'ctx>> {
        let ee = &self.engines[0];
        let func_ptr = self.builder.build_int_to_ptr(
            self.llvm_ctx
                .ptr_sized_int_type(ee.get_target_data(), None)
                .const_int(func_addr as u64, false),
            self.ptr_t,
            &format!("extern_ptr"),
        )?;
        Ok(func_ptr)
    }

    fn get_compiled_func_pointer(&self, key: FuncCacheKey) -> Result<Option<PointerValue<'ctx>>> {
        // TODO sort out which int type to use where
        match self.func_cache.get(&key) {
            Some(f) => {
                // pretty sure it doesn't matter which we look at
                let ee = &self.engines[0];
                let func_ptr = unsafe {
                    self.builder.build_int_to_ptr(
                        self.llvm_ctx
                            .ptr_sized_int_type(ee.get_target_data(), None)
                            .const_int(f.as_raw() as u64, false),
                        self.ptr_t,
                        &format!("{}_ptr", func_name(key.0 as i32)),
                    )?
                };
                Ok(Some(func_ptr))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_to_external() {
        // End result is:
        // pc <- r15 + r9
        let mut state = ArmState::new();
        for i in 0..NUM_REGS {
            state.regs[i as usize] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context).unwrap();

        let func = comp.new_function(0).unwrap();
        let ee = &comp.engines[func.module_ind];

        let add_res = comp
            .builder
            .build_int_add(func.reg_map[15], func.reg_map[9], "add_res")
            .unwrap();

        let interp_fn_type = comp
            .void_t
            .fn_type(&[comp.ptr_t.into(), comp.i32_t.into()], false);

        let interp_fn_ptr = comp
            .get_external_func_pointer(ArmState::jump_to as u64)
            .unwrap();
        let call = comp
            .builder
            .build_indirect_call(
                interp_fn_type,
                interp_fn_ptr,
                &[func.state_ptr.into(), add_res.into()],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        comp.builder.build_return(None).unwrap();

        let key = comp.compile(func).unwrap();
        // comp.dump();
        println!("{:?}", state.regs);
        comp.call_function(key, &mut state).unwrap();
        println!("{:?}", state.regs);
        assert_eq!(state.pc(), 306);
    }

    #[test]
    fn test_cross_module_calls() {
        // f1:
        //      pc <- r0 - r3 - r2
        // f2:
        //      r0 <- 999
        //      f1()
        // (don't yet write registers besides PC pack to state)
        let mut state = ArmState::new();
        for i in 0..NUM_REGS {
            state.regs[i as usize] = (i * i) as u32;
        }

        let context = Context::create();
        let mut comp = Compiler::new(&context).unwrap();

        let f1 = comp.new_function(0).unwrap();
        let ee = &comp.engines[f1.module_ind];

        let v0 = comp
            .builder
            .build_int_add(f1.reg_map[3], f1.reg_map[2], "v0")
            .unwrap();
        let v1 = comp.builder.build_int_sub(f1.reg_map[0], v0, "v1").unwrap();

        // Perform context switch out before jumping to ArmState code
        comp.context_switch_out(f1.state_ptr, f1.regs_ptr).unwrap();

        let func_ptr_param = comp
            .get_external_func_pointer(ArmState::jump_to as u64)
            .unwrap();

        let interp_fn_t = comp
            .void_t
            .fn_type(&[comp.ptr_t.into(), comp.i32_t.into()], false);

        let call = comp
            .builder
            .build_indirect_call(
                interp_fn_t,
                func_ptr_param,
                &[f1.state_ptr.into(), v1.into()],
                "fn_result",
            )
            .unwrap();
        call.set_tail_call(true);
        comp.builder.build_return(None).unwrap();
        let k1 = comp.compile(f1).unwrap();

        let f2 = comp.new_function(0).unwrap();
        unsafe {
            // This will later be part of a build_call method
            // 1. store latest version of each register back on the stack. Can probably optimize
            //    this later by only storing those that actually change (or maybe LLVM does this?)
            //    Only doing r0 for this test
            let r0_elem_ptr = comp
                .builder
                .build_gep(
                    comp.i32_t.array_type(17),
                    f2.regs_ptr,
                    &[comp.i32_t.const_zero(), comp.i32_t.const_zero()],
                    "r0_elem_ptr",
                )
                .unwrap();
            comp.builder
                .build_store(r0_elem_ptr, comp.i32_t.const_int(999, false))
                .unwrap();

            // 2. Construct the function pointer using raw pointer obtained from function cache
            let func_ptr_param = comp.get_compiled_func_pointer(k1).unwrap().unwrap();

            // 3. Perform indirect call through pointer
            let call = comp
                .builder
                .build_indirect_call(
                    comp.fn_t,
                    func_ptr_param,
                    &[f2.state_ptr.into(), f2.regs_ptr.into()],
                    "call",
                )
                .unwrap();
            call.set_tail_call(true);
            comp.builder.build_return(None).unwrap();
        }
        let key = comp.compile(f2).unwrap();

        comp.dump();
        println!("{:?}", state.regs);
        comp.call_function(key, &mut state).unwrap();
        println!("{:?}", state.regs);

        assert_eq!(
            state.regs,
            [
                999, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 986, 256
            ]
        );
    }
}

#[cfg(test)]
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
    let entry_point = comp.compile_entry_point().unwrap();

    let func_cache = HashMap::new();
    let f = comp.new_function(0, &func_cache).unwrap();

    let add_res = f
        .builder
        .build_int_add(f.reg_map.get(Reg::PC), f.reg_map.get(Reg::R9), "add_res")
        .unwrap();

    let interp_fn_type = f.void_t.fn_type(&[f.ptr_t.into(), f.i32_t.into()], false);

    let interp_fn_ptr = f
        .get_external_func_pointer(ArmState::jump_to as u64)
        .unwrap();
    let call = f
        .builder
        .build_indirect_call(
            interp_fn_type,
            interp_fn_ptr,
            &[f.arm_state_ptr.into(), add_res.into()],
            "fn_result",
        )
        .unwrap();
    // call.set_tail_call(true);
    f.builder.build_return(None).unwrap();

    let compiled = f.compile().unwrap();

    println!("{:?}", state.regs);
    unsafe {
        entry_point.call(&mut state, compiled.as_raw());
    }
    println!("{:?}", state.regs);
    assert_eq!(state.pc(), 306);
}

#[test]
fn test_cross_module_calls() {
    // f1:
    //  pc <- r0 - r3 - r2
    // f2:
    //  r0 <- 999
    //  f1()
    // (don't yet write registers besides PC pack to state)
    let mut state = ArmState::new();
    for i in 0..NUM_REGS {
        state.regs[i as usize] = (i * i) as u32;
    }

    let context = Context::create();
    let mut comp = Compiler::new(&context).unwrap();
    let entry_point = comp.compile_entry_point().unwrap();

    let mut cache = HashMap::new();
    let f1 = comp.new_function(0, &cache).unwrap();

    let r0 = f1.reg_map.get(Reg::R0);
    let r2 = f1.reg_map.get(Reg::R2);
    let r3 = f1.reg_map.get(Reg::R3);
    let v0 = f1.builder.build_int_add(r3, r2, "v0").unwrap();
    let v1 = f1
        .builder
        .build_int_sub(
            // r0,
            r0, v0, "v1",
        )
        .unwrap();

    // Perform context switch out before jumping to ArmState code
    f1.write_state_out().unwrap();

    let func_ptr_param = f1
        .get_external_func_pointer(ArmState::jump_to as u64)
        .unwrap();

    let interp_fn_t = f1
        .void_t
        .fn_type(&[f1.ptr_t.into(), f1.i32_t.into()], false);

    let call = f1
        .builder
        .build_indirect_call(
            interp_fn_t,
            func_ptr_param,
            &[f1.arm_state_ptr.into(), v1.into()],
            // &[
            //     f1.arm_state_ptr.into(),
            //     f1.i32_t.const_int(843, false).into(),
            // ],
            "fn_result",
        )
        .unwrap();
    call.set_tail_call(true);
    f1.builder.build_return(None).unwrap();

    let compiled1 = f1.compile().unwrap();
    let k1 = FuncCacheKey(0);
    cache.insert(k1, compiled1);

    let f2 = comp.new_function(1, &cache).unwrap();
    let r0_elem_ptr = unsafe {
        // This will later be part of a build_call method
        // 1. store latest version of each register back on the stack. Can probably optimize
        //    this later by only storing those that actually change (or maybe LLVM does this?)
        //    Only doing r0 for this test
        f2.builder
            .build_gep(
                f2.i32_t.array_type(17),
                f2.reg_array_ptr,
                &[f2.i32_t.const_zero(), f2.i32_t.const_zero()],
                "r0_elem_ptr",
            )
            .unwrap()
    };
    f2.builder
        .build_store(r0_elem_ptr, f2.i32_t.const_int(999, false))
        .unwrap();

    // 2. Construct the function pointer using raw pointer obtained from function cache
    let func_ptr_param = f2.get_compiled_func_pointer(k1).unwrap().unwrap();

    // 3. Perform indirect call through pointer
    let call = f2
        .builder
        .build_indirect_call(
            f2.fn_t,
            func_ptr_param,
            &[f2.arm_state_ptr.into(), f2.reg_array_ptr.into()],
            "call",
        )
        .unwrap();
    call.set_tail_call(true);
    f2.builder.build_return(None).unwrap();

    let compiled2 = f2.compile().unwrap();

    comp.dump();
    println!("{:?}", state.regs);
    unsafe {
        entry_point.call(&mut state, compiled2.as_raw());
    }
    println!("{:?}", state.regs);

    assert_eq!(
        state.regs,
        [
            999, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 986, 256
        ]
    );
}

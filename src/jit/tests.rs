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

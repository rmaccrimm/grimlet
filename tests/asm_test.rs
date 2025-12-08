use grimlet::arm::cpu::{ArmState, Reg};
use grimlet::arm::disasm::MemoryDisassembler;
use grimlet::emulator::Emulator;
use inkwell::context::Context;

#[test]
fn test_factorial() {
    let context = Context::create();
    let disasm = MemoryDisassembler::default();
    let mut emulator =
        Emulator::new(&context, disasm, Some("tests/programs/factorial.gba")).unwrap();
    let exit = Some(|st: &ArmState| -> bool { st.pc() >= 40 });

    let mut run = |n| -> u32 {
        emulator.state = ArmState::default();
        emulator.state.regs[Reg::R0 as usize] = n;
        emulator.run(Some(|st: &ArmState| -> bool { st.pc() == 44 }));
        emulator.state.regs[Reg::R1 as usize]
    };
    // assert_eq!(run(0), 1);
    // assert_eq!(run(1), 1);
    assert_eq!(run(2), 2);
    assert_eq!(run(3), 6);
    assert_eq!(run(4), 24);
    assert_eq!(run(5), 120);
    assert_eq!(run(12), 479001600);
}

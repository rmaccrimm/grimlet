use grimlet::arm::disasm::Disassembler;
use grimlet::arm::state::{ArmState, Reg};
use grimlet::emulator::{DebugOutput, Emulator};

#[test]
fn test_factorial() {
    let disasm = Disassembler::default();
    let mut emulator = Emulator::new(disasm, Some("tests/programs/factorial.gba")).unwrap();

    let mut run = |n| -> u32 {
        emulator.state.jump_to(0);
        emulator.state.regs[Reg::R0] = n;
        emulator.run(|st: &ArmState| -> bool { st.curr_instr_addr() >= 40 }, None);
        emulator.state.regs[Reg::R1]
    };
    assert_eq!(run(0), 1);
    assert_eq!(run(1), 1);
    assert_eq!(run(2), 2);
    assert_eq!(run(3), 6);
    assert_eq!(run(4), 24);
    assert_eq!(run(5), 120);
    assert_eq!(run(12), 479001600);
}

#[test]
fn test_basic_load_store() {
    let disasm = Disassembler::default();
    let mut emulator = Emulator::new(disasm, Some("tests/programs/load_store.gba")).unwrap();
    let exit = |st: &ArmState| -> bool { st.curr_instr_addr() >= 812 };
    emulator.run(exit, Some(DebugOutput::Assembly));
    println!("{:?}", &emulator.state.regs);
    println!("{:?}", &emulator.state.mem.bios[0x4000 - (32)..]);
    assert_eq!(emulator.state.mem.read::<u32>(0x3ffc), 1);
}

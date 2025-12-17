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
    let exit = |st: &ArmState| -> bool { st.regs[Reg::R11] == 25344 };
    emulator.run(exit, Some(DebugOutput::Struct));

    let bios = &emulator.state.mem.bios;
    println!("{:?}", &emulator.state.regs);
    println!("{:?}", &bios[0x4000 - (32)..]);

    let num_tests = 2;
    for (i, w) in bios
        .rchunks(4)
        .map(|ch| i32::from_le_bytes(ch.try_into().unwrap()))
        .enumerate()
    {
        if i == num_tests {
            break;
        }
        assert_eq!(w, 1);
    }
}

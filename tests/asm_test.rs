use grimlet::arm::disasm::Disassembler;
use grimlet::arm::state::{ArmState, Reg};
use grimlet::emulator::{DebugOutput, Emulator};

/// Labels produced by gvasm assume it's loaded into cartridge ROM
const CART_START_ADDR: u32 = 0x08000000;

#[test]
fn test_factorial() {
    let disasm = Disassembler::default();
    let mut emulator = Emulator::new(disasm);
    emulator
        .load_rom("tests/programs/factorial.gba", CART_START_ADDR)
        .unwrap();

    let mut run = |n| -> u32 {
        emulator.state.jump_to(CART_START_ADDR);
        emulator.state.regs[Reg::R0] = n;
        emulator.run(
            |st: &ArmState| -> bool { st.curr_instr_addr() >= 40 + (CART_START_ADDR as usize) },
            Some(DebugOutput::Assembly),
        );
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
    let mut emulator = Emulator::new(disasm);
    emulator
        .load_rom("tests/programs/load_store.gba", CART_START_ADDR)
        .unwrap();
    let exit = |st: &ArmState| -> bool { st.regs[Reg::R11] == 25344 };
    emulator.state.jump_to(CART_START_ADDR);
    emulator.run(exit, Some(DebugOutput::Assembly));
    println!("{:08x?}", emulator.state.regs);

    let num_asserts = emulator.state.regs[Reg::R8];
    let mut result_addr = 0x4000 - 4;
    for i in 1..=num_asserts {
        let word = emulator.state.mem.read::<u32>(result_addr);
        println!("{}: {}", i, word);
        assert_eq!(word, 1, "failed on assertion {}", i);
        result_addr -= 4;
    }
    println!("{} assertions passed!", num_asserts);

    assert_eq!(emulator.state.regs[Reg::SP], 0x4000 - (4 * num_asserts));
}

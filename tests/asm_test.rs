use grimlet::arm::disasm::Disassembler;
use grimlet::arm::state::memory::ReadVal;
use grimlet::arm::state::{ArmMode, ArmState, Reg};
use grimlet::emulator::{DebugOutput, Emulator};
use inkwell::context::Context;

/// Labels produced by gvasm assume it's loaded into cartridge ROM
const CART_START_ADDR: u32 = 0x08000000;

const STACK_ADDR: u32 = 0x03008000;

const EXIT_VAL: u32 = 0x6300;

/// Simple framework for test cases written in assembly. Tests run assertions which, push 1 onto the
/// stack (currently at 0x4000, end of the bios region), otherwise -1. The number of assertions made
/// is written to register r8 and an exit is signalled to the emulator by writin 25344 to r11 and
/// jumping to a no-op.
macro_rules! assembly_test {
    ($name:ident.gba) => {
        #[test]
        fn $name() {
            let path = format!("tests/programs/{}.gba", stringify!($name));
            let disasm = Disassembler::default();
            let ctx = Context::create();
            let mut emulator = Emulator::new(disasm, &ctx);
            emulator.load_rom(path, CART_START_ADDR).unwrap();
            let exit = |st: &ArmState| -> bool { st.regs[Reg::R11] == EXIT_VAL };
            emulator.state.jump_to(CART_START_ADDR, ArmMode::ARM as i8);
            emulator.run(exit, Some(DebugOutput::Assembly));
            println!("{:08x?}", emulator.state.regs);

            let num_asserts = emulator.state.regs[Reg::R8];
            let mut result_addr = STACK_ADDR - 4;
            for i in 1..=num_asserts {
                let ReadVal { value, .. } = emulator.state.mem.read::<u32>(result_addr);
                println!("{}: {}", i, value);
                assert_eq!(value, 1, "failed on assertion {}", i);
                result_addr -= 4;
            }
            println!("{} assertions passed!", num_asserts);
            assert_eq!(emulator.state.regs[Reg::SP], STACK_ADDR - (4 * num_asserts));
        }
    };
}

assembly_test!(load_store.gba);
assembly_test!(bx.gba);
assembly_test!(mov_pc.gba);
assembly_test!(factorial.gba);
assembly_test!(flags.gba);

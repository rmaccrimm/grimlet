use core::arm::state::memory::ReadVal;
use core::arm::state::{ArmMode, ArmState, Reg};
use core::emulator::Emulator;

/// Labels produced by gvasm assume it's loaded into cartridge ROM
const CART_START_ADDR: u32 = 0x08000000;

const STACK_ADDR: u32 = 0x03008000;

const EXIT_VAL: u32 = 0x6300;

/// Simple framework for test cases written in assembly. Tests run assertions which, push 1 onto the
/// stack (currently at 0x4000, end of the bios region), otherwise -1. The number of assertions made
/// is written to register r7 and an exit is signalled to the emulator by writin 25344 to r11 and
/// jumping to an infinite loop.
macro_rules! assembly_test {
    ($name:ident.gba) => {
        #[test]
        fn $name() {
            let path = format!("tests/bin/{}.gba", stringify!($name));
            let mut emulator = Emulator::new();
            let exit = |st: &ArmState| -> bool { st.regs[Reg::R11] == EXIT_VAL };

            emulator.load_rom(path, CART_START_ADDR).unwrap();
            emulator.state.jump_to(CART_START_ADDR, ArmMode::ARM as i8);
            emulator.run(exit);

            let num_cases = emulator.state.regs[Reg::R6];
            let mut result_addr = STACK_ADDR - 4;
            for i in 1..=num_cases {
                let ReadVal { value, .. } = emulator.state.mem.read::<u32>(result_addr);
                let result = value as i32;
                assert_eq!(result, 1, "failed on case {}", i);
                result_addr -= 4;
            }
            println!("{} cases passed", num_cases);
            assert_eq!(
                emulator.state.regs[Reg::SP],
                STACK_ADDR - (4 * num_cases),
                "wrong stack pointer - fewer than expected cases ran"
            );
        }
    };
}

assembly_test!(bx.gba);
assembly_test!(cond.gba);
assembly_test!(cmp_flags.gba);
assembly_test!(factorial.gba);
assembly_test!(load_store.gba);
assembly_test!(mov_pc.gba);
assembly_test!(shifts.gba);

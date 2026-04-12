use capstone::Capstone;
use capstone::arch::BuildsCapstone as _;
use capstone::arch::arm::ArchMode;
use clap::Parser;

#[derive(Parser)]
struct Args {
    n: Vec<i32>,
}

fn main() -> anyhow::Result<()> {
    let buf = [
        0xff, 0xf7, 0xf5, 0xff, 0xff, 0xf7, 0xf5, 0xff, 0x00, 0x20, 0x00, 0x21,
    ];
    let mut cs = Capstone::new()
        .arm()
        .mode(ArchMode::Arm)
        .detail(true)
        .build()?;

    cs.set_mode(ArchMode::Thumb.into()).unwrap();
    let ins = cs.disasm_all(&buf[0..4], 0)?;
    for instr in ins.iter() {
        println!("{instr}");
    }
    Ok(())
}

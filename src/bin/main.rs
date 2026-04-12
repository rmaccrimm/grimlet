use capstone::Capstone;
use capstone::arch::BuildsCapstone as _;
use clap::Parser;
use grimlet::utils::interval_tree::IntervalTree;

#[derive(Parser)]
struct Args {
    n: Vec<i32>,
}

fn main() -> anyhow::Result<()> {
    let buf = [0xff, 0xf7, 0xf5, 0xff, 0xea, 0x20];
    let cs = Capstone::new()
        .arm()
        .mode(capstone::arch::arm::ArchMode::Thumb)
        .detail(true)
        .build()?;

    let ins = cs.disasm_count(&buf, 0, 1)?;
    for instr in ins.iter() {
        println!("{:#?}", instr);
    }
    Ok(())
}

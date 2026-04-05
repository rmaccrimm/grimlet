use clap::Parser;
use grimlet::utils::interval_tree::IntervalTree;

#[derive(Parser)]
struct Args {
    n: Vec<i32>,
}

fn main() -> anyhow::Result<()> {
    // let Args { n } = Args::parse();
    let n = [10, 11];
    let mut t = IntervalTree::default();
    for i in 0..15 {
        t.insert((i, i));
    }
    for &i in n.iter() {
        t.remove((i, i))?;
    }
    println!("{}", t);
    Ok(())
}

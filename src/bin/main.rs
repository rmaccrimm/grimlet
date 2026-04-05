use grimlet::utils::interval_tree::IntervalTree;

fn main() -> anyhow::Result<()> {
    let mut t = IntervalTree::default();
    for i in 0..15 {
        t.insert((i, i));
    }
    t.remove((3, 3))?;
    println!("{}", t);
    Ok(())
}

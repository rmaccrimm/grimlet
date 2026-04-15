use core::utils::interval_tree::IntervalTree;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser)]
struct Args {
    path: PathBuf,
}

#[derive(Debug)]
enum Op {
    Insert(i32, i32),
    Delete(i32),
}

// Reference, 2026-04-08
// Without dheap: 1mil ins -> .5mil del -> 1mil ins -> .5mil del
// Completed in 24.20s
// 300000 rounds of 10 ins -> 1 del all
// Completed in 3.64s
fn main() -> Result<()> {
    let args = Args::parse();
    let f = File::open(&args.path)?;
    let reader = BufReader::new(f);

    let mut ops = vec![];
    for line in reader.lines() {
        let l = line?;
        let mut sp = l.split(" ");
        let op = sp.next().expect("no op field");
        let start: i32 = sp
            .next()
            .expect("no start field")
            .parse()
            .expect("parsing start failed");
        if op == "i" {
            let end: i32 = sp
                .next()
                .expect("no end field")
                .parse()
                .expect("parsing end failed");
            ops.push(Op::Insert(start, end));
        } else {
            ops.push(Op::Delete(start));
        }
    }

    println!("Performing {} operations...", ops.len());

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let mut t = IntervalTree::<i32>::default();
    let now = Instant::now();
    for op in ops {
        match op {
            Op::Insert(s, e) => t.insert((s, e)),
            Op::Delete(s) => {
                t.remove_all(s);
            }
        }
    }
    let elapsed = now.elapsed();
    println!("Completed in {:.2?}", elapsed);
    t.verify();
    println!("{} nodes in tree", t.len());
    t.print_stats();
    Ok(())
}

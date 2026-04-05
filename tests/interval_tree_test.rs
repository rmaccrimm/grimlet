use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::Result;
use grimlet::utils::interval_tree::{self, IntervalTree};

#[derive(Debug)]
enum Op {
    Insert(i32, i32),
    Delete(i32, i32),
}

#[test]
fn time_bulk_inserts_delete() -> Result<()> {
    let f = File::open("tests/data/tree_ops.txt")?;
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
        // let end: i32 = sp
        //     .next()
        //     .expect("no end field")
        //     .parse()
        //     .expect("parsing end failed");
        if op == "i" {
            ops.push(Op::Insert(start, start));
        } else {
            ops.push(Op::Delete(start, start));
        }
    }
    let mut t = IntervalTree::<i32>::default();
    let now = Instant::now();
    for op in ops {
        match op {
            Op::Insert(s, e) => t.insert((s, e)),
            Op::Delete(s, e) => match t.remove((s, e)) {
                Ok(()) => (),
                Err(e) => println!("{}", e),
            },
        }
    }
    let elapsed = now.elapsed();
    println!("Completed in {:.2?}s", elapsed);
    t.verify(t.root.unwrap(), None, None);
    Ok(())
}

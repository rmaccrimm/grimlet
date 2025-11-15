use anyhow::Result;

/// num_vertices, num_edges
/// (v1, v2, w)
const INPUT: &'static str = "5 6
1 2 2
2 5 5
2 3 4
1 4 1
4 3 3
3 5 1
";

#[derive(Default, Debug)]
struct MinHeap {
    // index, priority
    heap: Vec<(i32, i32)>,
}

impl MinHeap {
    fn push(&mut self, x: i32, priority: i32) {
        self.heap.push((x, priority));
        self.sift_up(self.heap.len() - 1);
    }

    fn top(&self) -> Option<(i32, i32)> {
        if self.heap.len() > 0 {
            Some(self.heap[0])
        } else {
            None
        }
    }

    fn pop(&mut self) -> Option<(i32, i32)> {
        if self.heap.len() == 0 {
            return None;
        }
        let top = self.heap[0];
        let last = self.heap.pop().unwrap();
        if self.heap.len() > 0 {
            self.heap[0] = last;
            self.sift_down(0);
        }
        Some(top)
    }

    fn parent(&self, ind: usize) -> Option<usize> {
        if ind == 0 { None } else { Some((ind - 1) / 2) }
    }

    fn children(&self, ind: usize) -> [Option<usize>; 2] {
        let c1 = 2 * ind + 1;
        let c2 = 2 * ind + 2;
        [
            (c1 < self.heap.len()).then(|| c1),
            (c2 < self.heap.len()).then(|| c2),
        ]
    }

    fn sift_down(&mut self, ind: usize) {
        let mut curr_ind = ind;
        loop {
            let curr = self.heap[curr_ind];
            let [i1, i2] = self.children(curr_ind);

            let mut g1 = true;
            let mut g2 = true;
            if let Some(i) = i1 {
                let c1 = self.heap[i];
                g1 = c1.1 > curr.1;
                if !g1 {
                    self.heap[curr_ind] = c1;
                    self.heap[i] = curr;
                    curr_ind = i;
                }
            } else if let Some(i) = i2 {
                let c2 = self.heap[i];
                g2 = c2.1 > curr.1;
                if !g2 {
                    self.heap[curr_ind] = c2;
                    self.heap[i] = curr;
                    curr_ind = i;
                }
            }
            if g1 && g2 {
                return;
            }
        }
    }

    fn sift_up(&mut self, ind: usize) {
        let mut curr_ind = ind;
        loop {
            let curr = self.heap[curr_ind];
            let pi = self.parent(curr_ind);

            if let Some(i) = pi {
                let p = self.heap[i];
                if p.1 > curr.1 {
                    self.heap[curr_ind] = p;
                    self.heap[i] = curr;
                    curr_ind = i;
                } else {
                    return;
                }
            } else {
                return;
            }
        }
    }
}

impl MinHeap {
    fn new() {
        todo!()
    }
}

struct Graph {
    v: usize,
    e: usize,
    adj: Box<[i32]>,
}

impl Graph {
    fn from_str(s: &str) -> Self {
        let mut line_iter = s.lines();
        let vals: Vec<usize> = line_iter
            .next()
            .unwrap()
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let [v, e] = vals[0..2] else { panic!() };
        let mut adj = vec![-1; v * v].into_boxed_slice();

        for _ in 0..e {
            let vals: Vec<i32> = line_iter
                .next()
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let [a, b, w] = vals[0..3] else { panic!() };
            adj[(2 * a + b) as usize] = w;
        }
        Graph { v, e, adj }
    }
}

fn main() -> Result<()> {
    let mut h = MinHeap::default();
    h.push(0, 5);
    assert_eq!(h.top().unwrap().0, 0);
    h.push(1, 12);
    assert_eq!(h.top().unwrap().0, 0);
    h.push(2, 7);
    assert_eq!(h.top().unwrap().0, 0);
    h.push(3, -1);
    assert_eq!(h.top().unwrap().0, 3);
    h.push(4, 0);
    println!("{:#?}", h);
    h.pop();
    println!("{:#?}", h);
    assert_eq!(h.top().unwrap().0, 4);

    Ok(())
}

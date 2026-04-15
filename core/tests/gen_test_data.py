from random import randint

with open('tests/data/tree_ops.txt', 'w+') as f:
    for _ in range(300000):
        for _ in range(10):
            start = randint(int(-1e6), int(1e6))
            size = randint(0, 1000)            
            f.write(f"i {start} {start + size}\n")

        d = randint(int(-1e6), int(1e6))
        f.write(f"d {d}\n")

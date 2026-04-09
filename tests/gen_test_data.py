from random import randint

inserted = []
with open('data/tree_ops.txt', 'w+') as f:
    for _ in range(2):
        for _ in range(1000000):
            start = randint(int(-1e6), int(1e6))
            size = randint(0, 1000)
            inserted.append((start, start + size))
            f.write(f"i {start} {start + size}\n")
        for _ in range(500000):
            i = randint(0, len(inserted) - 1)
            p = inserted.pop(i)
            f.write(f"d {p[0]} {p[1]}\n")

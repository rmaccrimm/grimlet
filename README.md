# Grimlet (working title)

My second attempt at a GBA emulator using just-in-time recompilation. This time, built on LLVM.

Named after this guy ↴

![185](https://github.com/user-attachments/assets/4a335bae-0c3b-4cf4-99af-378e03818895)


### Debugging Environment Variables

| Variable          | Description                                | Possible Values                                             |
|-------------------|--------------------------------------------|-------------------------------------------------------------|
| `PRINT_STATE`     | Print CPU state after each block executes  | `true`                                                      |
| `DEBUG_OUTPUT`    | Print each disassembled block              | `assembly`<br>`struct`                                      |
| `DUMP_LLVM`       | Dump generated LLVM IR to file             | `on-fail` <br>`before-compilation` <br> `after-compilation` |
| `LLVM_OUTPUT_DIR` | Where to write LLVM files (default "llvm") |                                                             |
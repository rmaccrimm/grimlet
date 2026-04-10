# Grimlet (working title)

My second attempt at a GBA emulator using just-in-time recompilation. This time, built on LLVM.

Named after this guy ↴

![185](https://github.com/user-attachments/assets/4a335bae-0c3b-4cf4-99af-378e03818895)


## Debugging Environment Variables

| Variable       | Description                               | Values                                                                |
|----------------|-------------------------------------------|-----------------------------------------------------------------------|
| `DEBUG_OUTPUT` | Print each compiled block                 | `assembly`- readable assembly <br>`struct` - in-memory representation |
| `PRINT_STATE`  | Print CPU state after each block executes | `true` <br> `false`                                                   |
| `DUMP_LLVM`    | Dump generated LLVM IR to file            | `on-fail` <br>`before-compilation` <br> `after-compilation`           |
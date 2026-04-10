# Grimlet (working title)

This is an early work-in-progress GBA emulator utilizing LLVM for just-in-time recompilation. Currently it is only targeting ARM-based Macs.

Named after this guy ↴

![185](https://github.com/user-attachments/assets/4a335bae-0c3b-4cf4-99af-378e03818895)


### Status
- CPU emulation is mostly complete. All instructions have been implemented though timing is very inaccurate and many have not been fully tested. Can run small ARM programs such as those in tests/programs.


### Environment Variable Reference

| Variable          | Description                                | Possible Values                                             |
|-------------------|--------------------------------------------|-------------------------------------------------------------|
| `PRINT_STATE`     | Print CPU state after each block executes  | `true`                                                      |
| `DEBUG_OUTPUT`    | Print each disassembled block              | `assembly`<br>`struct`                                      |
| `DUMP_LLVM`       | Dump generated LLVM IR to file             | `on-fail` <br>`before-compilation` <br> `after-compilation` |
| `LLVM_OUTPUT_DIR` | Where to write LLVM files (default "llvm") |                                                             |
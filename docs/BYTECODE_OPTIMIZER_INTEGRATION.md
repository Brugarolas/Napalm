# Napalm Reloaded: post-bytecode optimizer

## Files

- `napalm_bytecode_optimizer.rs`: new optimization pipeline.
- `napalm_bytecodecompiler.rs`: modified compiler with automatic invocation **after** `finalize_deopts` and metadata index rebuilding, **before** final bytecode verification.
- `napalm_bytecode.rs`: the supplied opcode format, unchanged in this increment. No interpreter changes are required for the safe optimizer.

Add this declaration in the crate root (`src/lib.rs` or `src/main.rs`, whichever declares the existing Napalm modules):

```rust
mod napalm_bytecode_optimizer;
```

Keep the existing declarations of `napalm_bytecode` and `napalm_bytecodecompiler`.

## CompileOptions

- `optimize_bytecode` (default **true**): run the pipeline automatically.
- `verify_each_bytecode_pass` (default: debug builds): verify after each pass; a final verification always runs.
- `verify_bytecode` (default **true**): the compiler's final verification step.

Compilation stats now expose `bytecode_words_before_optimization`, `bytecode_words_removed`, `bytecode_passes_changed`, and `superinstruction_candidates_detected`.

## Optimizations implemented and enabled

- Identity move/swap elimination.
- Normalization of integer, boolean and null constant-pool loads; compress small signed integers into `LoadI8`.
- Elimination of directly overwritten side-effect-free loads and copies.
- Cancellation of consecutive identical swaps and duplicate moves.
- Pure same-register integer identities and comparisons.
- Immediate-copy propagation without introducing new JIT-live source registers.
- Integer algebraic identities for additions, subtractions, multiplications, bitwise operations and shifts.
- Literal folding for arithmetic and integer comparisons, with checked arithmetic and safe shift-count guards.
- **Cross-block** constant folding and one-sided algebraic identities from the compiler's verified SSA liveness/constant-fact metadata.
- Elimination of equivalent consecutive guards (same operands, descriptor and deoptimization ID).
- Constant-condition branch selection; jump threading; conditional branch inversion over an unconditional jump; removal of jumps to the next instruction.
- Bytecode compaction and relocation of relative branches, switch targets, GC safepoints, live-register side-exit maps, deoptimization resume PCs, source ranges, exception ranges, and inlined frame resume PCs located in **other** bytecode functions.
- Atomic publication: the original program is unchanged when an optimizer pass or final verifier returns an error.

## Superinstructions and complex operations

`superinstruction_candidates()` enumerates 30 fusion/runtime candidates, and `find_superinstruction_sites()` detects concrete sites (including GC and control-flow entry hazards) for the bytecode compiler statistics. The catalog includes immediate arithmetic, return fusion, bulk object construction, typed structural objects, interfaces/types/class operations, map/filter/reduce, direct-call returns, and more.

**These are candidate specifications, not implemented VM instructions.** The submitted files do not contain the interpreter's opcode dispatch, runtime class/interface registration machinery, collection method implementation, GC/barrier handlers or tracing-JIT bytecode frontend. Emitting a new numerical opcode without updating those sites would make Napalm crash, reject bytecode, or execute it incorrectly. Therefore this iteration deliberately **does not** emit unsupported opcodes. The existing complex opcodes `CreateClass`, `BindInterface`, `AllocInstance`, `AllocMap`, `MapSet`, `StoreProperty`, `Invoke`, and `InvokeSpread` remain available and optimizable.

Next integration requires the interpreter and relevant runtime/JIT modules to define and enable executable fused opcodes and their precise GC/deopt state semantics.

## Validation

The compiler tests can be run as usual with the new module declaration. No standalone `cargo test` run was possible in this working environment because the full Napalm crate and Rust toolchain were not supplied; only the three listed Rust source files were available.

# Build artifacts (`.burning`)

```sh
napalm --build path/to/fib.napalm
```

The build pipeline resolves and parses the entire static module graph, installs
the builtin declarations, and lowers to HIR. Its separate
`napalm_build_pipeline::BuildCompilationPipeline` runs the normal HIR passes plus
the AOT-only passes described below, and calls the complete HIR optimizer
repeatedly until the HIR before and after a round is identical. Each round includes function/node
passes and whole-program finalizers, so global rewrites can expose opportunities
for the next round. There is no fixed iteration limit. The standalone
`optimize_hir` phase compares canonical HIR snapshots (including metadata and
exact numeric constant bits), independently of the passes' change flags. It
neither generates nor inspects bytecode; it returns with the HIR stabilized.

Once HIR is stable, the pipeline lowers it to register bytecode and runs the
normal bytecode optimizer. Bytecode generation happens once, after convergence.
The build pipeline has its own pass selection and orchestration so it can evolve
independently; ordinary source execution keeps its existing single HIR round,
and `.burning` execution performs no HIR optimization. Its final stage writes
`path/to/fib.burning`, replacing an existing artifact on a successful build.
No VM is initialized and no module body, import initializer or user code runs.

The output lives beside the entry source, with its extension replaced by
`.burning`. Imported modules are included in the same artifact; they do not get
separate output files. `--build` precedes the source path. Arguments after the
source path keep their existing `Process.args` meaning in normal execution.

Compilation errors retain exit code 1, usage errors 2, and file I/O errors 3.
Success returns 0. Serialization finishes before the output file is opened, so
compilation or serialization failures leave an existing artifact untouched.

## AOT-only HIR passes

These live in `src/napalm_ir/aot/` and are installed only by
`napalm_build_pipeline::optimization_passes`. They assume a closed program: the
artifact contains every module that can run. Budgets bound code growth and
compile-time evaluation per round; they never cap the outer fixed point.

| Pass | What it does |
| --- | --- |
| GVN (`Gvn`) | Dominator-scoped value numbering after copy propagation. |
| PRE (`Pre`) | At joins whose predecessors jump unconditionally, reuses an expression available on some paths and inserts it on the others; the join receives it as a parameter. |
| LICM (`Licm`) | Hoists pure, non-trapping computations out of natural loops into a dedicated preheader. Constants and copies stay put: hoisting them only lengthens live ranges and adds loop-carried state to JIT traces. |
| Escape analysis (`ScalarReplacement`) | The existing PEA/SROA, rerun after the AOT passes; inlining makes it interprocedural. |
| IPCP (`Ipcp`) | Binds parameters to constants when every direct call agrees and the function is never address-taken (no closure, method binding or worker binding). |
| SCCP (`Sccp`) | Sparse conditional constant propagation, including functions with `try`/`catch`. Folds builtin operators (`+ - * / % < <= > >= == != -x !x`) on Number/Bool/Null constants. |
| Partial evaluation (`PartialEvaluation`) | Evaluates direct calls with constant arguments in a scalar HIR interpreter (recursion and loops included, bounded by fuel) and replaces them with the result. |
| Specialization (`Specialization`) | Clones a callee per call site for constant arguments and for narrower object types (records, interfaces, class instances), so static type specialization can devirtualize inside the clone. A constant-only clone is kept only if SCCP on it removes work; an object fact only if the parameter is dispatched on, accessed or contract-checked. Unprofitable clones would split JIT warm-up and profiles. Clones keep the source name; `Function::specialization` records their origin and facts. |
| Inlining (`Inlining`) | Copies straight-line helpers in place, then inlines small closed callees with their whole CFG (branches, tail calls, `throw`). Callees containing loops are not inlined: the loop amortizes the call, and moving it into the caller (often a module body) costs the JIT its function entry and bloats trace state. |
| Automatic memoization (`MonotoneAutoMemoization`) | Runs before the AOT interprocedural passes. A classification is never withdrawn in a later round, even if the reshaped program defeats the whole-program proof (e.g. after literal hoisting). Hoisted shared literals count as the shareable allocation they replaced. |
| Tree shaking (`TreeShaking`) | Removes unused module-variable reads, unobserved closure bindings and functions that are no longer reachable. Function IDs are never renumbered. |

Shared safety rules:

- Builtin operators are folded only for concrete scalar operands. Builtin method
  tables are bound from the build's own manifest at VM setup and user code
  cannot rebind them. NaN results are left to run time.
- Contract checks (`BindInterface`) are never removed. When the checked value is
  a known scalar that satisfies the contract, users of the result read the
  constant.
- Recursive functions (any direct-call cycle; a clone counts as its origin) are
  neither inlined nor specialized. Inlining them would unroll one level per round.
- Memoization has priority. No pass clones or inlines a memoized or `computed`
  function, since a copy would bypass its memo table or reactive validation.
- Inlining keeps memoized/`computed` functions behind their call boundary. It
  rejects fiber switches, awaits and deopt-based speculation, and functions that
  use class field slots of another class or module-relative primitives from
  another module. Inlined blocks join every exception region that protects the
  call. Argument types are narrowed to the parameter types with a `Move`, which
  matches the unchecked call boundary. The callee's own entry contract checks are
  copied with its body.
- Error messages keep the source location of the inlined code.

## Execute an artifact

```sh
napalm path/to/fib.burning [program arguments...]
```

The CLI recognizes the `.burning` extension, reads and validates the artifact,
and executes its stored entry point. It skips source resolution, parsing,
lowering and compilation. The entry source and statically imported source files
are no longer needed; the artifact can be moved to another directory.

Execution uses the same VM initialization, builtins, GC, JIT and diagnostics as
source execution. `Process.args` contains the arguments following the artifact
path, and relative runtime file accesses use the process working directory as
usual. Native libraries and other files opened by the program are still runtime
dependencies.

Unreadable artifacts return exit code 3; invalid or incompatible artifacts return
1; runtime errors return 4. `--build file.burning` is a usage error (2) and leaves
the artifact untouched. `napalm file.burning --build` passes `--build` to the
program as an ordinary argument.

## Version 1 framing

All binary integers are unsigned and little-endian. Lengths count bytes unless
specified otherwise. The file starts with this 16-byte header:

| Offset | Size | Meaning |
| --- | --- | --- |
| 0 | 8 | Magic bytes `NAPBURN\0` (one final NUL byte) |
| 8 | 4 | Format version, currently `1` |
| 12 | 4 | Section count, currently `3` |

Each section consists of a four-byte ASCII tag, a `u64` payload length, and exactly
that many payload bytes. Writers emit `CODE`, `META`, then `BOOT`. Readers identify
sections by tag. A tool can extract or skip any section without decoding others.

| Section | Encoding | Contents |
| --- | --- | --- |
| `CODE` | Binary | Optimized instruction streams for all functions, including builtin bytecode bodies |
| `META` | UTF-8 JSON | `BytecodeProgram`, with instruction words and derived lookup indexes omitted |
| `BOOT` | UTF-8 JSON | Entry `FuncId`, module name, and complete `BuiltinManifest` |

`CODE` starts with a `u32` function count. Each function then has a `u32` function
ID, a `u64` instruction **word** count, and that many `u32` words. Extension words
are included unchanged. Function order and IDs match `META.functions`.

`META` retains every other function/chunk field: constant pools, frame layouts,
parameter and capture bindings, register kinds, return types, inline-cache
descriptors, specializations, type/class guards, switch tables, deoptimization
snapshots, GC safepoints and reference bitmaps, register liveness/types/constants,
resume points, source maps, profile descriptors, and exception handlers. Program
metadata retains strings, methods, module slots/exports/builtin bindings/worker
bindings, classes, foreign function declarations, structural layouts and named
contracts. All references use logical IDs, never process addresses.

Serde's named enum representation is used in JSON. `Constant::Num` stores a `u64`
IEEE-754 bit pattern, for example `{"Num":4631107791820423168}` for `42.0`. This
preserves signed zero, infinities, subnormals and NaN payloads exactly.

`BOOT.builtins` stores the complete builtin method/primitive/native mapping and
helper function IDs, including decorator wrappers. Native implementations are
represented by enum identities, not native machine code or function pointers.
The corresponding implementations are supplied by the runtime.

## Reconstruction and compatibility

`napalm_burning::decode` reconstructs a `BytecodeProgram` and `Bootstrap`, rebuilds
lookup indexes, checks the entry point, and runs the bytecode verifier. It rejects
unknown versions, malformed framing, duplicate/unknown sections, truncated code,
inconsistent function IDs/counts, and trailing data. The CLI validates the
artifact before initializing the VM.

A runtime can initialize the decoded program with `Interpreter::with_host`, using
`BuiltinHost::new(bootstrap.builtins.clone())`, install the VM builtins, and call
`run_module(bootstrap.entry)`. These are the same initialization steps as normal
source execution. Tests exercise reconstruction and execution, including imports,
classes, structural objects, closures, exception handlers, native calls and GC.

Runtime shapes and caches, prepared instruction dispatch/fusion, inherited field
relocations, heap objects, GC state and JIT machine code are created by normal VM
initialization/execution. They contain process-local state and are not compiler
output. The artifact stores the layouts, descriptors and GC/register maps needed
to recreate them; it is not a snapshot of an executing process.

Version 1 is tied to the current bytecode/runtime schema. Changes to opcode
encoding, metadata semantics, builtin identities or incompatible serialized
fields must bump `FORMAT_VERSION`. Cross-version runtime compatibility is not
implied by the JSON encoding.

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

## Standalone executables (`--compile`)

```sh
napalm --compile path/to/fib.napalm   # writes fib.burning and fib.exe (Windows) / fib
napalm --compile path/to/fib.burning  # embeds an existing artifact as is
./fib [program arguments...]
```

`--compile` runs the same build pipeline as `--build` (so it also writes
`fib.burning`) and then writes an executable beside the entry file, with the
extension replaced by the platform's executable extension. The executable is a
copy of the running Napalm runtime with the artifact appended, so it contains
the VM, GC, builtins, Cranelift and every statically linked Rust dependency. On
Windows the MSVC CRT is linked statically (`.cargo/config.toml`), so no Visual
C++ Redistributable is needed.

When run, it executes the embedded artifact with the same validation, VM
initialization, JIT and exit codes as `napalm fib.burning`. It parses no
command-line options: every argument is a program argument, and `Process.args`
contains them all. The `NAPALM_*` diagnostics environment variables keep
working. Native libraries and files opened by the program are still runtime
dependencies.

The `.burning` path is validated before anything is written. The executable is
assembled under a temporary name and renamed into place, so a failed compile
leaves an existing executable untouched. Exit codes match `--build`.

### LLVM

On Windows, `LLVM-C.dll` is delay-loaded (see `build.rs`), so the executable
embeds the copy this runtime would load (`NAPALM_LLVM_DLL`, then the one next
to `napalm.exe`, then the build-time LLVM prefix), compressed with zstd. It is
only extracted when the LLVM trace tier first starts, to
`%LOCALAPPDATA%\napalm\llvm\<content hash>\LLVM-C.dll` (`NAPALM_CACHE_DIR`
overrides `%LOCALAPPDATA%\napalm`). Later runs, and other executables built
with the same DLL, reuse it. Programs that never get hot enough never extract
it. Extraction runs on the background LLVM compiler thread. If the program ends
during an extraction, the process waits for it to finish before exiting (about
a second, once per machine) instead of abandoning it. Partial files left by
killed processes are removed by the next extraction. If the DLL cannot be found at compile time, `--compile` warns and the
executable falls back to the normal runtime lookup (Cranelift only if absent).

Builds without the `llvm-jit` feature embed nothing. On Linux and macOS libLLVM
is an ordinary dynamic dependency resolved before `main`, so it is not embedded:
it must be installed or placed next to the executable (`rpath` `$ORIGIN` /
`@executable_path`). On macOS the appended payload invalidates the code
signature; that platform is untested.

### Payload layout

```text
[runtime image][.burning artifact][LLVM library, zstd frame (optional)][trailer]
```

Operating systems ignore data appended after an executable image. At startup
the runtime reads the last 64 bytes of its own file; without the magic it
behaves as the normal CLI. Fields are little-endian:

| Offset | Size | Meaning |
| --- | --- | --- |
| 0 | 8 | Runtime image length (the artifact starts here) |
| 8 | 8 | Artifact length |
| 16 | 8 | Compressed LLVM library length, `0` if absent |
| 24 | 8 | Uncompressed LLVM library length |
| 32 | 8 | FNV-1a 64 hash of the uncompressed library (cache directory name) |
| 40 | 4 | Payload version, currently `1` |
| 44 | 4 | Reserved, `0` |
| 48 | 16 | Magic `NAPALM-STANDALN\0` |

The lengths must add up to the file size. Otherwise, or with another payload
version, the executable reports `invalid standalone executable` (exit 1). The
artifact itself is decoded and verified as usual. Signing a standalone
executable (Authenticode) appends data after the trailer and is not supported.

## Version 2 framing

All fixed-width integers are unsigned and little-endian. Lengths count bytes
unless specified otherwise. The file starts with this uncompressed 24-byte header:

| Offset | Size | Meaning |
| --- | --- | --- |
| 0 | 8 | Magic bytes `NAPBURN\0` (one final NUL byte) |
| 8 | 4 | Format version, currently `2` |
| 12 | 4 | Payload compression, currently `1` (zstd) |
| 16 | 8 | Uncompressed payload length |

The rest of the file is exactly one zstd frame (level 18, with content size and
checksum). The level only affects build time; decompression costs well under a
millisecond for typical programs. Readers reject declared payloads above 1 GiB
before allocating. They also reject frames that inflate to a different size or
fail their checksum, and any trailing data.

The decompressed payload starts with a `u32` section count, currently `3`. Each
section consists of a four-byte ASCII tag, a `u64` length, and exactly that many
bytes. Writers emit `CODE`, `META`, then `BOOT`. Readers identify sections by
tag. After decompression, a tool can extract or skip any section without
decoding the others.

| Section | Encoding | Contents |
| --- | --- | --- |
| `CODE` | Binary | Optimized instruction streams for all functions, including builtin bytecode bodies |
| `META` | [postcard](https://docs.rs/postcard) | `BytecodeProgram`, with instruction words and derived lookup indexes omitted |
| `BOOT` | postcard | Entry `FuncId`, module name, and complete `BuiltinManifest` |

Every section is binary. Postcard is not self-describing: `META` and `BOOT`
are only readable with the runtime's own type definitions, and a section must be
consumed exactly.

Format 1 stored `META` and `BOOT` as JSON without compression. The current
runtime rejects it with a message asking to rebuild. Compared with format 1,
`index.napalm` went from 1197 KB to 34 KB and `basic.napalm` from 167 KB to 9 KB.
Loading (decompression, decoding, index rebuild and verification) also takes
less time than parsing format 1's JSON alone.

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

`Constant::Num` stores its `u64` IEEE-754 bit pattern rather than a float. This
preserves signed zero, infinities, subnormals and NaN payloads exactly,
independently of the serializer.

`BOOT.builtins` stores the complete builtin method/primitive/native mapping and
helper function IDs, including decorator wrappers. Native implementations are
represented by enum identities, not native machine code or function pointers.
The corresponding implementations are supplied by the runtime.

## Reconstruction and compatibility

`napalm_burning::decode` reconstructs a `BytecodeProgram` and `Bootstrap`, rebuilds
lookup indexes, checks the entry point, and runs the bytecode verifier. It rejects
unknown versions and compressions, corrupt or mis-sized frames, malformed
framing, duplicate/unknown sections, truncated code, inconsistent function
IDs/counts, and trailing data. The CLI validates the artifact before
initializing the VM.

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

Version 2 is tied to the current bytecode/runtime schema. Changes to opcode
encoding, metadata semantics or builtin identities must bump `FORMAT_VERSION`.
Postcard encodes fields by position, so any change to a serialized struct or
enum (adding, removing or reordering fields or variants) is also a format break
and must bump it. Cross-version runtime compatibility is not implied.

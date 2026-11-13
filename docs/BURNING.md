# Build artifacts (`.burning`)

```sh
napalm --build path/to/fib.napalm
```

The build pipeline resolves and parses the entire static module graph, installs
the builtin declarations, lowers to HIR, runs the normal HIR optimization passes,
lowers to register bytecode, and runs the normal bytecode optimizer. It shares
the compilation implementation with normal execution. Its final stage writes
`path/to/fib.burning`, replacing an existing artifact on a successful build.
No VM is initialized and no module body, import initializer or user code runs.

The output lives beside the entry source, with its extension replaced by
`.burning`. Imported modules are included in the same artifact; they do not get
separate output files. `--build` precedes the source path. Arguments after the
source path keep their existing `Process.args` meaning in normal execution.

Compilation errors retain exit code 1, usage errors 2, and file I/O errors 3.
Success returns 0. Serialization finishes before the output file is opened, so
compilation or serialization failures leave an existing artifact untouched.

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

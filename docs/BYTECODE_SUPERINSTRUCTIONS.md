# Napalm Reloaded — Bytecode specialization contract

## Compatible-by-default compilation

`CompileOptions::default()` keeps `emit_specialized_bytecode = false`. Normal
optimizations in `napalm_bytecode_optimizer.rs` still run, as before. To test
specialized generation in a development tool, set
`options.emit_specialized_bytecode = true`; do **not** execute that result in the
current interpreter until the native handlers and tracing JIT lowering exist.

The code format remains a stream of `u32` words. Opcode IDs 0–134 are unchanged;
IDs 135–163 are reserved for the original 29 new operations, and IDs 164–179 for the 16 surviving fused operations. Each specialized descriptor is
stored in `BytecodeFunction.chunk.specializations` and referenced by `x0`, except
`NewObjectEmpty` and `AwaitPromise` (one-word operations without a descriptor).

The `bytecode` module exposes `Assembler::emit_specialized` for descriptor-backed
fixed and packed-register instructions and `Assembler::emit_specialized_shape`
for three-word shape field instructions. Both return `AssembleError` on a wrong
descriptor family. The verifier checks descriptor indices, compatible opcode
families, metadata, register bounds, call arity and GC safepoint presence.

## Operand conventions

| Group | Encoded operands | Descriptor | Notes |
|---|---|---|---|
| `DefineClass`, `DefineClassWithImplements` | A=result class; B=C=0; x0=descriptor | `Class` | Atomic create/register; all methods provided in descriptor |
| `DefineDerivedClass`, `DefineDerivedClassWithImplements` | A=result class; B=runtime superclass; C=1; x0=descriptor | `Class` | Resolve parent, relocated fields, inherited contracts |
| `DefineInterface`, `DefineType` | A=result; x0=descriptor | `Contract { name: Some }` | Registration must be idempotent with bootstrap metadata |
| `ImplementInterface`, `ImplementType` | A=checked result; B=value; x0=descriptor | `Contract` | Permanent checked contract installation |
| `DefineArrayWithGenerics` | A=list; B=capacity if present, else 0; x0=descriptor | `ArrayType {element,with_capacity}` | Preserve typed element constraints |
| `NewClassInstance` | A=instance; B=reserved; C=arg count; x0=descriptor then packed arguments | `Instance` | Allocate then invoke constructor, return instance |
| `NewInterfaceObject` | A=object; C=value count; x0=descriptor then packed values | `ObjectShape` | Constructor must enforce contract |
| `NewObjectShape` | A=object; x0=descriptor | `ObjectShape` | Preconfigured shape; no silent contract installation |
| `NewObjectEmpty` | A=object | None | Identical visible result type to `AllocMap` |
| `NewObjectWithProperties` | A=object; C=value count; x0=descriptor then packed values | `ObjectProperties` | Property names in descriptor, values in ordered GPRs |
| `GetFieldShape` | A=result; B=object; C=key; x0=descriptor; x1=slot | `FieldShape` | Guard shape; preserve generic lookup fallback |
| `SetFieldShape` | A=object; B=key; C=value; x0=descriptor; x1=slot | `FieldShape` | Guard shape, permanent contract and write barriers |
| `GetFieldClass` | A=result; B=object; x0=descriptor | `FieldClass` | Class-local offset relocated at link |
| `SetFieldClass` | A=object; B=value; x0=descriptor | `FieldClass` | Class-local offset relocated at link |
| `InvokeMethodDirect` | A=result; B=receiver; C=arg count; x0=descriptor + packed arguments | `DirectMethod` | Verify method identity before direct call; fallback IC |
| `ArrayMapPacked`, `ArrayFilterPacked`, `ArrayForEachPacked` | A=result; B=array; C=arg count; x0=descriptor + packed arguments | `PackedArray` | Validate builtin identity/storage, retain fallback IC |
| `ArrayReduceNumeric` | A=result; B=array; C=arg count; x0=descriptor + packed args | `PackedArray` with reducer proof | Do not assume arbitrary reducer is sum |
| `ArrayFilterReducePacket`, `ArrayMapReducePacked` | A=result; B=array; C=arg count; x0=descriptor + packed args | `PackedArray` with second IC | Both callbacks and fallback semantics mandatory |
| `ApplyDecoratorToFunction`, `ApplyDecoratorToClass` | A=decorated result; B=input; x0=descriptor | `Decorator` | Preserve identity, order, exceptions |
| `AwaitPromise` | A=settled result; B=promise | None | Can suspend and schedule a microtask |
| `LaunchConcurrentTask` | A=Promise result; B=closure; C=argc; x0=descriptor + packed args | `Concurrent` | Must use isolation/structured-copy boundary |

## Emission already wired in compiler (experimental flag)

- Class definitions: `ClassFusionPlan` consumes adjacent `BindMethod` and
  `BindForeignMethod` nodes (with no external deopt resume entry) and produces
  a single class-definition descriptor. The lowerer now binds methods before
  publishing the class object to its module binding.
- Constructor factories: `InstanceFusionPlan` consumes adjacent
  `AllocInstance` + known non-async `CallDirect` constructor when the ignored
  call result is provably unused; no intermediate escaping operation is crossed.
- Contracts: structural `BindInterface` with `Type::Interface` or `Type::Record`
  produces `ImplementInterface` or `ImplementType`.
- Arrays: statically typed `List<T>` allocations use `DefineArrayWithGenerics`;
  calls to `map`, `filter`, and `forEach` on statically typed `List<T>` use packed
  opcodes but must retain original dynamic dispatch as fallback.
- Objects/fields: `AllocMap` becomes `NewObjectEmpty`; class-local and structural
  field accesses become the corresponding known class/shape operations.
- Direct methods: monomorphic call-site profiles with known receiver class
  and direct target produce `InvokeMethodDirect`, retaining runtime method
  identity checks and the original inline-cache fallback.
- Concurrency: statically known concurrent closure identities use
  `LaunchConcurrentTask`; statically known Promise values use `AwaitPromise`.

No optimizer automatically emits speculative numeric reduction or fused
map/filter+reduce: it lacks a proven callback and complete array-observability
proof. Named `DefineInterface`/`DefineType` are also not emitted: declarations
are currently registered in `RuntimeMetadata.contract_names` during bootstrap,
not as HIR statements. `TODO.md` tracks the remaining emission paths.

## Build/integration

Place `napalm_bytecode.rs`, `napalm_bytecodecompiler.rs`, and
`napalm_lowerer.rs` under `src/`. Include `napalm_bytecode_optimizer.rs` if it
is not already present. Ensure `src/main.rs` or `src/lib.rs` declares the
`napalm_bytecode_optimizer` module before importing the compiler. The frontend,
HIR, and pipeline files supplied with this request already expose the required
metadata and do not require replacement for this phase.

Run `cargo fmt --all --check`, `cargo test --lib` (or your relevant executable
test target), and bytecode/verifier tests. Test the existing runtime with the
default flag first. Enable experimental emission only when the interpreter
has the corresponding handlers. This workspace does not include the complete
Cargo project or a Rust compiler; these commands have not been executed here.


## Phase 2 — deduplicated bytecode fusions (IDs 164–179)

The 14 redundant `SuperinstructionCandidate` entries are removed, not assigned
new opcode numbers. See `TODO.md` for their mapping to IDs 135–163 and the
16 remaining IDs. The experimental post-bytecode pipeline is
`superinstruction_emission_passes()` (`FuseSurvivorSuperinstructions` then
`CompactBytecode`), called after `finalize_deopts` and the normal bytecode
optimization pipeline when `emit_specialized_bytecode=true`.

All 16 opcodes have explicit formats, assembler/decoder/descriptors, and
verifier support. **15 have guarded automatic fusion patterns**; the 16th,
`MapLookupOrInsert`, has a dedicated assembler entry but still requires a new
HIR source construct to safely express presence-based insertion. Do not rewrite
`MapGet + MapSet` to it: the semantics differ.

For an operation with two native stages, `SpecializedDescriptor::Fusion`
contains `operation` plus `FusionContinuation`. The original PC's safepoint
remains the pre-stage map, while the second stage's references/liveness and
static constants/types travel in `FusionContinuation`. The interpreter's
handler must apply the correct stage-specific root map and exception behavior,
and any later tracing/JIT lowering must expose equivalent internal safepoints,
resume states, and side exits. The fuser rejects sequences with a branch,
OSR, exception-region boundary or deopt resume at the removed PC and requires
intermediate registers to be unobservable. The compacting phase relocates
all PC-relative tables atomically.

### Relevant opcode encodings

- `LoadI8IntegerArithmetic` / `LoadI8IntegerComparison`:
  A=output, B=other operand, C=signed i8, descriptor=exact integer opcode
  and operand order. The temporary literal register must be overwritten.
- `LoadI8Return`: A=signed i8, descriptor=return literal. `LoadLiteralReturn`:
  A=0 null, 1 false, 2 true. Both are terminators.
- `AllocListAndAppend`: A=new list, B=first value, descriptor stores the
  optional element type and capacity-register dependency.
- `AllocInstanceAndInitializeField`: A=instance, B=initial field value;
  descriptor=class/local slot. This operation does **not** run a constructor.
- `AllocInstanceAndInitializeProperty`: A=instance, B=value, C=key;
  descriptor=class, layout, property slot.
- `LoadFieldAndCall` / `LoadPropertyAndCall`: A=return destination,
  B=source object, C=call argc; descriptor carries exact field/layout,
  property key register as applicable, and call convention; packed args follow.
- `GuardThenFieldLoad` / `GuardThenPropertyLoad`: A=loaded value,
  B=guard source, optional C=property key; descriptor=type/class guard ID,
  deopt ID and field/layout slot.
- `MapLookupOrInsert`: A=result, B=map, C=key; descriptor=default value reg.
  Lookup MUST discriminate key absence from existing null value.
- `ImportAndReadVariable`: A=result, descriptor=module and variable IDs.
- `DirectCallAndReturn` / `ClosureCallAndReturn`: A=call destination,
  C=argc, optional B=closure; descriptor=call convention and target where
  applicable, with packed args; both are terminators.
- `ConcurrentLaunchAndAwait`: A=settled value, B=closure, C=argc,
  descriptor=optional known concurrent FuncId and second-stage GC state;
  packed args follow. The native handler must preserve promise scheduling,
  copying and the await suspension/resume protocol.

The previous baseline optimizer still runs by default and never emits an
unrecognized opcode. **The native interpreter and Cranelift/LLVM JIT handlers
are not included**; these binaries are NOT runnable with the current VM when
the experimental flag is set.

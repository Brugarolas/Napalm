# Napalm Reloaded — Bytecode specialization contract

## Compilation

`CompileOptions::default()` sets `emit_specialized_bytecode = true`. Setting it
to `false` produces the equivalent generic bytecode (used by differential tests).

The code format remains a stream of `u32` words. Opcode IDs 0–134 are the
generic set; IDs 135–153 are the specialized opcodes. Each specialized
descriptor is stored in `BytecodeFunction.chunk.specializations` and referenced
by `x0`, except `NewObjectEmpty` (one word, no descriptor) and
`MapLookupOrInsert` (`x0` = value GPR).

`Assembler::emit_specialized` emits descriptor-backed fixed and packed-register
instructions; `Assembler::emit_specialized_shape` emits the three-word shape
field instructions. The verifier checks descriptor indices, opcode/descriptor
families, metadata, register bounds, call arity and GC safepoint presence.

## Operand conventions

| Group | Encoded operands | Descriptor | Notes |
|---|---|---|---|
| `DefineClass`, `DefineClassWithImplements` | A=result class; B=C=0; x0 | `Class` | Atomic create/register with the complete method table |
| `DefineDerivedClass`, `DefineDerivedClassWithImplements` | A=result class; B=runtime superclass; C=1; x0 | `Class` | Resolve parent, relocated fields, inherited contracts |
| `DefineInterface`, `DefineType` | A=result; x0 | `Contract { name: Some }` | Verifies canonical bootstrap metadata |
| `ImplementInterfaceOrType` | A=checked result; B=value; x0 | `Contract` | Permanent structural contract |
| `MapLookupOrInsert` | A=result; B=Map/Set; C=key; x0=value GPR | — | Returns the existing value, or inserts and returns the default; Sets use the key as their value |
| `DefineArrayWithGenerics` | A=list; B=capacity if present, else 0; x0 | `ArrayType` | Only replaces an explicit `List<T>` binding |
| `NewClassInstance` | A=instance; C=arg count; x0 then packed args | `Instance` | Prepared: Y=class slot, CACHE=constructor index; constructor result ignored |
| `NewInterfaceOrTypeObject` | A=object; C=value count; x0 then packed values | `ObjectShape` | Installs the contract before publication |
| `NewObjectShape` | A=object; x0 | `ObjectShape` | Null-initialized layout |
| `NewObjectEmpty` | A=object | — | Same result as `AllocMap` |
| `NewObjectWithProperties` | A=object; C=value count; x0 then packed values | `ObjectProperties` | Names in descriptor, values in ordered GPRs |
| `GetFieldShape` / `SetFieldShape` | as `LoadProperty` / `StoreProperty`; x0; x1=slot | `FieldShape` | Prepared to the generic layout/slot operands |
| `GetFieldClass` / `SetFieldClass` | as `LoadField` / `StoreField`; x0 | `FieldClass` | Class-local slot relocated at link |
| `InvokeMethodDirect` | A=result; B=receiver; C=arg count; x0 then packed args | `DirectMethod` | Method identity guarded; IC fallback |

## Tier rules

- `MapLookupOrInsert` uses a GC-safe runtime call in Cranelift/LLVM traces and
  the whole-function JIT. Shared operands resume at the same interpreter PC
  before mutation, so the shared dispatch loop acquires the required locks.
  ConcurrentMap uses atomic get-or-insert; ConcurrentSet preserves the winning
  element and retries if a concurrent deletion races with its lookup.
- Tracing JIT: `decode_at` rewrites `Get/SetFieldClass`, `Get/SetFieldShape` and
  `InvokeMethodDirect` to their canonical opcodes, so the recorder and both
  inliners treat them exactly like the generic encodings. Guards still replay the
  specialized PC in the interpreter. LLVM lowers the same Trace IR.
- Object factories use the native Immix TLAB protocol (side-effect-free probe,
  reservation in generated code, non-safepoint commit) in both trace backends and
  in the whole-function JIT; `NewObjectEmpty` uses a plain runtime leaf in traces.

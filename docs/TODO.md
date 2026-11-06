# TODO — Napalm Reloaded bytecode specializations

> Estado a 7 de noviembre de 2026. La emisión de super-instrucciones está activa
> por defecto (`CompileOptions::emit_specialized_bytecode = true`).

## Opcodes especializados vigentes (`135..=153`)

| Opcode (ID) | Emisión | Intérprete | Tracing JIT (Cranelift + LLVM) | Whole-function JIT |
|---|---|:---:|:---:|:---:|
| `DefineClass` / `DefineDerivedClass` / `DefineClassWithImplements` / `DefineDerivedClassWithImplements` (135, 138–140) | `CreateClass` + todos sus `Bind*Method` contiguos | ✔ | runtime | runtime |
| `DefineInterface` / `DefineType` (136–137) | declaraciones con nombre (prólogo) | ✔ | no-op | runtime |
| `ImplementInterfaceOrType` (141) | `BindInterface` estructural | ✔ | runtime GC leaf | runtime |
| `MapLookupOrInsert` (142) | **ninguna** (se implementa con el merge de master) | ✗ | ✗ | ✗ |
| `DefineArrayWithGenerics` (143) | `AllocList` seguido de `BindInterface(List<T>)` (solo appends/constantes entre medias) | ✔ | factoría nativa | TLAB nativo (capacidad 0) |
| `NewClassInstance` (144) | `AllocInstance` + `CallDirect` contiguo al constructor (no async/concurrent/memoizado) | ✔ | factoría nativa + llamada | TLAB nativo (solo clases sin campos) |
| `NewInterfaceOrTypeObject` (145) | literal con claves constantes + `BindInterface` exacto | ✔ | factoría nativa | TLAB nativo (≤4 campos) |
| `NewObjectShape` (146) | literal de solo `null` con layout conocido | ✔ | factoría nativa | TLAB nativo |
| `NewObjectEmpty` (147) | `AllocMap` | ✔ | runtime leaf (como `AllocMap`) | TLAB nativo |
| `NewObjectWithProperties` (148) | `AllocMap` + `MapSet`/`StoreProperty` contiguos con clave constante | ✔ | factoría nativa | TLAB nativo (≤4 campos) |
| `GetFieldShape` / `SetFieldShape` (149–150) | todo `LoadProperty` / `StoreProperty` | ✔ (alias) | decodificado como `LoadProperty`/`StoreProperty` | PIC nativo |
| `GetFieldClass` / `SetFieldClass` (151–152) | todo `LoadField` / `StoreField` | ✔ (alias) | decodificado como `LoadField`/`StoreField` | carga/almacén nativo |
| `InvokeMethodDirect` (153) | `Invoke` con perfil monomórfico (≥32 muestras) | ✔ | decodificado como `Invoke` | llamada directa con guarda de clase |

Los opcodes 154–179 (fusiones de arrays, decoradores, `AwaitPromise`,
`LaunchConcurrentTask` y las 16 fusiones post-bytecode) se eliminaron: nunca
tuvieron handler y el intérprete ya fusiona en preparación (`install_superinstructions`).

## Invariantes que hay que mantener

- El tracing JIT normaliza `Get/SetField*` e `InvokeMethodDirect` en `decode_at`;
  así grabador e inliners solo ven la forma canónica (si no, se pierde el inlining).
- `DefineArrayWithGenerics` nunca debe salir del tipo **inferido** de un literal:
  `const xs: any = [1]` no tiene contrato.
- Los literales estructurales tipados se bajan como `AllocMap` + `StoreProperty`
  (no `MapSet`): el escape analysis de WIR solo hace scalar replacement así.
- Los constructores nunca se memoizan (análisis + verificador); `NewClassInstance`
  apila el frame del constructor sin sonda de memo.

## Pendiente

- [ ] `MapLookupOrInsert`: runtime + JITs (merge con master).
- [ ] `InvokeMethodDirect` no se emite en la práctica: nadie rellena `profiles.call_sites`.
- [ ] `Get/SetFieldClass` y `Get/SetFieldShape` son alias 1:1 sin ganancia medida.
- [ ] Factoría nativa de `NewClassInstance` para clases **con** campos (hoy solo campos = 0).

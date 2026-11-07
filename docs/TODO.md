# TODO — Napalm Reloaded bytecode specializations

> Estado a 7 de noviembre de 2026. La emisión de super-instrucciones está activa
> por defecto (`CompileOptions::emit_specialized_bytecode = true`).

## Opcodes especializados vigentes (`135..=153`)

| Opcode (ID) | Emisión | Intérprete | Tracing JIT (Cranelift + LLVM) | Whole-function JIT |
|---|---|:---:|:---:|:---:|
| `DefineClass` / `DefineDerivedClass` / `DefineClassWithImplements` / `DefineDerivedClassWithImplements` (135, 138–140) | `CreateClass` + todos sus `Bind*Method` contiguos | ✔ | runtime | runtime |
| `DefineInterface` / `DefineType` (136–137) | declaraciones con nombre (prólogo) | ✔ | no-op | runtime |
| `ImplementInterfaceOrType` (141) | `BindInterface` estructural | ✔ | no-op si el contrato ya está demostrado; si no, runtime GC leaf | runtime |
| `MapLookupOrInsert` (142) | **ninguna** (se implementa con el merge de master) | ✗ | ✗ | ✗ |
| `DefineArrayWithGenerics` (143) | `AllocList` seguido de `BindInterface(List<T>)` (solo appends/constantes entre medias) | ✔ | factoría nativa | TLAB nativo (capacidad 0) |
| `NewClassInstance` (144) | `AllocInstance` + `CallDirect` al constructor; `new C(..)` se devirtualiza y la factoría trivial se inlinea en el llamador | ✔ (clase y constructor resueltos al preparar) | asignación nativa + **constructor inlineado** (stores a offsets conocidos, sin guarda de clase); si no, factoría + llamada | TLAB nativo, cualquier clase |
| `NewInterfaceOrTypeObject` (145) | literal con claves constantes + `BindInterface` exacto | ✔ | factoría nativa | TLAB nativo (≤4 campos) |
| `NewObjectShape` (146) | literal de solo `null` con layout conocido | ✔ | factoría nativa | TLAB nativo |
| `NewObjectEmpty` (147) | `AllocMap` | ✔ | runtime leaf (como `AllocMap`) | TLAB nativo |
| `NewObjectWithProperties` (148) | `AllocMap` + `MapSet`/`StoreProperty` contiguos con clave constante | ✔ | factoría nativa | TLAB nativo (≤4 campos) |
| `GetFieldShape` (149) | todo `LoadProperty` | 1 búsqueda: Map → PIC/familia del shape; instancia de clase → caché clase→offset | como `LoadProperty` | PIC nativo |
| `SetFieldShape` (150) | todo `StoreProperty` | ruta autoritativa (contratos combinados del shape) | como `StoreProperty` | PIC nativo |
| `GetFieldClass` (151) | `LoadField` de un campo propio | `nullable` demostrado → sin trampa de no-inicializado | como `LoadField` | carga nativa (sin salida en `null` si `nullable`) |
| `SetFieldClass` (152) | `StoreField` de un campo propio | `unchecked_store` demostrado → sin comprobación de contrato, 1 búsqueda | sin guarda de tipo si demostrado | almacén nativo también para campos tipados demostrados |
| `InvokeMethodDirect` (153) | bytecode: `Invoke` con perfil de compilación (latente); **VM**: promoción en caliente de `Invoke` monomórficos (32 aciertos) | ✔ (handler perfilado) | como `Invoke` (ya especula por clase observada) | llamada directa nativa con guarda clase+epoch (antes rechazaba la función) |

Los opcodes 154–179 se eliminaron: nunca tuvieron handler y el intérprete ya
fusiona en preparación (`install_superinstructions`).

## Invariantes que hay que mantener

- El tracing JIT normaliza `Get/SetField*` e `InvokeMethodDirect` en `decode_at`
  (conservando los flags de `FieldClass` en `y`), y el inliner reconoce
  `ImplementInterfaceOrType`; si no, se pierde el inlining.
- `DefineArrayWithGenerics` nunca debe salir del tipo **inferido** de un literal.
- Los literales estructurales tipados se bajan como `AllocMap` + `StoreProperty`.
- Los constructores nunca se memoizan (análisis + verificador).
- `unchecked_store` solo para contratos que no instalan estado y con igualdad
  exacta en tipos nominales (los contratos de clase son exactos en ejecución).
- La asignación "solo cáscara" de `NewClassInstance` en trazas usa
  `OBJECT_FACTORY` con continuación = el mismo PC; `CONSTRUCTOR_FACTORY` ejecuta
  asignación + constructor.

## Pendiente

- [ ] `MapLookupOrInsert`: runtime + JITs (merge con master).
- [ ] Instancias con campos inline en el objeto gestionado (hoy `Box<[Value]>`:
      cada `new` hace un malloc; es el coste dominante restante de `NewClassInstance`).
- [ ] Reenvío store→load en trazas (constructor inlineado seguido de lecturas).
- [ ] `len2(p)` con `p` instancia de una clase `implements V`: el bind estructural
      no se puede demostrar no-op en la traza.
- [ ] WFJIT rechaza funciones que usan `this` (métodos y constructores).

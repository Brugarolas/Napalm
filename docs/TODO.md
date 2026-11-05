# TODO — Napalm Reloaded bytecode specializations

> Estado a 5 de noviembre de 2026, fase 2 (purga). Este archivo distingue **opcodes registrados**,
> **emisión experimental** y **ejecución real**. No se debe interpretar un opcode
> definido como un handler implementado.

## Reglas de activación

- [x] Añadir 29 opcodes nuevos sin renumerar los 135 existentes (`135..=163`).
- [x] Purgar los 14 candidatos que ya tenían equivalentes o podían eliminarse sin un opcode.
- [x] Reservar los 16 opcodes supervivientes (`164..=179`) sin alterar los anteriores.
- [x] Incorporar `FuseSurvivorSuperinstructions` y reubicación del bytecode.
- [x] Preservar las raíces GC y liveness de la segunda etapa en `FusionContinuation`.
- [x] Añadir rutas de emisión automáticas comprobadas estructuralmente para 15 de 16 supervivientes.
- [ ] Emitir `MapLookupOrInsert` desde un HIR dedicado que distinga clave ausente de valor nulo.
- [x] Añadir descriptores tipados por función en `BytecodeChunk.specializations`.
- [x] Extender decodificador, desensamblador, ensamblador y verificador de bytecode.
- [x] Incluir `CompileOptions.emit_specialized_bytecode` (predeterminado: `false`).
- [x] Mantener el optimizador post-HIR del bytecode sin emisión obligatoria de opcodes nuevos.
- [ ] Implementar handlers del intérprete para **todos** los opcodes nuevos.
- [ ] Implementar lowering y side exits correctos en Cranelift tracing JIT.
- [ ] Implementar lowering y alias/GC/deopt en el tier LLVM.
- [ ] Añadir pruebas de equivalencia diferencial (`legacy` frente a `specialized`) y fuzzing.
- [ ] Medir speedup por opcode y counters de hit/fallback; desactivar especializaciones con regresión.
- [ ] Congelar versión de serialización de descriptores antes del modo ejecutable autocontenido.

`emit_specialized_bytecode = true` genera bytecode que **todavía no es ejecutable**
con el intérprete actual. El verificador comprueba estructura y metadatos; no
sustituye las pruebas runtime. Cada fila bajo «Emisión» solo significa que existe
un camino en el compilador experimental, no que el handler esté terminado.

## Superinstrucciones solicitadas

| Opcode (ID) | Emisión en compilador | Interpreter | Tracing JIT | LLVM JIT | Pendiente |
|---|---|---|---|---|---|
| `DefineClass` (135) | Experimental, activable por flag | ☐ | ☐ | ☐ | Fusión del descriptor de clase y sus métodos en un único dispatch |
| `DefineInterface` (136) | Solo definición/assembler | ☐ | ☐ | ☐ | La declaración está en `RuntimeMetadata.contract_names`; falta punto de ejecución en HIR o bootstrap |
| `DefineType` (137) | Solo definición/assembler | ☐ | ☐ | ☐ | Como `DefineInterface`; los aliases no necesariamente son `Record` |
| `DefineDerivedClass` (138) | Experimental, activable por flag | ☐ | ☐ | ☐ | Fusión de clase y resolución de superclass |
| `DefineClassWithImplements` (139) | Experimental, activable por flag | ☐ | ☐ | ☐ | Fusión de clase y contratos `implements` |
| `DefineDerivedClassWithImplements` (140) | Experimental, activable por flag | ☐ | ☐ | ☐ | Herencia + implementación de contratos + métodos |
| `ImplementInterface` (141) | Experimental, activable por flag | ☐ | ☐ | ☐ | `BindInterface` de contrato abierto conocido |
| `ImplementType` (142) | Experimental, activable por flag | ☐ | ☐ | ☐ | `BindInterface` de contrato `Record` cerrado |
| `DefineArrayWithGenerics` (143) | Experimental, activable por flag | ☐ | ☐ | ☐ | `AllocList`/`AllocListCap` cuyo resultado es `List<T>` |
| `NewClassInstance` (144) | Experimental, activable por flag | ☐ | ☐ | ☐ | `AllocInstance` + `CallDirect` a constructor conocido, inmediatamente contiguos |
| `NewInterfaceObject` (145) | Solo definición/assembler | ☐ | ☐ | ☐ | Detectar literal + contrato + orden de inicializadores sin escapes |
| `NewObjectShape` (146) | Solo definición/assembler | ☐ | ☐ | ☐ | Demostrar inicialización de shape/contrato sin comprobaciones omitidas |
| `NewObjectEmpty` (147) | Experimental, activable por flag | ☐ | ☐ | ☐ | `AllocMap` |
| `NewObjectWithProperties` (148) | Solo definición/assembler | ☐ | ☐ | ☐ | Fusionar `AllocMap` y `MapSet` manteniendo orden y excepciones |
| `GetFieldShape` (149) | Experimental, activable por flag | ☐ | ☐ | ☐ | `LoadProperty` con layout y slot explícitos |
| `SetFieldShape` (150) | Experimental, activable por flag | ☐ | ☐ | ☐ | `StoreProperty` con layout y slot explícitos |
| `GetFieldClass` (151) | Experimental, activable por flag | ☐ | ☐ | ☐ | `LoadField` en función con propietario de clase conocido |
| `SetFieldClass` (152) | Experimental, activable por flag | ☐ | ☐ | ☐ | `StoreField` en función con propietario de clase conocido |
| `InvokeMethodDirect` (153) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Invoke` con perfil monomórfico y tipo de clase conocido, IC para fallback |
| `ArrayMapPacked` (154) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Invoke map` sobre `List<T>` tipada, IC para fallback |
| `ArrayFilterPacked` (155) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Invoke filter` sobre `List<T>` tipada, IC para fallback |
| `ArrayReduceNumeric` (156) | Solo definición/assembler | ☐ | ☐ | ☐ | Demostrar callback numérico puro y semántica de orden/NaN/Infinity |
| `ArrayForEachPacked` (157) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Invoke forEach` sobre `List<T>` tipada, IC para fallback |
| `ArrayFilterReducePacket` (158) | Solo definición/assembler | ☐ | ☐ | ☐ | Fusión de dos callbacks sin observabilidad de array intermedio |
| `ArrayMapReducePacked` (159) | Solo definición/assembler | ☐ | ☐ | ☐ | Fusión de dos callbacks sin observabilidad de array intermedio |
| `ApplyDecoratorToFunction` (160) | Solo definición/assembler | ☐ | ☐ | ☐ | Identificar decorador y target verificables en el HIR |
| `ApplyDecoratorToClass` (161) | Solo definición/assembler | ☐ | ☐ | ☐ | Conservar identidad de clase, orden y metadata del decorador |
| `AwaitPromise` (162) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Await` sobre `Builtin(Promise, …)` conocido |
| `LaunchConcurrentTask` (163) | Experimental, activable por flag | ☐ | ☐ | ☐ | `Launch` de `Closure(Some(FuncId))` concurrente conocido |

## Resultado de la purga de candidatos anteriores

Había **30** entradas en `SuperinstructionCandidate`, **no** 30 opcodes. Se
eliminan **14** entradas redundantes y los **16** supervivientes reciben
opcodes estables (`164..=179`). *La existencia de un opcode NO implica que
el intérprete ni los JITs lo ejecuten*.

| Opcode nuevo | ID | Generación experimental (`emit_specialized_bytecode=true`) | Interpreter | Tracing JIT | LLVM JIT |
|---|---:|---|:---:|:---:|:---:|
| `LoadI8IntegerArithmetic` | 164 | Emisión automática: LoadI8 + op entero; el registro inmediato debe quedar sobrescrito | ☐ | ☐ | ☐ |
| `LoadI8IntegerComparison` | 165 | Emisión automática: LoadI8 + ICmp; no se pierde el valor literal | ☐ | ☐ | ☐ |
| `LoadI8Return` | 166 | Emisión automática: LoadI8 + Return | ☐ | ☐ | ☐ |
| `LoadLiteralReturn` | 167 | Emisión automática: LoadNull/LoadTrue/LoadFalse + Return | ☐ | ☐ | ☐ |
| `AllocListAndAppend` | 168 | Emisión automática: AllocList/DefineArrayWithGenerics + ListAppend; tipos/capacidad y ambas etapas GC conservadas | ☐ | ☐ | ☐ |
| `AllocInstanceAndInitializeField` | 169 | Emisión automática: AllocInstance + StoreField/SetFieldClass; constructor NO incluido | ☐ | ☐ | ☐ |
| `AllocInstanceAndInitializeProperty` | 170 | Emisión automática: AllocInstance + StoreProperty/SetFieldShape; segunda etapa GC conservada | ☐ | ☐ | ☐ |
| `LoadFieldAndCall` | 171 | Emisión automática restringida: LoadField/GetFieldClass + CallClosure; temporal sobrescrito | ☐ | ☐ | ☐ |
| `LoadPropertyAndCall` | 172 | Emisión automática restringida: LoadProperty/GetFieldShape + CallClosure; temporal sobrescrito | ☐ | ☐ | ☐ |
| `GuardThenFieldLoad` | 173 | Emisión automática: GuardType/GuardClass + LoadField/GetFieldClass con registro del guard coalescido | ☐ | ☐ | ☐ |
| `GuardThenPropertyLoad` | 174 | Emisión automática: GuardType/GuardClass + LoadProperty/GetFieldShape con guard coalescido | ☐ | ☐ | ☐ |
| `MapLookupOrInsert` | 175 | Opcode + descriptor + `Assembler::emit_map_lookup_or_insert`; SIN emisión automática HIR segura | ☐ | ☐ | ☐ |
| `ImportAndReadVariable` | 176 | Emisión automática: ImportModule + ImportVariable con registro temporal sobrescrito | ☐ | ☐ | ☐ |
| `DirectCallAndReturn` | 177 | Emisión automática: CallDirect + Return del resultado | ☐ | ☐ | ☐ |
| `ClosureCallAndReturn` | 178 | Emisión automática: CallClosure + Return del resultado | ☐ | ☐ | ☐ |
| `ConcurrentLaunchAndAwait` | 179 | Emisión automática restringida: Launch/LaunchConcurrentTask + Await/AwaitPromise con registro de Promise sobrescrito; conserva los dos stages GC | ☐ | ☐ | ☐ |

### Candidatos eliminados (14): equivalencias

| Candidato purgado | Sustituto / motivo |
|---|---|
| `MoveReturn` | `Return` directo tras mover o coalescing; no justifica opcode nuevo |
| `AllocMapAndInsert` | `NewObjectWithProperties` (148) |
| `ArrayMap`, `ArrayForEach`, `ArrayFilter`, `ArrayReduce` | `ArrayMapPacked` (154), `ArrayForEachPacked` (157), `ArrayFilterPacked` (155), `ArrayReduceNumeric` (156) respectivamente; este último requiere prueba numérica, NO reduce genérico |
| `CreateClassWithSuperclass`, `CreateClassWithMethods` | `DefineDerivedClass` (138) / `Define*Class` (135, 138–140) |
| `RegisterInterfaceContract`, `RegisterTypeContract` | `DefineInterface` (136), `DefineType` (137) |
| `BindClassInterface`, `BindClassType` | `DefineClassWithImplements` (139) y `DefineDerivedClassWithImplements` (140) |
| `AllocateStructuralObject` | `NewInterfaceObject` (145) o `NewObjectShape` (146), según semántica |
| `CreateObjectWithProperties` | `NewObjectWithProperties` (148) |

### Restricciones de la fusión experimental

- El optimizador solo emite una fusión cuando la **segunda instrucción no es una entrada CFG externa**, excepción, OSR, punto de reentrada ni destino de deoptimización.
- Un temporal intermedio que pudiera leerse después impide la fusión. Las variantes que lo sobrescriben están expresamente reconocidas.
- `FusionContinuation` contiene los mapas de GC y liveness del *segundo* paso interno; el primer paso mantiene los mapas asociados al PC del nuevo opcode. **El handler deberá utilizar ambos**.
- `CompactBytecode` reubica saltos, tablas de excepciones, GC, liveness, source maps y deopts. No se lanza el pipeline de superinstrucciones sin activar el flag.
- `MapLookupOrInsert` **no** se infiere de `MapGet + MapSet`: una escritura incondicional es observable y no equivale a insertar solo si falta la clave.
- Las superinstrucciones aún **no son ejecutables** por el intérprete ni por ningún JIT existente. Se han añadido tests del formato y de rutas de fusión, pero no se han ejecutado `cargo test` ni smoke tests completos en este entorno.

## Validación obligatoria antes de permitir estos opcodes en producción

- [ ] Confirmar que cada operación compleja ejecuta callbacks en el mismo orden que la ruta base.
- [ ] Confirmar que errores, excepciones, getters y writes observables se propagan idénticamente.
- [ ] Precisión de GC: el handler debe enraizar los objetos intermedios creados dentro de un opcode.
- [ ] Stack maps en cada PC que puede asignar memoria, llamar o suspender un fiber.
- [ ] `NewClassInstance`: ignorar el valor devuelto por el constructor y devolver la instancia.
- [ ] `Define*Class`: conservar orden de registros y publicar la clase solo después de quedar completa.
- [ ] `Get/SetFieldClass`: resolver el offset relativo a la clase tras la herencia real.
- [ ] `Get/SetFieldShape`: validar shape y slot, hacer fallback dinámico si difieren.
- [ ] `Array*Packed`: validar identidad del método, storage, holes, mutaciones, callbacks y excepciones.
- [ ] `ArrayReduceNumeric`: no alterar semántica IEEE-754 ni orden de evaluación.
- [ ] `LaunchConcurrentTask`: respetar copia de argumentos y aislamiento de heaps.
- [ ] `AwaitPromise`: mantener pending/resolve/reject y scheduling de microtasks.
- [ ] Contemplar nuevas instrucciones en tracing/JIT, perfilado, serialización y cache invalidation.

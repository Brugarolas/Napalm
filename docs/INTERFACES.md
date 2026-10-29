# Object interfaces

An interface annotation on a variable validates the object and installs a permanent contract on that object. An invalid initializer or later mutation reports `Interface type error` and stops the program with a runtime error exit status. These errors bypass `try/catch` and `Fiber.try`; a program cannot continue with a violated contract.

```typescript
interface Position { x: number; y: number; }
interface Named { name: string; }
interface Player extends Named {
    readonly id: number;
    position: Position;
    scores: number[];
    nickname?: string;
}

const original = {
    id: 1, name: 'Ada', position: {x: 0, y: 0}, scores: [10]
};
const player: Player = original;
player.position.x += 1;
player.scores.add(20);

// The contract belongs to the object, including aliases created before binding.
original.position.x = 'invalid'; // Fatal type error before writing the value.
```

Declarations are module scoped and can refer to later interface declarations. `extends` supports multiple parents; duplicate fields and conflicting inherited definitions are rejected. Interfaces describe object literals/Maps, not class instances. Extra fields are allowed and remain unconstrained unless another attached interface constrains them.

Supported field types include `number`, `string`, `boolean`, `null`, `any`, `never`, function values, arrays (`T[]`, `Array<T>`), named interfaces, inline object schemas, and unions. For example, `child: { value: number; }` and `entries: Position[]` protect nested values too. Function fields check that the value is callable; parameter and return signatures are not checked. Literal annotations retain the existing widening rules (`'a'` denotes `string`, for example).

Optional fields may be absent; reading an absent property returns Napalm's `null`. An explicit `null` value requires `null` in the field type. Required fields cannot be removed or cleared. `readonly` fields cannot be assigned or removed after the contract is attached; this is shallow, so a readonly reference may still contain mutable fields. Optional non-readonly fields can be removed.

Dot assignments, bracket assignments, compound updates, Map removal/clear, List assignment/add/insert, closures and host collection APIs all enforce the same contract. Replacing a nested object or list validates and protects the replacement. Removing a reference never detaches the old object's contract. Multiple interfaces attached to one object all remain in force. For unions of mutable objects/lists, binding selects the first matching branch and retains that branch's contract.

Spread/rest create independent outer Maps. Their nested references remain shared and retain any existing contracts. `any` permits changing a binding's value, but never removes a contract from an object it references. Map builtin getters (`count`, `size`, `isEmpty`, `keys`, `toString`) retain their precedence over dot property reads; use brackets for fields with those names. Generic `Map<K, V>` and `Set<T>` also install persistent content contracts; see [collections](COLLECTIONS.md).

Generic and recursive interfaces, index signatures, interface merging, type aliases as contract definitions, interface imports, class `implements` checks, and parameter/return contracts are not implemented. Unsupported or unresolved field types in declared interfaces are diagnosed rather than silently weakened. Interface names used as variable annotations must be declared in the same module. Type-only imports and other previously erased annotation forms are still syntax-only.

## Compiler and JIT

`Type::Interface` records the schema in HIR and bytecode metadata. `BindInterface` is an effectful instruction: it validates the entire value before installing contracts and yields a checked SSA value. It cannot be discarded as an unused speculative guard. Statically known interface receivers use direct `MapGet` instructions with the declared field type; optional fields include `null`. These facts can support existing type-based guard elimination without assuming an unchecked receiver is valid.

Contracts contain no managed references and move/die with the owning GC object. Native collection reads retain the existing shape/bounds proofs. Native stores have an explicit `GuardNoContract` in Trace IR, so even traces compiled before a contract was attached must exit before an unchecked write. The check survives dead-store elimination. Writes observed on protected collections use checked runtime paths inside traces. Numeric, shape and bounds guards are retained wherever a contract does not prove them redundant.

## Validation

Run `cargo test --offline napalm_interfaces -- --test-threads=1`. The regression tests exercise both interpreter and JIT execution, including existing aliases, nested objects/lists, GC, readonly/optional fields, multiple contracts, rejected initializers, uncaught violations, and traces compiled before contract attachment. They also require native execution for hot typed mutation loops and verify that overwritten invalid writes still fail.

The pre-change repository's full suite has 34 failing tests after correcting three stale test ABI/type references needed to compile it. The implementation was compared against an isolated archive of that baseline; those existing failures are not interface regressions.

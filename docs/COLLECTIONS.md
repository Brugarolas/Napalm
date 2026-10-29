# Map and Set

```typescript
const scores = new Map<string, number>();
scores.set('Ada', 42).set('Lin', 7);
System.print(scores.get('Ada')); // 42
System.print(scores.size);       // 2

const names: Set<string> = new Set();
names.add('Ada').add('Ada').add('Lin');
System.print(names.size);        // 2
for (const name of names) System.print(name);
```

Both constructors currently take no arguments. Untyped `new Map()` and `new Set()`
accept mixed element types. Both collections preserve insertion order, including
when a Map changes from shaped storage to dictionary storage. Updating an existing
entry keeps its position; deleting and reinserting appends it to the end.

| Collection | Native operations |
| --- | --- |
| Map | `get(key)`, `set(key, value)`, `has(key)`, `delete(key)`, `clear()`, `size` |
| Set | `add(value)`, `has(value)`, `delete(value)`, `clear()`, `size` |

`Map.set` and `Set.add` return the receiver for chaining. `delete` returns whether
the entry existed, including when its stored value was `null`. Missing Map entries
return `null`. `clear()` returns `null`.

The historical Map API remains available: subscripts, `count`, `isEmpty`,
`containsKey`, `remove` (returns the removed value), and the `keys` getter. `keys()`
also returns the same live key view. Map iteration still yields entries with
`.key` and `.value` properties. Set iteration yields its elements, supports
`for...of` and spread, and is also available through `keys()` and `values()`.
Sets additionally expose `count` and `isEmpty`.

Keys and elements use Map key equality: strings compare by content, other objects
by identity, all NaNs compare as one key, and positive/negative zero share one key.
Iteration uses Napalm's existing numeric cursor protocol; mutation during iteration
does not promise JavaScript iterator behavior. Dictionary lookup/insertion uses a
hash index; deletion preserves order by shifting the following entries.

## Generic contracts

`new Map<K, V>()`, `new Set<T>()`, and variable annotations `Map<K, V>` / `Set<T>`
install persistent runtime contracts. Existing contents are validated before the
contract is installed. Subsequent writes through methods, Map brackets/properties,
closures, host APIs, or an `any` alias must satisfy every installed contract.
Clearing a collection does not remove its contract.

```typescript
interface Item { value: number; }
const groups = new Map<string, Set<Item>>();
const items = new Set<Item>();
items.add({value: 42});
groups.set('items', items);

const alias: any = items;
alias.add('wrong'); // Fatal structural type error before mutation.
```

Type arguments support scalars, unions, lists, nested collections, named object
contracts and class instances under the existing exact class type rules. Attaching
a structural element/key/value contract also protects that referenced object from
later incompatible mutations. As with typed lists and interfaces, contract errors
terminate execution and bypass `try/catch`.

This does not add reified user-defined generic functions/classes or instantiate
generic type aliases. Existing erased type parameters and unresolved type names
retain the language's current behavior. `ReadonlyMap` and `ReadonlySet` annotations
check content types but do not enforce readonly access.

## Runtime and validation

Set has its own runtime identity and a private Map backing store, traced by the GC.
Native collection primitives remain callable inside JIT traces. Map `get`, `set`
and `size` additionally use HIR wrappers over the existing Map bytecode intrinsics,
allowing the JIT to inline shaped/PIC access. Generic Maps use checked dictionary
storage so shape-only writes cannot bypass their key/value contracts, including
through aliases whose traces were compiled before the contract was attached.

Run `cargo run -- scripts/map.napalm` and
`cargo test --offline napalm_collections -- --test-threads=1`. The tests cover
interpreter/JIT parity, native loops, aliases, nested types, class keys, GC,
insertion order, duplicate elements, null values and rejected writes.

Validation on 2026-10-30: the example passes with JIT enabled and disabled, and
all eight collection regression tests pass. The full suite reports 330 passed,
52 failed and one ignored. A separate `HEAD` archive reports 318 passed, 54 failed
and one ignored; every remaining failure also occurs in that baseline. Both runs
use the same correction to stale test calls that passed `CompiledTrace` by value
instead of by reference. Structural-error tests accept a proven compile-time type
or readonly error as well as a runtime rejection; native-required cases must
still execute. Concurrent comprehension changes were retained and their missing
AST definition, multiline parsing and source-function closure count integrated.

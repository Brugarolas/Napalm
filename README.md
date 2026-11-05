
<p align="center">
  <a href="https://www.buymeacoffee.com/brugarolas"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffe"></a>
</p>

# Napalm

Napalm is the continuation of WrenJIT with a TypeScript-inspired syntax and a dynamic runtime. WrenJIT remains the frozen Wren-syntax project. This repository builds the `napalm` executable.

For the complete, current language reference and runnable examples, see [DOCUMENTATION.md](DOCUMENTATION.md).

Variable types are optional: the compiler checks known incompatible assignments. Object interfaces also install persistent runtime contracts, enforced through every alias and in the JIT. See [object interfaces](docs/INTERFACES.md). Napalm does **not** promise compatibility with the full TypeScript/JavaScript language, Node.js, or browser APIs.

## Stats

<p align="center">
  <a href="https://github.com/Brugarolas/Napalm/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/Brugarolas/Napalm/stargazers"><img src="https://img.shields.io/github/stars/Brugarolas/Napalm?style=flat-square" alt="Stars"></a>
  <a href="https://github.com/Brugarolas/Napalm/issues"><img src="https://img.shields.io/github/issues/Brugarolas/Napalm?style=flat-square" alt="Issues"></a>
</p>

<p align="center">
  <img src="[https://img.shields.io/badge/license-MIT-blue?style=flat-square](https://repobeats.axiom.co/api/embed/44379307ec6623c00131593148f800c0122f9562.svg)" alt="Repobeats analytics image">
</p>

## Run

```sh
cargo run --release -- scripts/complete.napalm
cargo run --release -- scripts/module_import.napalm
cargo test napalm -- --test-threads=1
```

Use `.napalm` for Napalm programs. An extensionless import tries the exact path, `.napalm`, and `index.napalm`.

```typescript
import { Answer } from './module_exports';

interface Named { name: string; }
type Names = string[];

class Greeter implements Named {
    name: string;
    constructor(name: string) { this.name = name; }
    greet(): string { return `Hello, ${this.name}!`; }
}

const person = new Greeter('Napalm');
const twice = (value: number): number => value * 2;
System.print(person.greet());
System.print(twice(Answer));
```

## Supported syntax

- `let`, `const` and `var`, including multiple declarations. Bindings are block-scoped; `const` requires an initializer and rejects reassignment, including from a closure. Objects held by a constant remain mutable.
- `function name(...) { ... }`, anonymous `function (...) { ... }`, arrow functions, direct calls `fn(...)`, and lexical closures. Functions are initialized when their declaration executes; there is no JavaScript hoisting.
- `class`, `constructor(...)`, `new Class(...)`, `extends`, `super(...)`, `super.method(...)`, declared instance/static fields, methods, `get` and `set`. Field initializers run for each construction; static initializers run once when the declaration executes. Use `this.member` explicitly.
- Decorators on functions, classes, methods, getters, setters and fields, including static members: `@decorator`, `@namespace.decorator` and `@factory(...)`. See [decorator semantics](#decorators) and the [executable decorator example](scripts/decorators.napalm).
- Optional variable annotations and object interfaces with required/optional fields, `readonly`, `extends`, unions, nested interfaces and typed lists. Interface violations terminate execution, including inside `try/catch`. Parameter/return annotations, `type` aliases, generic parameters, `implements`, `as Type` and class visibility/readonly modifiers remain erased; see [interface scope and limits](docs/INTERFACES.md).
- `if`/`else`, `while`, `for (let i = 0; i < n; i++)`, `for (const item of collection)`, `break`, `continue` and `return`.
- Arithmetic, comparisons, `==`/`!=` and `===`/`!==`, bitwise operators, `&&`, `||`, conditional expressions, assignment, `+=`, `-=`, `*=`, `/=`, `%=`, and prefix/postfix `++`/`--`. `instanceof` replaces Wren's `is` operator.
- Single/double quoted strings and backtick templates with nested `${expression}` interpolation. `%` is ordinary text.
- Array literals and object/Map literals with named, computed or shorthand keys. Dot access resolves Map keys when a built-in member does not already exist; brackets always access the key. Function-valued Map properties can be called directly.
- Array/List and object/Map destructuring in declarations, assignments, parameters and `for...of` bindings: nested patterns, holes, aliases, computed keys, lazy defaults and final `...rest` elements/properties. Patterns evaluate each source and computed key once. Missing values are `null`; defaults also apply to explicit `null`, following Napalm's single missing-value representation.
- Spread in arrays (`[...items]`), objects (`{...object}`), function/method calls (`fn(...args)`), constructors and `super` calls. Array and call spread use Napalm's iterator protocol; object spread copies Map entries and ignores `null`. Copies and rest collections are shallow and independent. Later object entries overwrite earlier keys. Expanded calls retain the 16-argument limit and report a catchable error when it is exceeded.
- Named/default/side-effect imports; named/default declaration exports, `export { local as public }`, and named re-exports with `from`. Only explicit exports are public.
- `async function`, async methods/arrows, `await`, `throw`, `try`/`catch`/`finally`, plus Napalm's `defer` and `errdefer` extensions.
- Semicolons and line-separated statements. Braced function bodies always contain statements; only expression-bodied arrows return implicitly. Napalm does not implement every ECMAScript automatic-semicolon-insertion edge case.
- HTTP services with `listen "127.0.0.1:3000" { route "GET" ["/"] (req) => ({body: "Hello!"}); }`, async handlers, path parameters, streaming/SSE and simple HTTP clients. See [HTTP services](docs/HTTP.md) and [the server example](scripts/server.napalm).

Examples are in [scripts/complete.napalm](scripts/complete.napalm), [scripts/module_import.napalm](scripts/module_import.napalm), [scripts/builtins.napalm](scripts/builtins.napalm) and [scripts/destructuring_spread.napalm](scripts/destructuring_spread.napalm).

## Decorators

Each decorator receives `(value, context)`. `context.name` is the declaration/member name,
and `context.kind` identifies its target. Decorator expressions/factories are evaluated
top-down in the declaration's lexical scope and decorators are applied bottom-up.
Returning `null` (including falling off the end) preserves a function, class or method.
Other results must be a class for a class decorator or a function for other targets;
invalid values produce catchable runtime errors.

Napalm preserves lexical `this`. Member callables therefore use an **explicit receiver**
as their first argument instead of JavaScript's dynamic `this` / `Function.call(thisArg, ...)`.

| `context.kind` | Value passed to the decorator | Replacement signature |
| --- | --- | --- |
| `"function"` | Original function | Same arguments as the original |
| `"class"` | Original class | Return the replacement class |
| `"method"` | Callable for the original method | `(receiver, ...originalArguments)` |
| `"getter"` | Callable for the original getter | `(receiver)` |
| `"setter"` | Callable for the original setter | `(receiver, nextValue)` |
| `"field"` | `null` | `(receiver, initialValue) => transformedValue` |

The `...` above denotes the signature convention, not supported rest-parameter syntax:
write each parameter explicitly. Decorated methods accept at most 15 ordinary arguments,
leaving one slot for the receiver. Members also receive `context.static` and `context.private`.
`private` is currently `false`: Napalm erases TypeScript visibility modifiers and does not
implement ECMAScript `#private` fields.

`context.addInitializer(fn)` accepts zero-argument callbacks for functions/classes and
one-argument `(receiver)` callbacks for members. It can only be called while that decorator
is executing, even if the decorator throws. The original function/class's final binding
is installed before its callbacks run. Member decoration runs once per class declaration:

- Instance method/getter/setter callbacks run for each construction, after `super()` in
  derived classes and before ordinary field initialization.
- Static method/getter/setter callbacks run once, before static fields and blocks.
- Field transformers run top-down for each instance (once for static fields), using `null`
  when there is no initializer. Each field's extra callbacks run after that field is assigned.
- Extra callbacks run in registration order. An error aborts initialization and propagates;
  later callbacks do not run.

This is a Napalm extension inspired by standard JavaScript decorators, not full proposal
compatibility: `context.access`, metadata, auto-accessors, constructor decorators, computed/watch
decorators, foreign/abstract/declare member decorators and operator decorators are unsupported.
Place decorators before `export` for exported declarations.

The [decorators example](scripts/decorators.napalm) includes assertions for all six targets,
stacking, static members, initializers and hot-loop dispatch with GC. A failed assertion throws:

```powershell
cargo run -- scripts\decorators.napalm
cargo test napalm_decorators -- --test-threads=1
```

### Native decorators

The following globals provide Rust-backed caching and scheduling. Use the options as
an object literal, for example `@debounce(1000, {leading: false, trailing: true})`;
`options = {...}` is an assignment expression, not a named-argument syntax.
All time arguments, including `maxWait` and cache expiration, are **milliseconds**.

| Decorator | Behavior |
| --- | --- |
| `@memoize` | Cache successful return values by the complete argument tuple. |
| `@memoizeExpiring(ttl)` | Memoize each key for `ttl` milliseconds from insertion; hits do not renew its lifetime. |
| `@once` | Invoke until the first successful return, then reuse that result regardless of subsequent arguments. |
| `@delay(wait)` | Schedule every call independently after `wait` milliseconds; return a Promise for its result. |
| `@debounce(wait, options)` | Group a burst of calls; invoke using the latest arguments after a quiet period. Defaults: `leading: false`, `trailing: true`. Optional `maxWait` limits postponement during a continuous burst. |
| `@throttle(wait, options)` | Limit execution to a time window, retaining the latest arguments for its trailing edge. Defaults: `leading: true`, `trailing: true`. |

These decorators support functions and callable class members, not classes or field
transformers. The explicit member receiver is preserved. Memoization includes the
receiver in its key; `once`, debounce and throttle maintain separate state per receiver.

Caches reuse the runtime's AES-backed hasher and compare complete keys rather than
treating a hash as a unique identity. Strings compare by content, objects by identity,
and numeric signed zero compares as one key. Objects are not deep-hashed: mutating an
argument does not invalidate its cached result. Captured state and global variables
are not part of the key, so memoize only functions whose result is appropriate to reuse.

Memoization and `once` preserve the original synchronous/async return convention.
Synchronous exceptions are not cached; a returned Promise is cached immediately,
including its eventual rejection. Caches retain the managed arguments/results needed
by their entries; avoid unbounded memoization of an unlimited argument space.

Delay, debounce and throttle always return Promises and adopt an async function's
result. Calls grouped into the same invocation share its outcome. With both leading
and trailing enabled, a lone leading call does not produce a duplicate trailing call;
later calls in that window receive the trailing outcome. With trailing disabled,
suppressed calls share the leading invocation's outcome. Disabling both edges is
invalid. Scheduled exceptions reject the corresponding Promise instead of disappearing
inside the timer callback.

Run the [native decorator example/test](scripts/native_decorators.napalm):

```powershell
cargo run -- scripts\native_decorators.napalm
cargo test native_decorators -- --test-threads=1
```

## Migration from WrenJIT

| WrenJIT | Napalm |
| --- | --- |
| `var x = 1` | `let x = 1;` or `const x = 1;` |
| `Fn.new { |x| x * 2 }` | `(x: number) => x * 2` |
| `fn.call(x)` | `fn(x)` |
| `construct new(x) { ... }` | `constructor(x) { ... }` |
| `Thing.new(x)` | `new Thing(x)` |
| `class B is A` | `class B extends A` |
| `value is Thing` | `value instanceof Thing` |
| `_field`, `__field` | declare `field` / `static field`; use `this.field` / `Thing.field` |
| `name { expression }` | `get name() { return expression; }` |
| `name=(value) { ... }` | `set name(value) { ... }` |
| `for (item in items)` | `for (const item of items)` |
| `"value: %(x)"` | backtick template with `${x}` |
| `Timer.delay(10) { ... }` | `Timer.delay(10, () => { ... })` |
| `import "dep" for Thing` | `import { Thing } from "dep";` |
| implicit public module bindings | `export const Thing = ...;` |
| `1...5`, `1..5` | `new Range(1, 5)`, `new Range(1, 5, true)` |
| `matrix[x, y]` | `matrix[x][y]` |
| named constructor | static factory method returning `new Class(...)` |

The public frontend does not recognize Wren trailing-block closures, pipe-delimited parameters, implicit receivers, `construct`, range operators or Wren import syntax. Historical Wren source fixtures and their parser remain solely to exercise the inherited compiler and runtime.

## Runtime and current limits

The register VM, SSA optimizer, tracing JIT, moving GC, Fibers, Promise/timer scheduler, exceptions, standard library and FFI are retained. See [engine architecture](docs/ENGINE.md), [FFI](README_FFI.md) and [platform helpers](README_SYSTEM_PLATFORM.md).

The standard library retains its existing API (`System.print`, `list.add`, `list.count`, `Timer.delay`, `Fiber`, etc.). Values, truthiness, equality, numbers, strings, collections and iteration keep the engine's dynamic semantics; using familiar syntax does not change them into ECMAScript semantics. Missing values use `null`; there is no separate `undefined`. Both equality spellings use the same non-coercing operation. The VM accepts at most 16 parameters/arguments.

`Map` and `Set` provide native collection operations, insertion order, and persistent generic contracts through `Map<K, V>` and `Set<T>`, including generic constructors. See [collections](docs/COLLECTIONS.md) and [the Map/Set example](scripts/map.napalm).

Destructuring currently operates on Lists and Maps, including empty patterns; it rejects `null` and other receiver types. Rest parameters (`function f(...args)`), defaults for whole parameters and omitted optional arguments remain unsupported; defaults inside destructured parameters are supported. Parameter arity is still fixed.

Other limits include `switch`, generators, regular-expression literals, namespace imports/exports, JSX, enums and user-defined operator/subscript declarations. Use explicit constructors and `super(...)` in derived classes, and normal named methods in place of operator declarations. Methods can still be overloaded by arity as an engine extension.

For class fields containing functions, use `(object.callback)(...)` to call the stored function; `object.method(...)` selects method dispatch. `private`, `protected`, `public`, `readonly` and `override` are erased modifiers, not runtime access controls. There is no function hoisting or per-iteration rebinding of a captured C-style `for` variable.

Package modules use `@lib/...`, standard modules use `@std/...`, with default roots `.napalm/packages` and `.napalm/std`. Configure `NAPALM_LIB_PATH`, `NAPALM_STD_PATH`, and `NAPALM_HOME`; `napalm.toml` or `.git` marks a project root. Package namespace traversal remains rejected.

`NAPALM_NO_JIT`, `NAPALM_JIT_STATS`, and `NAPALM_GC_STATS` control CLI diagnostics. Existing internal `WREN_*` tuning flags remain available for the inherited engine. Benchmark programs are in `scripts/jit.napalm`, `scripts/jit_entries.napalm` and `scripts/gc.napalm`; historical Wren benchmark results are not claims about this syntax revision.

### JIT benchmarks

```sh
npm run cargo::bench::jit
npm run node::bench::jit
npm run cargo::bench::entries
npm run node::bench::entries
bun bench/jit.js
bun bench/jit_entries.js
```

The Napalm commands build and run **release** executables. Compare the elapsed
times printed by each program, not Cargo/build time or process startup. Run each
command several times and compare medians on an otherwise idle machine.

Both versions print matching checksums outside the timed region. The general
benchmark includes cold JIT compilation; the entry benchmark warms up with 1,000
calls before timing 1,000,000 iterations. The JavaScript xorshift uses `>>>`
because Napalm's `>>` is an unsigned 32-bit shift.

Primitive type bindings remain eligible for replay-safe numeric inlining.
Structural/collection bindings still use the runtime to install persistent
contracts, and failed inlined primitive checks replay the original call to
preserve type diagnostics. Native boolean branch guards use the comparison
directly rather than materializing an integer and comparing it again.

The JIT also preserves live SSA literals at loop entry, folds successful literal
numeric guards, and merges small, effect-free numeric branches into unboxed
selections. Speculative checks replay the original branch; stores, allocations,
and arbitrary calls are not evaluated speculatively. Cold native-link exit
stubs stay outside the hot loop layout.

Remainders with integral constant divisors use exact integer strength reduction
when the dividend magnitude and divisor fit signed 32-bit range. Fractional
dividends and signed zero retain their original semantics; other values use the
existing floating-point remainder helper. These optimizations do not enable
approximate arithmetic or change the workloads or iteration counts.

## Verification

The new regression tests compile and execute programs through the full pipeline with JIT both disabled and enabled, covering closures, typed syntax, class initialization/inheritance, control flow, evaluation order, objects, imports/exports and asynchronous cleanup. Historical backend regressions are retained separately. See [validation notes](docs/VALIDATION.md) for the baseline comparison.

The syntax is inspired by the official TypeScript handbook's [classes](https://www.typescriptlang.org/docs/handbook/2/classes.html), [functions](https://www.typescriptlang.org/docs/handbook/2/functions.html), and [modules](https://www.typescriptlang.org/docs/handbook/2/modules.html).

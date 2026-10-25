# Napalm Reloaded

Napalm is the continuation of WrenJIT with a TypeScript-inspired syntax and a dynamic runtime. WrenJIT remains the frozen Wren-syntax project. This repository builds the `napalm` executable.

Type annotations are optional and erased. Napalm does **not** type-check programs or promise compatibility with the full TypeScript/JavaScript language, Node.js, or browser APIs.

## Run

```sh
cargo run --release -- scripts/complete.nap
cargo run --release -- scripts/module_import.nap
cargo test napalm -- --test-threads=1
```

Use `.nap` for Napalm programs. The module resolver also accepts `.ts` files written in the supported subset. An extensionless import tries the exact path, `.nap`, `.ts`, `index.nap`, and `index.ts`.

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
- Optional parameter, return and variable annotations; named/generic/array/tuple/object/function/union/intersection types; erased `type`, `interface`, generic declaration parameters, `implements`, `as Type` and class visibility/readonly modifiers. These provide syntax and documentation, with no type checking or runtime visibility enforcement.
- `if`/`else`, `while`, `for (let i = 0; i < n; i++)`, `for (const item of collection)`, `break`, `continue` and `return`.
- Arithmetic, comparisons, `==`/`!=` and `===`/`!==`, bitwise operators, `&&`, `||`, conditional expressions, assignment, `+=`, `-=`, `*=`, `/=`, `%=`, and prefix/postfix `++`/`--`. `instanceof` replaces Wren's `is` operator.
- Single/double quoted strings and backtick templates with nested `${expression}` interpolation. `%` is ordinary text.
- Array literals and object/Map literals with named, computed or shorthand keys. Dot access resolves Map keys when a built-in member does not already exist; brackets always access the key. Function-valued Map properties can be called directly.
- Named/default/side-effect imports; named/default declaration exports, `export { local as public }`, and named re-exports with `from`. Only explicit exports are public.
- `async function`, async methods/arrows, `await`, `throw`, `try`/`catch`/`finally`, plus Napalm's `defer` and `errdefer` extensions.
- Semicolons and line-separated statements. Braced function bodies always contain statements; only expression-bodied arrows return implicitly. Napalm does not implement every ECMAScript automatic-semicolon-insertion edge case.

Examples are in [scripts/complete.nap](scripts/complete.nap), [scripts/module_import.nap](scripts/module_import.nap) and [scripts/builtins.nap](scripts/builtins.nap).

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

Not implemented: destructuring, spread/rest/default/optional parameters, optional chaining, nullish coalescing, `switch`, `do...while`, JavaScript `for...in`, generators, regular-expression literals, namespace imports/exports, JSX, enums, decorators, TypeScript's advanced type operators, call-site type arguments, constructor parameter properties or user-defined operator/subscript declarations. Use explicit constructors and `super(...)` in derived classes, and normal named methods in place of operator declarations. Methods can still be overloaded by arity as an engine extension.

For class fields containing functions, use `(object.callback)(...)` to call the stored function; `object.method(...)` selects method dispatch. `private`, `protected`, `public`, `readonly` and `override` are erased modifiers, not runtime access controls. There is no function hoisting or per-iteration rebinding of a captured C-style `for` variable.

Package modules use `@lib/...`, standard modules use `@std/...`, with default roots `.napalm/packages` and `.napalm/std`. Configure `NAPALM_LIB_PATH`, `NAPALM_STD_PATH`, and `NAPALM_HOME`; `napalm.toml` or `.git` marks a project root. Package namespace traversal remains rejected.

`NAPALM_NO_JIT`, `NAPALM_JIT_STATS`, and `NAPALM_GC_STATS` control CLI diagnostics. Existing internal `WREN_*` tuning flags remain available for the inherited engine. Benchmark programs are in `scripts/jit.nap`, `scripts/jit_entries.nap` and `scripts/gc.nap`; historical Wren benchmark results are not claims about this syntax revision.

## Verification

The new regression tests compile and execute programs through the full pipeline with JIT both disabled and enabled, covering closures, typed syntax, class initialization/inheritance, control flow, evaluation order, objects, imports/exports and asynchronous cleanup. Historical backend regressions are retained separately. See [validation notes](docs/VALIDATION.md) for the baseline comparison.

The syntax is inspired by the official TypeScript handbook's [classes](https://www.typescriptlang.org/docs/handbook/2/classes.html), [functions](https://www.typescriptlang.org/docs/handbook/2/functions.html), and [modules](https://www.typescriptlang.org/docs/handbook/2/modules.html).

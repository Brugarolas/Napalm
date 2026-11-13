# Napalm: Language Manual and Usage Reference

Reference to the implementation in this repository, reviewed on
**November 3, 2026**. Written for anyone who wants to **program in Napalm**:
syntax, values, types, functions, objects, libraries, and examples.
It does not describe the JIT implementation, bytecode, or optimizers.

Napalm has syntax inspired by TypeScript, but **it is not JavaScript or
TypeScript with a different executable**. It also does not provide Node.js, the
DOM, or the complete standard library of those languages. A familiar-looking
construct may have different semantics.

This manual is based on the public parser, the installed library, observable
interpreter operations, its language tests, and the programs in
[scripts](scripts). Some comments and historical documents no longer describe
current behavior; important differences are explained here instead of being
presented as existing features.

## Table of Contents

1. [Getting Started and Running Programs](#1-getting-started-and-running-programs)
2. [Basic Syntax and Literals](#2-basic-syntax-and-literals)
3. [Values, Identity, and Conversions](#3-values-identity-and-conversions)
4. [Variables, Scopes, and Constants](#4-variables-scopes-and-constants)
5. [Operators and Precedence](#5-operators-and-precedence)
6. [Control Flow](#6-control-flow)
7. [Functions and Closures](#7-functions-and-closures)
8. [Types and Persistent Contracts](#8-types-and-persistent-contracts)
9. [Lists and Arrays](#9-lists-and-arrays)
10. [Objects, Map, and Set](#10-objects-map-and-set)
11. [Ranges and Custom Iteration](#11-ranges-and-custom-iteration)
12. [Destructuring, Rest, and Spread](#12-destructuring-rest-and-spread)
13. [List Comprehensions](#13-list-comprehensions)
14. [Classes and Inheritance](#14-classes-and-inheritance)
15. [Reactive Properties and Watchers](#15-reactive-properties-and-watchers)
16. [Enums](#16-enums)
17. [Decorators](#17-decorators)
18. [Native Decorators](#18-native-decorators)
19. [Exceptions and Cleanup](#19-exceptions-and-cleanup)
20. [Async, Promise, and Timers](#20-async-promise-and-timers)
21. [Concurrency with launch](#21-concurrency-with-launch)
22. [Fibers and Generators](#22-fibers-and-generators)
23. [Modules, Imports, and Exports](#23-modules-imports-and-exports)
24. [Strings](#24-strings)
25. [Numbers and Randomness](#25-numbers-and-randomness)
26. [System and Output](#26-system-and-output)
27. [Dates](#27-dates)
28. [Proxy](#28-proxy)
29. [HTTP Client and fetch](#29-http-client-and-fetch)
30. [HTTP Servers, Streaming, and SSE](#30-http-servers-streaming-and-sse)
31. [FFI and External Functions](#31-ffi-and-external-functions)
32. [Limitations and Differences from JavaScript](#32-limitations-and-differences-from-javascript)
33. [Complete Recipes](#33-complete-recipes)
34. [Example and Source Map](#34-example-and-source-map)
35. [Current Performance](#35-current-performance)
36. [Platform Detection (System.os, System.arch)](#36-platform-detection)
37. [Napalm Native FFI](#37-napalm-native-ffi)
38. [JSON](#38-json)
39. [Bytes](#39-bytes)
40. [ByteBuffer](#40-bytebuffer)
41. [Fs](#41-fs)
42. [Path](#42-path)
43. [Uri](#43-uri)

### How to Read the Examples

Unqualified `napalm` blocks are standalone programs: they can be saved to a
file with the `.napalm` extension. Variables in one block do not depend on
those in another.

- `napalm compile-error`: an example deliberately rejected at compile time.
- `napalm runtime-error`: an example deliberately rejected at runtime.
- `napalm fragment`: part of a program or module; follow the accompanying
  instructions.
- `// ...` comments show results or explain a detail; they are not a
  specification of JavaScript.

Tables use `x?` to indicate an **overload with fewer arguments** when one
exists in a native API. This does not mean Napalm supports default arguments
in function declarations.

## 1. Getting Started and Running Programs

### 1.1 A First Program

```napalm
const language = "Napalm";
function double(value: number): number {
    return value * 2;
}

System.print(`Hola, ${language}!`);
System.print(double(21)); // 42
```

From the repository root:

```powershell
cargo run --release -- .\scripts\complete.napalm
cargo run --release -- .\scripts\module_import.napalm
```

For your own file, for example `hello.napalm`:

```powershell
cargo run --release -- .\hello.napalm
```

If it has already been compiled:

```powershell
.\target\release\napalm.exe .\hello.napalm
```

The CLI accepts an entry file followed by optional program arguments:
`napalm program.napalm [arguments...]`. `Process.args` contains only the
arguments after the filename. The entry file can also be a compiled
`.burning` artifact (see [1.2](#12-building-and-distributing-programs)). This
implementation does not provide a REPL, `process.argv`, or an `-e` option.

| Command | Effect |
| --- | --- |
| `napalm program.napalm [arguments...]` | Compiles and runs the program. |
| `napalm program.burning [arguments...]` | Runs a program compiled with `--build`. |
| `napalm --build program.napalm` | Writes `program.burning` without running anything. |
| `napalm --compile program.napalm` | Writes `program.burning` and a standalone executable. |
| `napalm --compile program.burning` | Writes a standalone executable from an existing artifact. |
| `napalm --help` | Prints the usage summary (exit code 2). |

`scripts/process_args.napalm` prints these arguments as a JSON array and checks
that they are Strings. It accepts any argument list, including none (`[]`):

```powershell
cargo run --release -- scripts/process_args.napalm
cargo run --release -- scripts/process_args.napalm alpha "two words"
```

**Current extension:** use `.napalm`. Imports with an extension accept
`.napalm`; imports without an extension look for the file and directory index
with that extension. Older references to `.nap` or `.ts` in some texts do not
describe the current resolver.

### 1.2 Building and Distributing Programs

Napalm can compile a program ahead of time instead of compiling it on every
run. There are two output forms:

- **`--build`** writes a `.burning` artifact: the compiled program, including
  every module it imports. It runs with `napalm program.burning`.
- **`--compile`** also writes a **standalone executable** that contains the
  Napalm runtime and the program. It runs on its own, with no Napalm install.

#### `--build`: `.burning` artifacts

```powershell
napalm --build .\scripts\fib.napalm     # writes .\scripts\fib.burning
napalm .\scripts\fib.burning 30 extra   # Process.args is ["30", "extra"]
```

- The artifact is written beside the entry file, with the same name and a
  `.burning` extension. An existing artifact is replaced only if the build
  succeeds.
- Statically imported modules (`import ... from './x'`, `@std`, `@lib`) are
  compiled into the same artifact. They do not get their own `.burning` files.
- **No code runs during a build**: no module body, import or `static` block
  executes, and `System.print` produces no output. Compile errors are reported
  as in a normal run.
- `--build` must come **before** the entry file. In `napalm program.burning
  --build`, `--build` is an ordinary program argument.
- `--build` takes a source file. `napalm --build program.burning` is a usage
  error (exit code 2) and leaves the artifact untouched.

Running an artifact skips parsing and compiling. The original sources are no
longer needed: the artifact can be moved to another directory or machine.
Builtins, the JIT, error messages, `Process.args` and exit codes behave as
they do for source files.

The build uses extra whole-program optimizations that a normal run skips.
They assume the artifact contains every module that can run. For example, a
function called with constant arguments may be evaluated during the build:

```napalm
function fibonacci(n: number): number {
    return n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
System.print(fibonacci(25)); // 75025; with --build it may be computed during the build
```

These optimizations do not change what a program prints, returns or throws.
Side effects (output, I/O, timers, exceptions) and their order are preserved,
and source locations in error messages remain those of the original code.
Building takes longer than a normal run's compilation; running the artifact
is faster to start.

#### `--compile`: standalone executables

```powershell
napalm --compile .\app.napalm     # writes .\app.burning and .\app.exe
.\app.exe alpha "two words"       # Process.args is ["alpha", "two words"]
```

`--compile` builds the program exactly like `--build`, keeps the `.burning`
file, and also writes an executable beside the entry file: `app.exe` on
Windows, `app` (marked executable) on Linux and macOS.
`napalm --compile app.burning` turns an existing artifact into an executable
without needing the sources.

Running the executable runs the embedded program:

- **Every argument goes to the program.** The executable has no CLI options:
  `app.exe --help` puts `"--help"` in `Process.args`.
- Exit codes, error messages and output are the same as
  `napalm app.burning`.
- It can be renamed and moved freely. Neither the sources, the `.burning`
  file nor a Napalm installation are needed.
- Files, native libraries (FFI) and network resources that the program opens
  at runtime are still read from the system as usual. Relative paths are
  resolved against the current working directory, not the executable's.

**Dependencies.** The executable contains the interpreter, garbage collector,
builtins and both JIT tiers. On Windows it also contains the LLVM library
(`LLVM-C.dll`) and needs no Visual C++ Redistributable. Expect roughly 45 MB.
The LLVM library is unpacked automatically to
`%LOCALAPPDATA%\napalm\llvm\` the first time a program becomes hot enough to
use it. This happens once per machine and takes about a second. Executables
that embed the same LLVM library share the unpacked copy. Set
`NAPALM_CACHE_DIR` to use a different directory. If the LLVM library cannot be
found when compiling, `--compile` prints a warning and the executable still
works, but uses only the Cranelift JIT unless it finds LLVM at runtime.

On Linux and macOS the LLVM library is not embedded: install LLVM 23 or put
`libLLVM` next to the executable. Standalone executables are untested on macOS.
Code-signing a standalone executable (Authenticode on Windows, `codesign` on
macOS) is not supported.

An executable runs only with the runtime that produced it. Recompile after
updating Napalm. A `.burning` artifact is also tied to the Napalm version that
built it: a runtime with a different format rejects it with exit code 1, and
it must be rebuilt.

The `NAPALM_*` diagnostic environment variables (for example `NAPALM_NO_JIT`)
also affect standalone executables.

#### Build and compile errors

| Situation | Exit code |
| --- | --- |
| Success (nothing is printed, except the LLVM warning described above) | `0` |
| Syntax or type error in the program; invalid or incompatible `.burning` file | `1` |
| Missing entry file after `--build`/`--compile`; `--build` given a `.burning` file | `2` |
| Unreadable source; output file cannot be written | `3` |

A failed build or compile leaves any existing `.burning` file and executable
untouched.

### 1.3 Process Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | The program completed successfully. |
| `1` | Compilation error, including syntax errors or statically checked types; invalid or incompatible `.burning` artifact or standalone executable. |
| `2` | Incorrect CLI usage; this is also the current exit code for `--help`. |
| `3` | I/O error while loading sources or artifacts, writing build output, or emitting diagnostics. |
| `4` | Uncaught runtime error or a module suspension that was not resumed. |

Standalone executables and `.burning` artifacts use the same codes.

An open HTTP listener, a periodic timer, or pending asynchronous work can keep
the process alive after the module body has run. Close or cancel resources that
are no longer needed.

### 1.4 Process and Standard Streams

`Process.cwd()`, `Process.chdir(path)` and `Process.getenv(name)` expose the
working directory and environment; a missing environment variable returns
`null`. `Process.exit(code)` immediately exits the host process with the given
integer status, overriding the normal CLI exit codes above.

`Process.stdin` is an `InputStream`; `Process.stdout` and `Process.stderr` are
distinct `OutputStream` singletons. Their operations are synchronous:

| Operation | Result |
| --- | --- |
| `stdin.read(count)` | Up to `count` bytes, or `null` at EOF. |
| `stdin.readExact(count)` | Exactly `count` bytes; throws if EOF arrives early. |
| `stdin.readLine()` | A UTF-8 string without LF/CRLF, or `null` at EOF; an empty line is `""`. |
| `stdout.write(bytes)` / `stderr.write(bytes)` | Number of bytes written; accepts `Bytes` or `ByteBuffer`. |
| `stream.writeAll(bytes)` | Writes all bytes and returns their count. |
| `stream.writeText(text)` | Writes a UTF-8 String and returns its byte count. |
| `stream.flush()` | Flushes output and returns `null`. |
| `stream.close()` | Returns `null`; standard streams remain usable. |

Input reads wait for data or EOF. Zero-byte reads return empty `Bytes` without
waiting. Binary reads accept integer counts from 0 through 256 MiB; lines are
limited to 16 MiB. Invalid types, lengths, UTF-8 and I/O errors throw catchable
runtime errors.

`scripts/process_complete.napalm` runs self-contained checks by default and
exits successfully without reading stdin. Its optional `lsp`, `line`, `eof`
and `binary` modes require specific input; `exit` deliberately returns status
17. The LSP mode reads one Content-Length frame and emits only its framed
JSON response. Keep diagnostics on stderr when using protocol stdout.

Run every mode with controlled binary input, expected output/status checks
and timeouts:

```powershell
cargo build --release
python tools/verify_process_integration.py
```

## 2. Basic Syntax and Literals

### 2.1 Statements, Blocks, and Comments

Semicolons and statements separated by line breaks are supported. Blocks use
braces.

```napalm
// Comment through the end of the line.
let x = 1;
/* Block comment. */
{
    const increment = 2;
    x += increment;
}
System.print(x); // 3
```

The public parser's `/* ... */` comments **do not nest**. Ordinary identifiers
use ASCII letters, digits, `_`, and `$`; they cannot start with a digit.
Strings do support Unicode.

Do not rely on all of ECMAScript's automatic semicolon insertion rules. In
particular:

- `return` followed immediately by a line break returns `null`.
- The expression in a `throw` statement must start on the same line.
- Postfix `++` and `--` must stay next to their operand.
- A continuation with `.`, `?.`, or a binary operator may cross a line break.
- To return an object from an expression-bodied arrow, use `() => ({x: 1})`.

### 2.2 Numbers

```napalm
const decimal = 1_000_000;
const fractional = .25;
const exponent = 1.5e2;
const hex = 0xff;
const binary = 0b1010;
const octal = 0o17;
System.print([decimal, fractional, exponent, hex, binary, octal]);
// [1000000, 0.25, 150, 255, 10, 15]
```

Separators `_` can appear between digits, including in bases and exponents.
The minus sign is an operator, not part of a literal.
Language numbers use double-precision floating-point semantics; there is no
`BigInt` literal or `n` suffix.

### 2.3 Booleans and null

The literals are `true`, `false`, and `null`. **There is no `undefined`**.
Missing data, list holes, and returns without a value use `null`, although a
non-nullable annotation may reject it.

### 2.4 Strings and Templates

```napalm
const single = 'texto';
const double = "otro texto";
const name = "Ada";
System.print(`${name}: ${6 * 7}`);
System.print(`anidada: ${`valor ${2 + 3}`}`);
System.print(`primera linea
segunda linea`);
System.print("50%"); // A percent sign does not start interpolation.
System.print("\u{1f600}");
```

Only backticks interpolate `${expression}` and allow literal line breaks.
Single and double quotes cannot contain a line break without escaping it.

| Escape | Value |
| --- | --- |
| `\n`, `\r`, `\t` | Newline, carriage return, and tab. |
| `\0`, `\b`, `\f`, `\v` | NUL, backspace, form feed, and vertical tab. |
| `\\`, `\'`, `\"` | Backslash and quotes. |
| `` \` ``, `\$` | Backtick and literal dollar sign. |
| `\xHH` | Short hexadecimal code. |
| `\uHHHH`, `\u{...}` | Unicode code point. |
| Backslash followed by a line break | Continuation without adding that break to the value. |

Unknown escapes and invalid Unicode codes are rejected.
Do not assume a pair of UTF-16 surrogate escapes is automatically treated as
a single character; use `\u{1f600}`, for example.

## 3. Values, Identity, and Conversions

### 3.1 Value Families

| Family | Examples |
| --- | --- |
| Scalars | `null`, `true`, `42`, `"texto"`. |
| Collections | List/Array, Map/object literal, Set. |
| Ranges | `new Range(0, 5)`. |
| Functions | Declarations, `function` expressions, arrows, and bar lambdas. |
| User objects | Classes and their instances. |
| Async control | Promise and Fiber. |
| Reactive/event endpoints | Event, Emitter, Channel, and Receiver. |
| Native library | Date, Random, Proxy, Response, Headers, and FFI objects. |

Assigning a list, Map, or instance to another variable copies the **reference**,
not its contents. `const` does not change this rule.

```napalm
const original = {value: 1};
const alias = original;
alias.value = 2;
System.print(original.value); // 2
System.print(original === alias); // true
System.print(original === {value: 2}); // false
```

### 3.2 Conditions and Falsy Values

In conditions, ternaries, `&&`, and `||`, **only `false` and `null` are falsy**.
`0`, `""`, `Num.nan`, `[]`, and `{}` are truthy.

```napalm
System.print(0 ? "verdadero" : "falso");  // true
System.print("" ? "verdadero" : "falso"); // true
System.print(null || "alternativa");     // alternativa
System.print(0 || 99);                   // 0
System.print(false ?? 99);               // false
System.print(null ?? 99);                // 99
```

`&&` and `||` return an operand and short-circuit; they do not necessarily
convert the result to a boolean. `??` replaces only `null`.

**Caution with `!`:** it is a dispatch operation, not a universal conversion
to boolean. It works with Bool and Null, and classes can overload it.
For example, `!0` currently produces an undefined-method error.
To convert an arbitrary condition to Bool, use `value ? true : false`; to
negate it, use `value ? false : true`.

### 3.3 Equality

`==` and `===` represent the same operation without coercion. The same applies
to `!=` and `!==`. There is no JavaScript-style loose equality.

- Numbers: numeric comparison; `Num.nan == Num.nan` is false.
- Strings: comparison by contents.
- Booleans and null: comparison of their values.
- Ordinary collections and objects: identity, not deep equality.
- Instances with equality operators: may define their own comparison.

There is a deliberate difference in **Map/Set key** equality: all NaN values
are grouped as one key; `0` and `-0` are also grouped together.

### 3.4 Conversion to Text

Use interpolation or `value.toString()` for values that implement that
protocol. Many builtins also retain the `value.toString` getter.
Classes can implement `toString()` and participate in interpolation,
concatenation, and `System.print`.

Do not use `String(value)`, `Number(value)`, or `Boolean(value)` by analogy
with JavaScript: equivalent conversion APIs are not provided here. The textual
output of Maps and lists is for presentation, **not JSON**. Use
[`JSON.stringify(value)`](#38-json) to serialize supported values as JSON.

### 3.5 typeof

`typeof value` returns a String naming the runtime kind of a value. Values that
carry generic or contract information include it:

| Value | `typeof` |
| --- | --- |
| `1`, `true`, `null`, `"s"` | `Number`, `Boolean`, `Null`, `String` |
| `[1, 2]` typed `number[]` / untyped list | `Array<Number>` / `Array` |
| `new Map()`, typed `Map<string, number>` | `Map`, `Map<String, Number>` |
| `{}` / object bound to interface or type `Named` | `Object` / `Object implementation of Named` |
| `new Set()`, typed `Set<number>` | `Set`, `Set<Number>` |
| interface name / type alias name | `Interface` / `Type` |
| class object | `Class` |
| instance of `Dog`; of `Holder<number>` | `Instance of Dog`; `Instance of Holder<Number>` |
| function or closure | `Function` |
| Promise; pending `Timer.delay`/`Timer.every` | `Promise`; `Timer` |
| `new Emitter<number>()`, its `.event` | `Emitter<Number>`, `Event<Number>` |
| Channel / Receiver | `Channel`, `Receiver` (with `<T>` when typed) |
| `Fs.stat(path)`, `Fs.open(path, mode)` | `FileInfo`, `File` |
| Date, Proxy, Range, Fiber, Generator, Module | `Date`, `Proxy`, `Range`, `Fiber`, `Generator`, `Module` |
| native decorator | `NativeDecorator` |

A settled timer Promise reports `Promise`. User decorators are ordinary
functions (`Function`).

## 4. Variables, Scopes, and Constants

### 4.1 Declarations

```napalm
let count = 0;
var legacyName = "tambien tiene ambito de bloque";
const limit = 3;
let left = 1, right = 2;
count += left + right;
System.print([count, limit, legacyName]);
```

`let`, `var`, and `const` are lexical and have block scope. `var` does not gain
JavaScript's function-scoping rules.
**All variables require an initializer**, not just `const`.
Declared class properties are the exception.

```napalm compile-error
let pending;
```

A local variable can shadow an outer one; the same name cannot be redeclared
in the same scope.

```napalm
const value = 1;
{
    let value = 2;
    value++;
    System.print(value); // 3
}
System.print(value); // 1
```

### 4.2 const and Mutability

`const` prevents replacing a binding's value, including through closures,
compound assignments, or destructuring. It does not freeze an object.

```napalm
const items = [1];
items.add(2);
items[0] = 9;
System.print(items); // [9, 2]
// items = []; // Error: reassignment of a constant.
```

### 4.3 Inference

The absence of an annotation **does not necessarily mean `any`**.
The compiler retains types inferred from initializers when it can determine
them.

```napalm compile-error
let amount = 10;
amount = "diez";
```

To intentionally mix value families:

```napalm
let value: any = 10;
value = "diez";
value = null;
System.print(value);

let selected: number | null = null;
selected = 42;
System.print(selected);
```

This distinction matters especially when initializing a handle with `null`:
declare it as `any` or with an appropriate union if it will later hold another
type.

## 5. Operators and Precedence

### 5.1 Reference

| Category | Operators |
| --- | --- |
| Access and call | `.`, `[]`, `()`, `?.`, `?.[]`, `?.()`. |
| Update | `++x`, `--x`, `x++`, `x--`. |
| Unary | `-`, `+`, `!`, `~`, `await`, `delete`, `launch` in its own syntax. |
| Exponentiation | `**`, right-associative. |
| Multiplicative | `*`, `/`, `%`. |
| Additive | `+`, `-`. |
| Shifts | `<<`, `>>`. |
| Comparison | `<`, `<=`, `>`, `>=`, `in`, `instanceof`, `implements`, `inherits` (see §14.7). |
| Equality | `==`, `!=`, `===`, `!==`. |
| Bitwise | `&`, `^`, `|`. |
| Logical | `&&`, `||`, `??`. |
| Pipeline | `|>`; thread-first and left-associative. |
| Conditional | `condition ? yes : no`. |
| Assignment | `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `**=`, `&&=`, `||=`, `??=`. |

From highest to lowest, the parser gives precedence to member access/calls,
updates, prefixes, exponentiation, multiplication, addition, shifts,
comparisons, equality, `&`, `^`, `|`, `&&`, `||`/`??`, `|>`, the ternary
operator, and assignment. Use parentheses when mixing logical, nullish,
pipeline, or conditional expressions when the intended grouping is not obvious.

Two differences are especially easy to miss:

- `-2 ** 2` is currently parsed as `(-2) ** 2`, yielding `4`.
  Write `-(2 ** 2)` to get `-4`.
- Unary `+` is removed: it **does not convert** strings to numbers.
  `+"12"` remains `"12"`.

### 5.2 Arithmetic and Concatenation

`+` adds numbers, concatenates strings, and concatenates lists.
A string can be combined with another value through the `toString()` protocol;
a class can define its own `operator +`.
The other numeric operators do not automatically convert strings.

```napalm
System.print(2 ** 3 ** 2); // 512
System.print(7 / 2);       // 3.5
System.print(-7 % 3);      // -1
System.print("valor: " + 42);
System.print([1, 2] + [3]); // [1, 2, 3]
```

### 5.3 Bitwise Operators

Bitwise operators work on **unsigned** 32-bit values.
`>>` is a logical shift; it is not JS's arithmetic shift.
`>>>` is not implemented.

```napalm
System.print(~0);       // 4294967295
System.print(-1 >> 1);  // 2147483647
System.print(1 << 4);   // 16
System.print(10 & 3);   // 2
```

Numeric operands are converted to the 32-bit domain; the shift amount is reduced
to its five lowest bits. Do not use these operators for exact 64-bit integer
arithmetic.

### 5.4 Short-Circuiting and Optional Access

```napalm
let selected: any = null;
selected ??= {value: 7};
selected.value *= 2;
System.print(selected.value); // 14

let skipped = 0;
const absent = null;
System.print(absent?.child?.[skipped++]); // null
System.print(absent?.(++skipped));        // null
System.print(skipped);                    // 0

const object = {double: (n) => n * 2};
System.print(object?.double(4)); // 8
```

`&&=`, `||=`, and `??=` evaluate the right-hand side only when appropriate. The
receiver and key of a compound assignment are evaluated only once.
`?.` short-circuits on `null`; it does not protect against a missing method on a
non-null object or turn a missing key into a valid API.

### 5.5 Thread-First Pipeline `|>`

The pipeline operator passes the value on its left as the **first argument** to
the callable or call expression on its right.

```napalm
function double(value: number): number {
    return value * 2;
}

function add(value: number, amount: number): number {
    return value + amount;
}

System.print(21 |> double);          // double(21) -> 42
System.print(20 |> add(2));          // add(20, 2) -> 22
System.print(5 |> double |> add(1)); // add(double(5), 1) -> 11
```

A bare callable:

```napalm
value |> transform
```

is equivalent in result to:

```napalm
transform(value)
```

A call target keeps its existing arguments after the piped value:

```napalm
value |> transform(a, b)
```

is equivalent in result to:

```napalm
transform(value, a, b)
```

The same thread-first rule applies to method calls:

```napalm
value |> object.transform(a)
```

passes `value` as the first explicit method argument:

```napalm
object.transform(value, a)
```

Call spread is supported on the right:

```napalm
value |> transform(prefix, ...rest)
```

and behaves like `transform(value, prefix, ...rest)`, subject to the normal
maximum of 16 expanded call arguments.

Pipelines are **left-associative**:

```napalm
source
    |> normalize
    |> validate
    |> encode
```

means:

```napalm
encode(validate(normalize(source)))
```

A line break may precede `|>`, which makes vertical pipelines convenient.

The left input is evaluated **exactly once and before the pipeline target**.
This is stronger than a purely textual rewrite. For example, if both
`makeValue()` and `makeTransform()` have side effects:

```napalm
makeValue() |> makeTransform()
```

evaluates `makeValue()` first, then obtains/calls the target.

`|>` binds more weakly than `||`, `??`, and the other ordinary binary
operators, but more strongly than `?:` and assignment. Parenthesize a complete
conditional when piping its result:

```napalm
const selected = (enabled ? primary : fallback) |> normalize;
```

The pipeline is syntax-owned and **cannot be overloaded** with `operator`.
After frontend lowering it becomes the same ordinary call IR used by normal
calls, so there is no separate runtime pipeline object or dispatch protocol.

There is no placeholder syntax attached to `|>`. `_` remains an ordinary
identifier. To place the piped value anywhere other than the first argument,
use a lambda:

```napalm
value |> (x => transform(prefix, x))
```

This form is also useful when the right-hand operation needs a more complex
expression than a simple thread-first call.

## 6. Control Flow

### 6.1 if, while, and do...while

```napalm
let n = 0;
while (n < 3) {
    n++;
}

do {
    n--;
} while (n > 1);

if (n === 1) {
    System.print("uno");
} else if (n === 2) {
    System.print("dos");
} else {
    System.print("otro");
}
```

`do...while` executes its body at least once. `if` and `while` use Napalm's
truthiness rules, not JS's.

### 6.2 Classic for

```napalm
let total = 0;
for (let i = 0; i < 10; i++) {
    if (i === 3) continue;
    if (i === 8) break;
    total += i;
}
System.print(total); // 25
```

All three parts of a `for` loop may be omitted, for example `for (;;)` .
`continue` proceeds through the `for` increment; `break` ends the loop.
There are no `break label`/`continue label` labels.

**Closures:** a classic `for` binding is not recreated for each iteration as
it is in JavaScript with `let`. To capture a different value each time, pass it
to a function that creates the closure.

### 6.3 for...of and for...in

```napalm
for (const value of [10, 20]) {
    System.print(value);
}

const record = {b: 2, a: 1};
for (const key in record) {
    System.print(`${key}: ${record[key]}`);
}

for (const entry of record) {
    System.print([entry.key, entry.value]);
}
```

- `for (const x of collection)` traverses the `iterate`/`iteratorValue` protocol.
- `for (const key in map)` traverses **the `map.keys` getter**.
- The loop binding is declared with `let`, `var`, or `const`.
- Destructuring patterns are supported in the binding.
- `for...in` does not implement prototype enumeration or array indices.
  The receiver must provide a `keys` getter.
- Leaving a `for...of` over a generator early (`break`, `return`, or an
  exception) closes the generator ([section 22.2](#222-generators)).

There is no `switch` or `for await...of` in this version.

## 7. Functions and Closures

### 7.1 Function Declaration Forms

```napalm
function add(a, b) { return a + b; }
const multiply = function (a, b) { return a * b; };
const increment = (x) => x + 1;
const square = x => x * x;
const compact = |x, y| x + y;
const zero = | | 0;
const objectFactory = () => ({answer: 42});

System.print([add(2, 3), multiply(2, 3), increment(3), square(4)]);
System.print([compact(1, 2), zero(), objectFactory().answer]);
```

The declaration keyword is **`function`**, not `fn`.
The `|x| expression` and `|x| { statements }` lambda forms are supported.
For zero parameters with bars, write `| |`, not `||`, which is OR.
Destructured parameters are best written using the parenthesized
`function` or arrow form.

### 7.2 Return Values

A function with a block body returns a value only with `return`.
Falling through to the end or writing `return;` returns `null`.
An expression-bodied arrow or lambda implicitly returns its expression.

```napalm
function noReturn() { 42; }
const explicit = () => { return 42; };
const implicit = () => 42;
System.print([noReturn(), explicit(), implicit()]); // [null, 42, 42]
```

### 7.3 Lexical Scope and Recursion

```napalm
function counter(start) {
    let current = start;
    return () => ++current;
}
const next = counter(40);
System.print(next()); // 41
System.print(next()); // 42

function factorial(n: number): number {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
System.print(factorial(5)); // 120
```

Closures retain captured bindings and observe changes to them.
`this` is lexical too: a closure created inside a method retains that method's
receiver.

**There is no function hoisting:** the declaration installs the function when
executed. Declare functions before calling code that uses them.
Recursion is possible once the binding has been installed.

### 7.4 Calls, Arity, and Missing Parameters

Functions are called with `f(...)`; `f.call(...)` is also available.
`f.arity` returns the number of declared parameters. `Fn.new(f)` preserves
a closure as a function.

```napalm
function pair(a, b) { return [a, b]; }
System.print(pair.arity);      // 2
System.print(pair(1, 2));      // [1, 2]
System.print(pair(1));         // [1, null]
System.print(pair(1, 2, 3));   // [1, 2]
System.print(pair.call(4, 5)); // [4, 5]
```

An ordinary closure call fills missing parameters with `null` and ignores
excess arguments, up to the global limit of **16 arguments**. This does not
implement optional or rest parameters:

- A `b?: string` parameter retains the `string` annotation. Omitting it passes
  `null` and may cause a fatal type error.
- An unannotated parameter can receive that `null`.
- Methods and constructors are looked up by **name and arity**; do not assume
  they use the same rules as a closure.
- `launch` requires exactly the declared arity.
- There is no `function f(...args)` or `function f(x = 1)`.
- If default values are needed, use a check in the body or a destructured
  parameter with defaults inside the pattern.

For a function that explicitly accepts absence:

```napalm
function label(name: string, suffix?: string | null) {
    return name + (suffix ?? "");
}
System.print(label("Napalm"));      // Napalm
System.print(label("Napalm", "!")); // Napalm!
```

`call` **does not accept a `thisArg`** like `Function.prototype.call`.
There is no `apply`, `bind`, or standard `arguments` object.

## 8. Types and Persistent Contracts

### 8.1 What Is Actually Checked

Napalm combines inference, static diagnostics, and runtime validation.
It is not accurate to say that all its annotations are erased.

| Surface | Current effect |
| --- | --- |
| Annotated variables or variables with known inferred types | Check assignments; a dynamic source is validated at runtime. |
| Annotated parameters | Validated on function entry, including structural contracts. |
| Annotated or inferred class fields | Constrain writes and reads of uninitialized fields. |
| Interfaces, structural aliases, typed lists, generic Map/Set | Can install persistent contracts on objects. |
| Return annotation | Accepted, but is not a return-value contract. |
| `class C implements I` | Adds the fields of `I` to `C` and requires the required ones to be initialized (§14.7). |
| Visibility modifiers, `override` | Do not by themselves implement access control. |
| `as T`, `<T>value`, `satisfies T`, `value!` | Syntax accepted without converting or checking the value at runtime. |
| Generic interfaces, types and classes | Each concrete use (`Box<number>`, `Holder<string>`) is checked; the bare generic name is unconstrained (§8.8). |
| Generic functions | Type parameters are erased to `any`. |

A known violation can prevent compilation. A runtime contract violation
produces **`Structural type error`** and terminates the program.
**It cannot be recovered with `try/catch` or `Fiber.try`.**

```napalm runtime-error
function needsNumber(value: number) {
    return value;
}
try {
    needsNumber("no es un numero");
} catch (error) {
    System.print("este catch no recupera una infraccion de tipo");
}
```

By contrast, a return annotation does not enforce the result:

```napalm
function documentedReturn(): number {
    return "texto";
}
System.print(documentedReturn()); // texto
```

To validate a dynamic return, assign the result to an annotated variable.

### 8.2 Recognized Types

| Form | Meaning |
| --- | --- |
| `number`, `Number`, `num`, `Num` | Number. |
| `string`, `String` | String. |
| `boolean`, `Boolean`, `bool`, `Bool` | Boolean. |
| `null` | Null only. |
| `any` | No type restriction on the binding. |
| `never` | No value can satisfy it. |
| `T[]`, `Array<T>`, `List<T>` | List of T elements. |
| `Map<K, V>` | Map with key and value contracts. |
| `Set<T>` | Set with an element contract. |
| `Event<T>`, `Emitter<T>` | Typed event subscription and publication endpoints. Without `<T>`, payloads are `any`. |
| `Channel<T>`, `Receiver<T>` | Typed FIFO send/receive endpoints. Without `<T>`, payloads are `any`. |
| `A \| B` | Union of alternatives. |
| `(x: T) => U`, `Function`, `Fn` | Closure value; this annotation does not validate a complete signature. |
| Interface/alias name | Schema declared in the same module. |
| Known class name | Class identity, not a literal object with the same fields. |
| `{ x: number; }` | Inline structural schema. |
| `Range`, `Fiber`, `Map`, `Set`, `List` | Native families recognized by the type resolver. |
| `FileInfo`, `File` | Native filesystem metadata and open file handles. |

Additional syntax is accepted as type information, but it is not equivalent
to TypeScript's type system:

- Type literals are widened: `"yes"` constrains to string, not just `"yes"`;
  `42` constrains to number.
- `[number, string]` is treated as a list of the union of its elements, not as
  a fixed-length tuple with distinct positions.
- `ReadonlyArray`, `ReadonlyMap`, and `ReadonlySet` retain content types but do
  not make the collection immutable.
- `Record<K, V>` is classified as a Map; it does not replace the contract of
  `Map<K, V>`.
- Unresolved type names in some annotations may degrade to `any`. In declared
  interfaces and aliases, unsupported or unresolved types are diagnosed.
- Writing `Promise<T>` or `Date` as an annotation does not by itself install a
  complete contract on those native APIs.

### 8.3 Interfaces: Open Objects

```napalm
interface Position {
    x: number;
    y: number;
}
interface Named {
    name: string;
}
interface Player extends Named {
    readonly id: number;
    position: Position;
    scores: number[];
    nickname?: string;
}

const raw = {
    id: 1,
    name: "Ada",
    position: {x: 0, y: 0},
    scores: [10],
    extra: true
};
const player: Player = raw;
player.position.x += 1;
player.scores.add(20);
player.nickname = "A";
delete player.nickname;
player.other = 7; // An interface allows additional fields.
System.print([player.position.x, player.scores, player.nickname]);
```

An interface requires its declared fields but allows additional ones.
Its declarations have module scope and can refer to other declarations that
appear later, provided they do not form recursive schemas.

- `extends A, B` combines multiple compatible schemas.
- Duplicate fields and incompatible inherited definitions are rejected.
- `field?: T` allows **absence**, not an explicit `null` unless T includes it.
- Reading an absent optional property returns `null`.
- A required field cannot be deleted or removed with `clear()`.
- A `readonly` field cannot be changed or deleted after the contract is
  installed. The protection is shallow: a contained object can still be
  mutable.
- Interfaces and object types describe **data only**. A method signature such
  as `area(): number;` (or `map<T>(value: T): T;`) is a compile error:
  declare a class when you need methods. A function-typed field such as
  `onDone: () => void;` is still allowed; it is a closure field whose arity
  and return value are not checked.
- An interface- or type-annotated binding (`const p: Player = ...`, a
  `p: Player` parameter, a `Player[]` element) accepts a Map/object literal
  that satisfies the contract, or an instance of a class that declares
  `implements Player` (directly, through an ancestor, or through an interface
  that extends `Player`). Class instances are matched nominally: a class with
  the right fields but no `implements` clause is rejected (§14.7).

### 8.4 type: Closed Objects

Aliases **do have an effect** in this implementation. A structural object
alias produces a **closed** contract: additional keys are not allowed,
either initially or later.

```napalm
type Point = {x: number; y: number};
type MaybePoint = Point | null;
type Names = string[];

const point: Point = {x: 1, y: 2};
point.x = 3;
const names: Names = ["Ada"];
names.add("Lin");
System.print([point.x, names]);
```

```napalm compile-error
type Point = {x: number};
const point: Point = {x: 1};
point.extra = 2;
```

`type Closed = SomeInterface` closes the object bound as `Closed`.
It does not recursively change the semantics of a child declared with another
open interface. Structural alternatives in an alias union are also closed
where applicable.

Generic aliases substitute their type arguments (§8.8):
`type Box<T> = {value: T}` bound as `Box<number>` requires a Number `value`.

### 8.5 The Contract Belongs to the Object

Contracts do not belong only to the variable that installed them:

```napalm runtime-error
interface Counter { value: number; }
const original = {value: 1};
const checked: Counter = original;
const alias: any = original;
alias.value = "invalido";
```

Mutations through properties, brackets, collection methods, closures, and
aliases created before the contract are checked. An `any` alias does not remove
the protection.

Other important effects:

1. Existing contents are validated before the contract is installed.
2. Contracts for nested objects and lists propagate to those references.
3. Replacing a child validates and protects the new child.
4. Removing a reference does not unprotect the object it used to contain.
5. If an object receives multiple contracts, it must satisfy all of them.
6. For unions of mutable objects/lists, the first alternative that matched
   when bound is retained.
7. A shallow spread copy creates another container but shares children that
   remain subject to their contracts.

### 8.6 Class Fields and Exact Types

A `value: number;` field can be left without an initializer in the
declaration, but reading it before assigning a number produces an error; this
does not permit treating it as `number | null`.

Class contracts are based on known concrete classes, not an entire structural
hierarchy. Do not assume a `Base` annotation automatically accepts every
`Derived` instance, even when `derived instanceof Base` is true.

### 8.7 Type Syntax Without Executable Code

Among other forms, the following are accepted:

- Generic parameters on functions and type arguments in calls (erased; generic
  interfaces, types and classes are checked, see §8.8).
- Overload signatures without bodies, followed by the implementation.
- `abstract class`, `abstract` or `declare` members.
- `declare function`, `declare const/let/var`, `declare class`,
  `declare namespace`, and `declare module`.
- `import type` and type exports.
- `name!` and `name?` properties.

`declare` declarations do not create values or load libraries.
Signatures without bodies do not create an executable overload.
`abstract` should not be interpreted as a complete guarantee that the class
cannot be instantiated. Type imports do not carry contracts between modules.

### 8.8 Generics

Generic interfaces, object types and classes are checked per concrete use. Using
the generic name without arguments keeps the old unconstrained behavior: every
type parameter is `any`.

```napalm
interface Box<T> { value: T; }
type Pair<A, B> = { first: A; second: B };
class Holder<T> {
    item: T;
    constructor(item: T) { this.item = item; }
}

const box: Box<number> = { value: 1 };        // value must be a Number
const pair: Pair<string, number[]> = { first: "a", second: [1] };
const loose: Box = { value: "anything" };      // unconstrained

const holder = new Holder<number>(1);          // this instance holds Numbers
holder.item = 2;                               // checked against Number
const free = new Holder(1);                    // no arguments: unconstrained
free.item = "text";
```

- `Box<number>` and `Pair<string, number[]>` become ordinary contracts with the
  arguments substituted, so nested contents (`number[]`) are protected too.
- A generic class instance created as `new Holder<number>(...)` or bound to a
  `Holder<number>` annotation carries its arguments. Every later write to a
  field declared with a type parameter (`item: T`) is checked against them,
  including writes from methods and from optimized native code.
- `Promise<T>`: the settled value is checked against `T` at `await`.
- `Map<K, V>`, `Set<T>`, `T[]`/`Array<T>`, `Event<T>`, `Emitter<T>`,
  `Channel<T>` and `Receiver<T>` work as before (§10.5, §15.3).
- Recursive generic contracts (`next: Node<T>`) are not supported, like
  recursive interfaces in general. Type parameters of generic functions are
  erased to `any`.

`typeof` reports generic arguments carried by a value (§3.5).

## 9. Lists and Arrays

### 9.1 Creating Lists

```napalm
const literal = [1, 2, 3];
const sparse = [1, , 3];
const slots = Array(3);
const otherSlots = new Array(2);
const zeros = new Array<Number>(3);
const filled = List.filled(3, "x");
System.print([literal, sparse, slots, otherSlots, zeros, filled]);
```

- `[]` is an empty list.
- Holes and `Array(n)` are filled with `null`.
- `new Array<Number>(n)` and `Array<number>(n)` are filled with zero and install
  a numeric contract.
- Array constructors with a size require exactly one argument.
- Only `Number`/`number`/`Num` and `any` are supported as types for these
  generic sized constructors; not `new Array<string>(3)`.
- `new List(n)` and `List.new(n)` expose the same native list with null slots.
- `List.filled(n, value)` repeats **the same reference** if value is an object.

Array and List are not separate collection identities. `Array` is syntactic
sugar for construction, not a replacement for all JavaScript `Array` APIs.

### 9.2 Indices, Growth, and length

```napalm
const values: any[] = [10, 20, 30];
System.print(values[-1]); // 30
System.print(values[99]); // null
values[5] = 60;
System.print(values);     // [10, 20, 30, null, null, 60]
values.setLength(2);
System.print(values);     // [10, 20]
values.setLength(4);
System.print(values);     // [10, 20, null, null]
```

Indices are integers. A negative index counts from the end.
An out-of-range read returns `null`; an excessively negative write fails.
Writing to a positive index beyond the current size expands the list and fills
the gap with null.

Growth of a typed list also validates those null values. Therefore, a
`number[]` list cannot grow through `setLength` or an out-of-range write if
that operation introduces null; `length` is read-only. Use
`add`/`push`/`insert` to add numbers.

### 9.3 List API

| Member | Result and effect |
| --- | --- |
| `count`, `length` | Number of elements. |
| `length()` | Equivalent historical form. |
| `isEmpty` | Bool; use as a property. |
| `add(value)` | Adds an element and returns that element. |
| `push(value)` | Adds an element and returns the new size. |
| `pop()` | Removes the last element; null if empty. |
| `shift()` | Removes the first element; null if empty. |
| `unshift(value)` | Inserts at the beginning and returns the new size. |
| `insert(index, value)` | Inserts at a position and returns value; `-1` inserts at the end. |
| `removeAt(index)` | Removes and returns the element; supports valid negative indices. |
| `remove(value)` | Removes the first `==` match; returns the element or null. |
| `clear()` | Empties the list and returns null. |
| `setLength(n)` | Truncates or grows with null; returns n. |
| `fill(value)` | Replaces all elements and returns the list. |
| `map(callback)` | New list; calls `callback(element)` once per element. |
| `filter(callback)` | New list with elements whose condition is truthy. |
| `reduce(callback, initial)` | Accumulates through `callback(accumulator, element)`. |
| `+ otherList` | New shallow concatenation. |
| `[range]` | New list with the selected segment. |
| `iterate(cursor)`, `iteratorValue(cursor)` | Iteration protocol. |
| `toString`, `toString()` | Text representation. |

`push`, `add`, and `unshift` accept **a single element**, not a variable
number of elements. map/filter callbacks do not automatically receive an
index or collection. `reduce` requires an initial value.

```napalm
const source = [1, 2, 3, 4];
System.print(source.map(x => x * 2));       // [2, 4, 6, 8]
System.print(source.filter(x => x > 2));    // [3, 4]
System.print(source.reduce((a, x) => a + x, 0)); // 10
System.print(source);                      // [1, 2, 3, 4]
```

`forEach`, `find`, `findIndex`, `sort`, `splice`, and the complete JS Array
catalog are not installed. Use a loop or compose the available operations.
Do not modify the source list inside callbacks while assuming ECMAScript's
exact rules.

### 9.4 Shallow Copies and Limits

`[...list]`, `list + other`, and slices create a new container but retain
inner references.

Sized construction, `List.filled`, `setLength`, and index-based growth have a
size limit of **16,777,216** elements.
Do not interpret this limit as a promise of available memory or as a uniform
quota for all list operations.

## 10. Objects, Map, and Set

### 10.1 Object Literals Are Maps

```napalm
const name = "Ada";
const key = "score";
const person = {
    name,
    [key]: 42,
    "display-name": "A",
    double: (n) => n * 2
};
System.print(person.name);
System.print(person["display-name"]);
System.print(person.double(21));
person.score++;
System.print("score" in person); // true
delete person.score;
System.print(person.score);      // null
```

Named, string, and numeric keys, computed keys `[expression]`, and the
`{name}` shorthand are supported.
To store a function, use `method: (...) => ...` or
`method: function (...) {...}`; do not assume support for JS object method
shorthand syntax.

`obj.key` resolves as a key when a builtin member does not take precedence.
**Bracket notation always accesses the key.**

```napalm
const obj = {size: "dato", count: "otro", keys: "contenido"};
System.print(obj.size);     // 3: Map property
System.print(obj["size"]); // dato
System.print(obj["keys"]); // contenido
```

Functions stored in Maps can be called with `obj.fn(...)`.
They do not automatically receive the Map as `this`; they remain lexical
closures.

### 10.2 Map API

```napalm
const scores = new Map<string, number>();
scores.set("Ada", 42).set("Lin", 7);
System.print(scores.get("Ada"));
System.print(scores.has("Lin"));
System.print(scores.size);
scores.delete("Lin");
```

| Member | Result |
| --- | --- |
| `new Map()` | Empty Map; does not accept an initial iterable. |
| `get(key)`, `[key]` | Value or null if absent. |
| `set(key, value)` | Inserts/updates and returns the Map. |
| `set_if_not_exists(key, value)` | Returns the existing value (including null), or inserts and returns `value`. |
| `[key] = value` | Writes an entry. |
| `has(key)`, `containsKey(key)`, `key in map` | Whether the key exists, even if its value is null. |
| `delete(key)`, `delete map.key`, `delete map[key]` | Bool indicating whether it existed. |
| `remove(key)` | Returns the removed value or null. |
| `clear()` | Empties the Map; returns null; preserves contracts. |
| `size`, `count` | Number of entries. |
| `isEmpty` | Bool. |
| `keys` | Live iterable view of keys in insertion order. |
| `keys()` | New list of keys sorted by their textual representation. |
| `values()` | New list of values, in the same order as `keys()`. |
| `entries()` | New list of `[key, value]` pairs, in that order. |
| `iterate`, `iteratorValue` | Iteration in insertion order. |
| `toString`, `toString()` | Presentation representation, not JSON. |

**`map.keys` and `map.keys()` are not equivalent.**
This distinction also affects `for...in`.

```napalm
const map = {z: 1, a: 2};
const live = map.keys;
const snapshot = map.keys();
map.m = 3;
System.print([...live]);       // [z, a, m]
System.print(snapshot);        // [a, z]
System.print(map.entries());   // [[a, 2], [m, 3], [z, 1]]
for (const entry of map) {
    System.print([entry.key, entry.value]); // z, a, m
}
```

Updating an existing key preserves its insertion position.
Deleting it and inserting it again moves it to the end.
There is no guarantee of JavaScript iterator behavior when mutating during
iteration.

### 10.3 Keys and MapEntry

Map accepts keys of different types. `"1"` and `1` are different keys.
Strings are compared by contents; other objects by identity.
All NaN values represent one key, as do signed zeros.

`for...of map` returns `MapEntry` objects with `.key` and `.value` getters,
**not list pairs**. To destructure pairs, use `map.entries()`.

```napalm
const map = new Map();
const objectKey = {id: 1};
map.set(objectKey, "dato");
map.set(null, 7);
for (const [key, value] of map.entries()) {
    System.print(value);
}
System.print(map.get(objectKey)); // dato
System.print(map.get({id: 1}));   // null
```

### 10.4 Set

```napalm
const names = new Set<string>();
names.add("Ada").add("Lin").add("Ada");
System.print(names.size); // 2
System.print([...names]); // [Ada, Lin]
System.print(names.has("Ada"));
System.print(names.delete("Ada")); // true
```

| Member | Result |
| --- | --- |
| `new Set()` | Empty Set; does not accept an initial iterable. |
| `add(value)` | Inserts if absent; returns the Set. |
| `set_if_not_exists(value)` | Returns the stored equal element, or inserts and returns `value`. |
| `has(value)` | Bool. |
| `delete(value)` | Bool indicating whether it was present. |
| `clear()` | Empties and returns null. |
| `size`, `count`, `isEmpty` | Size and state. |
| `keys()`, `values()` | Return the Set itself as an iterable, not a new list. |
| `iterate`, `iteratorValue` | Iteration protocol. |

Set preserves insertion order and uses Map key equality.
To get an independent list, use `[...set]` or `[...set.values()]`.
There is no `new Set(list)`; add elements in a loop.

### 10.5 Generic Collections

`set_if_not_exists` also works on `ConcurrentMap` and `ConcurrentSet`, where
conditional insertion is atomic across workers. Arguments are evaluated before
the lookup; insertion contracts are checked only when no entry exists. Existing
entries are never overwritten by this operation.

`new Map<K, V>()`, `new Set<T>()`, and equivalent annotations install
persistent contracts. These include nested contents, unions, and structural
schemas.

```napalm
interface Item { value: number; }
const items = new Set<Item>();
items.add({value: 42});
const groups = new Map<string, Set<Item>>();
groups.set("items", items);
System.print(groups.get("items").size); // 1
```

An invalid write through `any` is still fatal. User-defined generic
interfaces, types and classes are checked as well (§8.8).

## 11. Ranges and Custom Iteration

### 11.1 Range

```napalm
const exclusive = new Range(1, 4);
const inclusive = new Range(1, 4, true);
System.print([...exclusive]); // [1, 2, 3]
System.print([...inclusive]); // [1, 2, 3, 4]
System.print([...new Range(3, 0)]); // [3, 2, 1]
```

The step is `+1` or `-1` depending on direction. Finite fractional endpoints
can be used; for example, `new Range(0.5, 2.5, true)` produces `0.5, 1.5, 2.5`.
There is no step parameter.

| Getter | Meaning |
| --- | --- |
| `from`, `to` | Original endpoints. |
| `min`, `max` | Smaller and larger endpoint. |
| `isInclusive` | Whether the final endpoint is included. |
| `count` | Count for traversal in steps of one. |
| `toString` / `toString()` | Range representation. |

Use finite endpoints to ensure iteration terminates. Wren operators
`1..5` and `1...5` **are not part of the current public syntax**.

### 11.2 List Slices

```napalm
const trees = ["cedar", "birch", "oak", "willow"];
System.print(trees[new Range(1, 3)]);       // [birch, oak]
System.print(trees[new Range(1, 3, true)]); // [birch, oak, willow]
System.print(trees[new Range(-1, 0, true)]);// [willow, oak, birch, cedar]
```

Slice endpoints must be integers. Negative values are normalized relative to
the size, reverse traversal is allowed, and a shallow copy is created. There
is no assignment to a full range as a slice.
A range outside the supported bounds produces an error, not necessarily
`Array.slice`'s silent clipping behavior in JS.

The public syntax for matrices is `matrix[row][column]`, not
`matrix[row, column]`, even though internal tests cover historical forms.

### 11.3 Iteration Protocol

The consumer starts with a `null` cursor, calls `iterate(cursor)`, and ends
when it receives a value that is falsy according to Napalm. For each valid
cursor it calls `iteratorValue(cursor)`.

```napalm
class Countdown {
    top;
    constructor(top) { this.top = top; }
    iterate(cursor) {
        const next = cursor == null ? this.top : cursor - 1;
        return next >= 0 ? next : false;
    }
    iteratorValue(cursor) { return cursor; }
}

System.print([...new Countdown(3)]); // [3, 2, 1, 0]
for (const n of new Countdown(2)) {
    System.print(n);
}
```

The cursor `0` is valid because it is truthy in Napalm.
`Symbol.iterator` and `{done, value}` objects are not part of this protocol.
Lists, strings, ranges, Maps, key views, Sets, and generators
([section 22.2](#222-generators)) implement it.
List and call spread also use it.

## 12. Destructuring, Rest, and Spread

### 12.1 List Patterns

```napalm
const [first, , third = 30, ...tail] = [10, 20, null, 40, 50];
System.print([first, third, tail]); // [10, 30, [40, 50]]
let a = 1;
let b = 2;
[a, b] = [b, a];
System.print([a, b]); // [2, 1]
```

Holes are skipped, missing values are null, and a default is evaluated only if
the selected value is null. Therefore, the default also applies to an
**explicit** null, but not to false, zero, or an empty string.

### 12.2 Object Patterns

```napalm
const dynamicKey = "name";
const {
    [dynamicKey]: title,
    settings: {enabled = true},
    ...remaining
} = {name: "Napalm", settings: {}, version: 1};

System.print([title, enabled, remaining.version]);
```

Aliases, computed keys, nesting, and defaults are supported.
The source and each computed key are evaluated once.
The rest is stored in a new Map without the extracted keys.

### 12.3 Parameters and Loops

```napalm
function describe({name, scores: [best = 0, ...others]}) {
    return `${name}: ${best}; restantes ${others}`;
}
System.print(describe({name: "Ada", scores: [42, 21, 7]}));
for (const [name, score] of [["Ada", 42], ["Lin", 7]]) {
    System.print(`${name}: ${score}`);
}
```

An inner default in a pattern does not make the entire argument optional:
`describe(null)` still fails because the pattern requires a Map.

### 12.4 Spread

```napalm
const input = [2, 3];
const expanded = [1, ...input, ...new Range(4, 6), ..."ab"];
const original = {x: 1, nested: [2]};
const changed = {...original, x: 9};
changed.nested.add(3);
function sum(a, b, c) { return a + b + c; }
System.print(expanded);          // [1, 2, 3, 4, 5, a, b]
System.print([original.x, changed.x, original.nested]);
System.print(sum(1, ...input));  // 6
```

Spread is also available in method, constructor, and `super` calls.
Arguments are expanded from left to right.
After expansion, they are still limited to 16.
An expanded call that is too long produces a recoverable error.

Rules and limits:

- Rest must appear at the end of the pattern; a comma after it is not allowed.
- Destructuring lists requires a List, and destructuring objects requires a
  Map. This is not a general adapter for iterables or class instances.
- Even `const [] = null` and `const {} = null` fail.
- List and call spread require an iterable.
- Object spread copies Maps and also uses Proxy traps. Spreading `null`
  currently produces a runtime error; it is not ignored.
- Rest/spread containers are independent; their children are shared.
- Later keys overwrite earlier keys in an object literal.
- Rest in patterns does not imply support for **rest parameters**.
- Spread is not supported across the `launch` boundary.

## 13. List Comprehensions

A comprehension builds a list from one or more traversals:

```napalm
const numbers = [1, 2, 3, 4, 5];
const squares = [n * n for n in numbers if n % 2 == 1];
const pairs = [
    [x, y]
    for x in [1, 2]
    for y in [10, 20]
    if x + y > 11
];
System.print(squares); // [1, 9, 25]
System.print(pairs);   // [[1, 20], [2, 10], [2, 20]]
```

The expression comes first, followed by the `for name in iterable` clauses.
Each clause can have multiple `if` filters and can depend on earlier bindings.
Do not write `let`, `const`, or `of` in this syntax.

**Here `in` means traversing values through the iteration protocol**; it does
not mean keys as in `for (const key in map)`.
A comprehension over a Map receives MapEntry values, as `for...of` does.
Clause bindings are local to the comprehension.
Comprehensions are not lazy generators: the result is a materialized list.
Each clause binding is a simple identifier, not a pattern.

## 14. Classes and Inheritance

### 14.1 Fields, Constructor, and Methods

```napalm
class Counter {
    value: number;
    history = [];
    static created = 0;

    constructor(start: number) {
        this.value = start;
        Counter.created++;
    }

    add(amount: number) {
        this.value += amount;
        this.history.add(this.value);
        return this.value;
    }

    get doubled() { return this.value * 2; }
    set doubled(value) { this.value = value / 2; }
    toString() { return `Counter(${this.value})`; }
}

const counter = new Counter(10);
counter.add(2);
counter.doubled = 30;
System.print(counter);         // Counter(15)
System.print(Counter.created); // 1
```

Instance field initializers run for each construction; a list declared as
`items = []` is not shared across instances.
Fields without initializers are usually assigned in the constructor.

A direct assignment to `this.name` inside a constructor also declares a public
instance field, so a separate `name;` declaration is optional:

```napalm
class Counter {
    constructor(value) { this.value = value; }
}
const counter = new Counter(1);
counter.value += 1;
System.print(counter.value); // 2
```

These fields have the same storage, external access and reactive tracking as
explicit class fields. Repeated assignments declare one field. Explicit fields
and accessors take precedence regardless of their position in the class.
Discovery includes constructor control-flow blocks and closures with lexical
`this`; nested classes have their own fields. A conditional assignment still
declares the field for every instance, with `null` until it is initialized.
Only direct named writes (`this.name`, including compound/destructuring
assignments) declare fields this way; reads, assignments to other objects and
writes found only in ordinary/static methods do not declare public fields.

A base class without an explicit constructor can be constructed without
arguments.

Always use `this.member` in methods. `function method(...) {...}` is also
accepted inside a class; `fn` is not.

Instances are not open Maps: `object.extra = value` cannot add an arbitrary
field. Declare the fields that make up their API. Instances provide `keys()`,
`values()`, `entries()` and `size` for their stored fields, including inherited
fields, in field-slot order. Computed getters and static fields are excluded.
User-defined members can override these defaults.

Stored fields can also be addressed by name with a computed key:

```napalm
class Point { x = 1; y = 2; }
const point = new Point();
const key = "x";
System.print(point[key]);     // 1
point[key] = 10;              // same checks as point.x = 10
System.print(point["z"]);     // null: unknown names read as null
```

`instance[key]` reads the stored field named `key` (inherited fields included)
and yields `null` for unknown names and non-String keys. `instance[key] = value`
keeps the field's declared type and fails for a name that is not a field, like
`object.extra = value`. Only stored fields are reachable this way: methods,
getters, computed members and static fields are not.

### 14.2 Inheritance and super

```napalm
class Shape {
    name;
    constructor(name) { this.name = name; }
    describe() { return this.name; }
}
class Rectangle extends Shape {
    width;
    height;
    constructor(width, height) {
        super("rectangle");
        this.width = width;
        this.height = height;
    }
    get area() { return this.width * this.height; }
    describe() { return `${super.describe()}: ${this.area}`; }
}
const rectangle = new Rectangle(6, 7);
System.print(rectangle.describe());       // rectangle: 42
System.print(rectangle instanceof Shape); // true
```

`extends` provides single inheritance.
`super(...)` initializes the same instance; `super.method(...)` calls the
superclass implementation.

**Write an explicit derived constructor with `super(...)`** when relying on
base-class initialization. Do not assume an empty derived class automatically
forwards all arguments or runs all initializers.
When derived initializers are present, the parser requires a direct call to
super before inserting them.

`instanceof` is useful for user class hierarchies: `value instanceof C` is true
for instances of `C` and of any subclass. Its right-hand side can also be an
interface or object type (§14.7). Do not use it as a universal `typeof` for
builtins: for example, a list is not reliably identified by
`list instanceof List` in the current implementation.

### 14.3 Static Members and static Blocks

```napalm
class Registry {
    static count = 0;
    static {
        Registry.count = 10;
    }
    static next() {
        return ++Registry.count;
    }
}
System.print(Registry.next()); // 11
```

Static initializers and blocks run once when the class declaration executes.
Use the class name to access its static members.
Do not assume inheritance of static methods/factories: `Derived.factory()`
does not necessarily find `Base.factory()`.

### 14.4 Parameter Properties

```napalm
class User {
    constructor(public name: string, private readonly id: number) {}
    describe() { return `${this.id}: ${this.name}`; }
}
const user = new User("Ada", 1);
System.print(user.describe());
```

Constructor parameter modifiers can synthesize the property.
Do not confuse this with access security: class `public`, `private`,
`protected`, `readonly`, and `override` are erased as visibility or
immutability controls. **Interface** `readonly` is part of a contract.
There are no ECMAScript `#private` fields.

### 14.5 Functions Stored as Fields

```napalm
class Bag {
    callback = (n) => n + 1;
    makeReader() { return () => this.callback; }
}
const bag = new Bag();
System.print((bag.callback)(41)); // 42
```

The parentheses in `(bag.callback)(...)` force a field read followed by a call
to that function. `bag.callback(...)` selects method dispatch and is not
necessarily equivalent to calling the field.
Reading a method should not be assumed to produce a receiver-bound function
as in other libraries.

### 14.6 Overloads and Operators

Methods can be defined with the same name and different arities.
The public syntax allows only one constructor implementation.

```napalm
class Vector {
    x;
    constructor(x) { this.x = x; }
    operator +(other) { return new Vector(this.x + other.x); }
    operator -() { return new Vector(-this.x); }
    operator ==(other) { return this.x == other.x; }
    toString() { return `Vector(${this.x})`; }
}
const a = new Vector(2);
const b = new Vector(3);
System.print(a + b);                // Vector(5)
System.print(-a);                   // Vector(-2)
System.print(a == new Vector(2));   // true
```

Operators allowed with `operator`:

- Unary: `!`, `~`, `-`, with no parameters.
- Binary: `+`, `-`, `*`, `**`, `/`, `%`, `<<`, `>>`, `&`, `^`, `|`,
  `<`, `<=`, `>`, `>=`, `==`/`===`, `!=`/`!==`, with one parameter.

These are concrete, synchronous instance methods; they cannot be `static`,
`async`, `foreign`, `abstract`, or decorated.
`&&`, `||`, `??`, assignment, and subscripts cannot be overloaded using an
`operator []` declaration.

### 14.7 Implementing Interfaces and Types

`implements` lists interfaces and object type aliases declared in the same
module. Each contract field, including fields inherited through an interface's
`extends`, becomes an instance field of the class with the declared type:

```napalm
interface Named {
    name: string;
    nickname?: string;
}
interface Aged extends Named {
    age: number;
}
type Point = {x: number; y: number; label: string | null};

class Animal implements Named {
    constructor(name: string) { this.name = name; }
}
class Dog extends Animal implements Aged, Point {
    x = 0;
    y = 0;
    constructor(name: string, age: number) {
        super(name);
        this.age = age;
    }
}

const dog = new Dog("Rex", 3);
System.print(dog.keys());                // [name, nickname, x, y, age, label]
System.print([dog.nickname, dog.label]); // [null, null]
```

- A field is **required** unless it is optional (`field?: T`), its type
  accepts `null` (`T | null`), or it is `any`. Optional and nullable fields
  start as `null`.
- A required field must be initialized by the class: a field initializer,
  a definite field (`field!: T`), a parameter property, a `this.field = ...`
  assignment in the constructor, or an ancestor class that already does so.
  Otherwise compilation fails.
- An `abstract` class may leave required fields uninitialized; its first
  concrete subclass must initialize them.
- Fields already declared by the class or inherited from a superclass are
  reused, not duplicated. Writes are checked against the contract's type.
- Generic aliases can be implemented with type arguments, as in
  `implements Box<number>`.
- An unknown name, a non-object alias (`type Id = string | number`), a field
  declared with conflicting types by two contracts, or a method with the same
  name as a contract field is a compile error.

```napalm compile-error
interface Named { name: string; }
class Anonymous implements Named {
    constructor() {} // `name` is required and never initialized
}
```

```napalm compile-error
interface Shape {
    name: string;
    area(): number; // interfaces and types cannot declare methods
}
```

Three operators inspect classes and contracts. All of them return `false` for
`null` and primitive values:

| Expression | True when |
| --- | --- |
| `value instanceof C` | `value` is an instance of class `C` or of a subclass. |
| `value instanceof I` | `value` is an instance whose class or an ancestor implements `I` (or contracts that cover all of its fields), or a Map/object literal that currently satisfies `I` structurally. |
| `value implements I` | Same as `instanceof I`, but a class object also qualifies: `Dog implements Named`. The right-hand side must be an interface or object type. |
| `value inherits C` | `value` is an instance or class object whose class **strictly** descends from `C`: `Dog inherits Animal` is true, `Dog inherits Dog` is false. |

```napalm
interface Named { name: string; }
class Animal implements Named { constructor() { this.name = "a"; } }
class Dog extends Animal {}

const dog = new Dog();
System.print([dog instanceof Animal, dog instanceof Named]);   // [true, true]
System.print([Dog implements Named, Dog instanceof Named]);    // [true, false]
System.print([dog inherits Animal, Dog inherits Animal, Animal inherits Animal]); // [true, true, false]
System.print([{name: "Ada"} implements Named, 42 implements Named]); // [true, false]
```

A structural check on a Map does not install a contract on it; use a typed
binding for that (§8.5).

## 15. Reactive Properties and Watchers

### 15.1 computed

```napalm
let evaluations = 0;
class Vector {
    x = 2;
    y = 3;
    computed magnitude2() {
        evaluations++;
        return this.x * this.x + this.y * this.y;
    }
}
const vector = new Vector();
System.print(vector.magnitude2); // 13
System.print(vector.magnitude2); // 13, usa cache
vector.x = 4;
System.print(vector.magnitude2); // 25, vuelve a calcular
System.print(evaluations);       // 2
```

`computed name() {...}` declares a **property**, read without parentheses.
It is evaluated lazily and retains its result until tracked dependencies
change. Dependencies are discovered by reading fields during evaluation,
including reads through other computed properties.
Reevaluation can change the set of dependencies.

Use computed properties for derivations without side effects. The counter in
the example only demonstrates when evaluation occurs.
Do not assume that every global, List element, or Map entry is observed: the
implemented tracking focuses on field accesses.
Also do not assume deep comparison or that assigning an equal value never
invalidates.

### 15.2 watch

```napalm
const observations = [];
class Model {
    value = 1;
    watch report() {
        observations.add(this.value);
    }
}
const model = new Model();
await Timer.microtask(() => null);
model.value = 2;
model.value = 3;
await Timer.microtask(() => null);
System.print(observations); // [1, 3]
```

A watcher is registered during construction, after ordinary initializers, and
runs its first evaluation as a microtask.
When a dependency changes, it is scheduled again. Multiple changes before it
runs may be grouped into a single execution.
It is not called synchronously and immediately when the field is written.

Both computed and watch:

- They are concrete, synchronous, parameterless instance members.
- They do not allow the `function` prefix, decorators, `static`, `async`, or
  `foreign`.
- They are dependency-tracked state derivation primitives, not explicit event
  streams.
- The low-level `__Reactive` functions are internal support and are not an
  application API. Use `Event`/`Emitter` when explicit subscription and
  publication are required.

### 15.3 Event and Emitter

`Emitter<T>` is Napalm's native one-to-many publication endpoint.
Its `event` getter returns an `Event<T>` view that can subscribe but cannot
publish. The split is deliberate: code can be given an `Event<T>` without also
being given the capability to emit values.

The generic payload is optional. Without `<T>`, payloads are `any`.

```napalm
const emitter = new Emitter<number>();
const event: Event<number> = emitter.event;

const token = event.on((value) => {
    System.print(`value: ${value}`);
});

event.once((value) => {
    System.print(`first emission only: ${value}`);
});

System.print(emitter.emit(10)); // number of listener deliveries queued
System.print(emitter.emit(20));

event.off(token);
System.print(event.listenerCount);

emitter.clear();
emitter.close();
System.print(event.closed);
```

| Member | Effect |
| --- | --- |
| `new Emitter<T>()` | Creates an emitter; `<T>` is optional and defaults to `any`. |
| `emitter.event` | Read-only `Event<T>` capability sharing the same event state. |
| `emitter.emit(value)` | Queues delivery to the current listeners and returns the number of queued deliveries. |
| `emitter.listenerCount` | Number of currently registered listeners. |
| `emitter.clear()` | Removes all current listeners. |
| `emitter.close()` | Closes the emitter and prevents future publication/subscription. |
| `event.on(callback)` | Adds a persistent listener and returns a subscription token. |
| `event.once(callback)` | Adds a listener consumed by the first emission that observes it. |
| `event.off(token)` | Removes the subscription represented by that token. |
| `event.off(callback)` | Removes subscriptions using that callback. |
| `event.listenerCount` | Number of currently registered listeners. |
| `event.closed` | Whether the shared emitter/event state is closed. |

Event delivery uses the VM's **microtask queue**. `emit()` does not invoke
listeners inline, which prevents arbitrary listener reentrancy inside the
publisher. Listener callbacks may take zero or one argument.

`emit()` snapshots the listeners that should receive that emission. Therefore,
calling `off()`, `clear()`, or `close()` after `emit()` does not retract
deliveries that have already been queued.

A `once()` listener is consumed by the first emission that sees it, even though
the actual callback runs later as a microtask.

A `concurrent function` is not a direct event listener. If an event should
start parallel work, use an ordinary or async listener that explicitly performs
`launch`.

Typed event contracts are persistent runtime contracts in addition to static
checks. Hiding a typed endpoint behind `any` does not remove its payload
restriction:

```napalm runtime-error
const emitter = new Emitter<number>();
const dynamic: any = emitter;
dynamic.emit("not a number");
```

`Event<T>` and its paired `Emitter<T>` share one payload contract. Applying a
typed `Event<T>` contract to an initially untyped event also constrains its
paired emitter.

Event/Emitter endpoints are currently **VM-local**. They can be used by
Fibers/tasks in one Interpreter, but they cannot cross the isolated
`launch concurrent` worker boundary.

## 16. Enums

```napalm
enum Color { Red, Green, Blue }
enum Status { Pending = 10, Running, Failed = -1, Done = 20 }
enum Direction { Up = "up", Down = "down" }
enum Flags { Read = 1, Write = 2, Both = Read | Write }

System.print([Color.Red, Color.Green, Color.Blue]); // [0, 1, 2]
System.print(Status.Running);                     // 11
System.print(Color[2]);                           // Blue
System.print(Direction.Up);                       // up
System.print(Flags.Both);                         // 3
```

Enums are real Map-based values:

- Without an initializer, the first numeric member is zero.
- Subsequent members can auto-increment after a constant numeric member.
- After a string or non-constant numeric initializer, explicitly specify the
  next value.
- Numeric members create a direct `Name -> value` entry and a reverse
  `value -> Name` entry.
- String members do not create an automatic reverse entry.
- Initializers can refer to earlier members.
- Local and exported enums are allowed; their initializers run when the
  declaration is reached.
- The enum binding is constant; the Map is not automatically frozen.
- There is no TypeScript `const enum` or a new runtime enum-value class.

If a numeric value is repeated, the last reverse entry for that key overwrites
the previous one, as in any Map.

## 17. Decorators

### 17.1 Form and Order

`@decorator`, `@namespace.decorator`, and `@factory(...)` are supported on
functions, classes, methods, getters, setters, and fields, including static
members.

```napalm
function suffix(text) {
    return (value, context) => {
        return (name) => value(name) + text;
    };
}
@suffix("A")
@suffix("B")
function greet(name) { return `Hola ${name}:`; }
System.print(greet("Ada")); // Hola Ada:BA
```

Decorator expressions/factories are evaluated from top to bottom in their
lexical scope.
Decorators are applied from bottom to top. Each receives `(value, context)`.
Returning null, including by falling through to the end of the function,
preserves the current value.

### 17.2 Context and Signatures

| `context.kind` | Received value | Expected replacement |
| --- | --- | --- |
| `"function"` | Original function. | Function with the original arguments. |
| `"class"` | Original class. | Replacement class. |
| `"method"` | Method callable. | Function `(receiver, arguments...)`. |
| `"getter"` | Getter callable. | Function `(receiver)`. |
| `"setter"` | Setter callable. | Function `(receiver, nextValue)`. |
| `"field"` | null. | Transformer `(receiver, initialValue)`. |

`arguments...` is table notation, **not rest-parameter syntax**.
Non-null replacements for functions/members must be functions; class
replacements must be classes. An invalid result produces a recoverable error.

`context.name` is the declared name.
Members receive `context.static` and `context.private`; the latter is
currently false because visibility modifiers do not create privacy.

### 17.3 Explicit Member Receiver

```napalm
function plusOne(original, context) {
    return (receiver, amount) => original(receiver, amount) + 1;
}
class Counter {
    value = 10;
    @plusOne
    add(amount) { return this.value + amount; }
}
System.print(new Counter().add(2)); // 13
```

Wrappers do not receive a dynamic JavaScript `this`: the receiver is the first
explicit argument. This reduces the number of ordinary arguments to a
decorated method to **15**, reserving one for the receiver.

### 17.4 Field and Initializer Transformation

```napalm
function doubleField(value, context) {
    context.addInitializer((receiver) => {
        System.print(receiver.amount);
    });
    return (receiver, initialValue) => initialValue * 2;
}
class Invoice {
    @doubleField
    amount = 21;
}
const invoice = new Invoice(); // The callback prints 42.
System.print(invoice.amount); // 42
```

`context.addInitializer(fn)` can only be called **while the decorator is
executing**:

- For a function/class, the callback takes no arguments.
- For a member, it receives the receiver.
- The final function/class binding is installed before its callbacks run.
- Instance method/getter/setter callbacks run for each construction, after
  `super` in derived classes and before ordinary fields.
- Static equivalents run once before static fields and blocks.
- Transformers for a stacked field run from top to bottom.
- A field without an initializer passes null to the transformer.
- Extra callbacks for that field run after it is assigned.
- Callbacks run in registration order; an error aborts the remaining ones.

Place decorators **before `export`**.
Decorators are not supported on parameters, constructors, static blocks,
operators, computed/watch members, or foreign/abstract/declare members.
There is no `context.access`, auto-accessors, or complete implementation of
all JavaScript decorator proposals.

## 18. Native Decorators

### 18.1 Catalog

All durations are expressed in **milliseconds**.

| Decorator | Function |
| --- | --- |
| `@memoize` | Cache by the complete argument tuple. |
| `@memoizeExpiring(ttl)` | Cache by key with expiration measured from insertion. |
| `@once` | Reuses the first successful return, regardless of later arguments. |
| `@delay(wait)` | Schedules each call independently and returns a Promise. |
| `@debounce(wait, options?)` | Groups a burst of calls and uses the latest arguments. |
| `@throttle(wait, options?)` | Limits invocations per time window. |

They apply to functions and callable members, not classes or field
transformers. They preserve the explicit receiver of members.

### 18.2 Caching

```napalm
let evaluations = 0;
@memoize
function multiply(a, b) {
    evaluations++;
    return a * b;
}
System.print(multiply(6, 7));
System.print(multiply(6, 7));
System.print(evaluations); // 1

@once
function initialize(value) { return {value}; }
const first = initialize(1);
System.print(initialize(9) === first); // true
```

The key includes all arguments and, for a method, the receiver.
Strings are compared by contents and objects by identity, not by deep
serialization.

Caveats:

- Mutating an argument does not invalidate its entry.
- Globals and captured variables are not part of the key.
- References to arguments and results are retained; an unbounded cache can
  retain a large amount of memory.
- A synchronous error is not cached.
- A returned Promise is **cached immediately**, even if it is later rejected.
- `once` on a member has separate state for each receiver.
- `memoizeExpiring` hits do not renew the TTL; with a TTL of zero, there is no
  durable reuse.

### 18.3 delay, debounce, and throttle

```napalm
let saves = 0;
@debounce(10)
function save(value) {
    saves++;
    return value;
}
const first = save(1);
const second = save(2);
const third = save(3);
System.print(await Promise.all([first, second, third])); // [3, 3, 3]
System.print(saves); // 1
```

Options:

| Option | debounce | throttle |
| --- | --- | --- |
| `leading` | false by default. | true by default. |
| `trailing` | true by default. | true by default. |
| `maxWait` | Limits postponement of a continuous burst. | Do not use it as an additional throttle contract. |

Pass options as an object:
`@debounce(100, {leading: true, trailing: false})`.
`options = {...}` would be an assignment, not a named argument.
Disabling both edges is invalid. Durations for these decorators must be finite,
non-negative, and no greater than `2,147,483,647` ms.

The three timing decorators:

- Always return Promises, even when decorating a synchronous function.
- Adopt the result if the decorated function returns another Promise.
- Convert scheduled exceptions into rejections.
- Group the result of calls that correspond to the same invocation.
- Keep timing state separate by receiver for members.

With leading and trailing enabled, a single call does not trigger a duplicate
trailing execution. Later calls in the window receive the trailing result.
If trailing is disabled, suppressed calls share the leading result.
`delay` does not group calls: it schedules each one.

## 19. Exceptions and Cleanup

### 19.1 throw, try, catch, and finally

```napalm
function divide(a, b) {
    if (b == 0) throw {message: "division no permitida", divisor: b};
    return a / b;
}
try {
    divide(10, 0);
} catch (error) {
    System.print(error.message);
} finally {
    System.print("limpieza");
}
```

Any value can be thrown, not just an Error instance.
Do not assume a global `Error` constructor with the JS API.
Ordinary runtime errors, such as a missing method or invalid index, can also
enter `catch`.

`catch { ... }` without a binding is allowed.
`try` must have `catch`, `finally`, or both.
An error can be rethrown with `throw error`.
`finally` runs on normal exit, throw, return, or control-flow jump; a return or
throw inside finally can replace the pending one.

Compilation errors and fatal type contracts are not exceptions that can be
recovered through this mechanism.

### 19.2 defer

```napalm
function work() {
    let value = 1;
    defer System.print(`ultimo: ${value}`);
    defer {
        System.print("primero");
    }
    value = 7;
    return value;
}
System.print(work());
// primero
// ultimo: 7
// 7
```

`defer statement` and `defer { block }` register cleanup for scope exit. They
run in **reverse order** of registration.
They do not run when the defer is declared and observe bindings at cleanup
time.
Only defers whose declaration point was reached are registered.
Inside a loop body, they can clean up each iteration, including on
break/continue.

### 19.3 errdefer

```napalm
function work(fail) {
    defer System.print("siempre");
    errdefer System.print("solo en error");
    errdefer |error| System.print(`detalle: ${error}`);
    if (fail) throw "boom";
    return 42;
}
System.print(work(false));
try {
    work(true);
} catch (error) {
    System.print(`capturado: ${error}`);
}
```

`errdefer` runs only when a scope is exited because of an error.
The optional binding uses **bars**: `errdefer |error| ...`.
The form `errdefer (error)` is invalid, even if it appears in an old script.
Do not rely on defer/errdefer to recover from fatal errors, forced process
termination, or native-code failures.

## 20. Async, Promise, and Timers

### 20.1 async and await

```napalm
async function answer() {
    await Timer.delay(1);
    return 42;
}
const plusOne = async (n) => {
    return (await Promise.resolve(n)) + 1;
};
const task = answer();
System.print(await task);        // 42
System.print(await plusOne(41)); // 42
System.print(await Promise.resolve(7)); // 7
```

Async functions, methods, arrows, and lambdas are supported.
An async call immediately returns a Promise; its work is scheduled as a
cooperative task. **Do not assume its body runs synchronously up to the first
await as it does in JavaScript.**

`await` is allowed at module level and inside async/concurrent functions.
It is rejected in an ordinary function without those qualifiers.
Awaiting a rejected Promise throws its reason. `await` requires a Promise;
awaiting an ordinary value produces a runtime error.

Async does not mean parallel CPU execution: use `concurrent`/`launch` for that.
A long synchronous computation can delay other tasks and timers.

### 20.2 Promise API

| API | Effect |
| --- | --- |
| `Promise.resolve(value)` | Resolved Promise; adopts values/Promises through the adoption mechanism. |
| `Promise.reject(reason)` | Rejected Promise. |
| `Promise.all(list)` | Results in input order; fails if any entry fails. |
| `Promise.any(list)` | First fulfillment; AggregateError if all fail. |
| `Promise.race(list)` | First settled result, success or failure. |
| `promise.cancel()` | Requests cancellation; true if the state changed, false if it could no longer change. |
| `isPending`, `isFulfilled`, `isRejected`, `isCancelled` | State getters. |

Combinators accept **a List**, not any iterable, and allow ordinary values to
be mixed with Promises.

```napalm
const results = await Promise.all([
    Timer.delay(1, () => 10),
    Promise.resolve(20),
    3
]);
System.print(results); // [10, 20, 3]

try {
    await Promise.any([Promise.reject("A"), Promise.reject("B")]);
} catch (error) {
    System.print(error.name);   // AggregateError
    System.print(error.errors); // [A, B]
}
```

Empty cases:

- `Promise.all([])` fulfills with `[]`.
- `Promise.any([])` rejects with AggregateError and an empty error list.
- `Promise.race([])` remains pending.

The public constructors `new Promise(executor)`, `Promise(executor)`,
`Deferred()`, `.then()`, `.catch()`, `.finally()`, and `allSettled` are not
installed. Use async/await and try/catch/finally.
Do not assume a combinator automatically cancels competing work; explicitly
cancel tasks that are no longer needed.

### 20.3 Timer

| API | Effect |
| --- | --- |
| `Timer.delay(ms)` | Promise that fulfills after the delay. |
| `Timer.delay(ms, callback)` | Runs the callback with no arguments and resolves with its result. |
| `Timer.microtask(callback)` | Schedules after the current work; returns a Promise. |
| `Timer.every(ms, callback)` | Cancelable, non-overlapping periodic timer. |
| `setTimeout(callback, ms)` | Global alias for a one-shot timer. |

Delays are **finite, non-negative milliseconds**.
`Timer.every` requires a delay greater than zero.
A zero delay does not mean the callback runs inline.

```napalm
const timer = setTimeout(() => System.print("no se ejecuta"), 1000);
System.print(timer.cancel());      // true
System.print(timer.cancel());      // false
System.print(timer.isCancelled);   // true
System.print(await Timer.delay(0, () => 42)); // 42
```

Intervals do not overlap callback executions, even if the callback returns
a Promise. Cancel them so the program can terminate:

```napalm
let ticks = 0;
let interval: any = null;
interval = Timer.every(1, () => {
    ticks++;
    System.print(ticks);
    if (ticks >= 3) interval.cancel();
});
```

Cancellation does not roll back effects that have already occurred.
For a timer, it prevents pending callbacks; for an async task, the result
depends on its state. There is no standard `clearTimeout`: use the returned
Promise and `cancel`.
`Date.sleep` is blocking and does not replace `await Timer.delay`.

## 21. Concurrency with launch

### 21.1 Running CPU Work in Parallel

```napalm
concurrent function sumTo(limit: number) {
    let total = 0;
    for (let i = 0; i <= limit; i++) total += i;
    return total;
}
const first = launch sumTo(100);
const second = launch sumTo(200);
System.print(await Promise.all([first, second])); // [5050, 20100]
```

`concurrent function` declares a function that can run in isolated workers.
`launch f(args)` returns a Promise.
Launch all tasks before awaiting them to allow concurrency.
`await launch f(...)` is useful for a single task, but doing so inside a loop
can serialize the work.

The `concurrent (...) => ...`, `concurrent function (...) {...}`, and
`concurrent |x| ...` closure forms are also available.
`async concurrent` is not allowed: the launch boundary is already asynchronous.

### 21.2 Target Restrictions

- The target must be a function declared `concurrent`.
- It cannot be called directly with `f(...)`; use launch.
- Launch accepts a direct call to a function value, not arbitrary method
  dispatch.
- Exactly the declared parameters must be passed.
- Argument spread is not allowed in launch.
- It cannot capture lexical state; pass that state as arguments.
- It cannot read/write module globals arbitrarily.
- It can use builtins and synchronous, capture-free helper functions that the
  runtime can materialize as code in the worker.
- Nested launch from a worker is not allowed.

Do not assume that importing a stateful function or using a helper that
depends on globals shares that state with the worker.

### 21.3 Copying Data, Not Sharing Mutable Memory

```napalm
concurrent function incrementCopy(object) {
    object.value++;
    return object;
}
const original = {value: 4};
const changed = await launch incrementCopy(original);
System.print([original.value, changed.value]); // [4, 5]
```

The following can cross the boundary:

- null, Bool, numbers, and strings.
- Nested acyclic Lists and Maps.
- Map keys that are null, Bool, number, or string.

Functions, captures, Promises, Fibers, Event/Emitter, Channel/Receiver,
instances/classes, Proxy, Set, Range, Date, Random, Response/Headers, FFI
libraries/functions/pointers, and external objects cannot cross.
Neither can Maps with object keys or cyclic graphs.
The maximum depth is 256.

Arguments are snapshotted at launch and the result is materialized in the
receiver. Repeated references within the transferred graph are preserved as
aliases in the new graph, not as references to the original.
Do not assume that all contracts installed on the source are transferred;
annotate boundaries that need data validation.

Worker errors reject the Promise with a `ConcurrentError` reason.
`cancel()` requests cancellation; it does not reverse external effects already
performed and does not guarantee immediate interruption of a blocking native
call.
The number of workers and the speedup depend on the machine; five tasks do not
imply a fivefold speedup.

### 21.4 Channel and Receiver

`Channel<T>` and `Receiver<T>` provide native FIFO message passing inside one
Interpreter. `Channel<T>` owns the send capability; its `receiver` getter
returns the receive capability.

The generic payload is optional. Without `<T>`, payloads are `any`.

```napalm
const channel = new Channel<string>(64);
const receiver: Receiver<string> = channel.receiver;

System.print(channel.send("hello")); // true
System.print(channel.send("world")); // true

System.print(await receiver.receive()); // hello
System.print(receiver.tryReceive());    // world

channel.close();
System.print(await receiver.receive()); // null
```

| Member | Effect |
| --- | --- |
| `new Channel<T>()` | Creates an unbounded FIFO channel. `<T>` is optional and defaults to `any`. |
| `new Channel<T>(capacity)` | Creates a bounded channel. |
| `channel.receiver` | Returns the paired `Receiver<T>` capability. |
| `channel.send(value)` | Sends without blocking; returns false when closed or currently unable to accept the value. |
| `channel.close()` | Closes the channel while preserving already-buffered values for draining. |
| `channel.length` | Number of currently buffered messages. |
| `channel.capacity` | Configured capacity. |
| `channel.closed` | Whether the shared channel state is closed. |
| `receiver.receive()` | Returns `Promise<T | null>`; waits asynchronously when no message is available. |
| `receiver.tryReceive()` | Returns an immediately available value or null. |
| `receiver.close()` | Closes the shared channel state. |
| `receiver.length` | Number of currently buffered messages. |
| `receiver.closed` | Whether the shared channel state is closed. |

A bounded channel never blocks the sender. When its buffer is full,
`send(value)` returns `false`.

Capacity `0` creates a **rendezvous channel**. It has no buffer:
`send(value)` succeeds only when a receiver is already waiting.

```napalm
const channel = new Channel<number>(0);
const receiver = channel.receiver;

const pending = receiver.receive();
System.print(channel.send(42)); // true
System.print(await pending);    // 42
```

`receive()` resolves immediately when a buffered value is available. Otherwise
it remains pending until a sender supplies a value or the channel closes.

Closing a channel has these semantics:

1. No new sends are accepted.
2. Already-buffered values remain available and are drained in FIFO order.
3. Pending receivers that cannot receive a buffered value complete with `null`.
4. After the buffer is empty, subsequent `receive()` calls resolve with `null`.

The nullable result is part of the type: `Receiver<T>.receive()` has an await
result of `T | null`, not just `T`.

As with Event/Emitter, typed channel contracts survive aliases through `any`:

```napalm runtime-error
const numbers = new Channel<number>();
const dynamic: any = numbers;
dynamic.send("not a number");
```

`Channel<T>` and its paired `Receiver<T>` share one payload contract. Applying
a typed contract to one side constrains the other side as well.

These endpoints are currently **VM-local**. They work across cooperative tasks
and Fibers in one Interpreter, but they do not yet cross `launch concurrent`.
The worker boundary copies supported values into an isolated Interpreter; a
future cross-worker channel requires transport in the concurrent runtime rather
than sharing VM heap handles.

## 22. Fibers and Generators

### 22.1 Fibers

Fibers are explicit-transfer coroutines. They are not threads or Promises.

```napalm
const fiber = new Fiber((first) => {
    const resumed = Fiber.yield(first + 1);
    return resumed + 2;
});
System.print(fiber.call(10)); // 11
System.print(fiber.call(20)); // 22
System.print(fiber.isDone);   // true
```

The first call passes the initial argument. `yield` passes a value to the
caller and suspends the Fiber. The next call returns its argument as the result
of that yield.

| API | Effect |
| --- | --- |
| `new Fiber(body)`, `Fiber.new(body)` | Creates a Fiber from a closure. |
| `Fiber.current` | Current Fiber; in async tasks it does not expose the internal handle and returns null. |
| `fiber.call(value?)` | Activates/resumes while saving the caller. |
| `fiber.try(value?)` | Like call, but returns the recoverable error if the Fiber fails. |
| `fiber.error` | Stored error. |
| `fiber.isDone` | Whether it finished or aborted. |
| `Fiber.yield(value?)` | Returns control to the caller. |
| `fiber.transfer(value?)` | Peer-to-peer transfer, without creating the same caller link as call. |
| `fiber.transferError(error)` | Transfers control while delivering an error. |
| `Fiber.abort(error)` | Aborts with an error; `abort(null)` is a no-op. |
| `Fiber.suspend()` | Suspends to the host embedding the interpreter. |

```napalm
const failed = new Fiber(() => {
    throw "boom";
});
System.print(failed.try()); // boom
System.print(failed.error); // boom
System.print(failed.isDone);
```

The root Fiber cannot be activated with `call` like a new Fiber.
`Fiber.suspend()` requires the host to resume the VM; in the ordinary
executable, a final module suspension is a runtime error.
Fiber control operations are unavailable inside async tasks. Do not mix the
two models as if they were equivalent.
Fatal contracts do not become recoverable with `fiber.try()` either.

### 22.2 Generators

A generator function is declared with `function*` (or `*name()` for a class
method). Calling it does **not** run the body: it returns a `Generator` whose
body runs lazily, suspending at each `yield` until the consumer asks for the
next value.

```napalm
function* conversation(): Generator<string, string, number> {
    const age = yield "How old are you?";
    yield `In ten years you will be ${age + 10}`;
    return "bye";
}

const talk = conversation();          // nothing has run yet
System.print(talk.next().value);      // How old are you?
System.print(talk.next(30).value);    // In ten years you will be 40
const last = talk.next(0);
System.print([last.value, last.done]);  // [bye, true]
System.print(talk.next(0).done);        // true (value is null)
```

`next()`, `return()` and `throw()` return an object `{value, done}`. The
argument of the first `next()` is ignored; each later argument becomes the
result of the `yield` expression where the body is suspended.

| Form or API | Effect |
| --- | --- |
| `function* f() {}`, `const f = function* () {}` | Generator function declaration or expression. |
| `*name() {}` in a class | Generator method; `this` is the receiver as usual. |
| `yield value`, `yield` | Suspends and produces `value` (or `null`). Evaluates to the next `next()` argument. |
| `yield* sequence` | Delegates to another generator, forwarding values, `next()` arguments, `return()` and `throw()`; evaluates to the delegate's return value. Any other iterable (list, range, Map...) is also accepted; then it evaluates to `null`. |
| `gen.next(value?)` | Resumes the body. |
| `gen.return(value?)` | Completes the generator as if `return value` ran at the suspended `yield`; `finally` blocks run. |
| `gen.throw(error)` | Raises `error` at the suspended `yield`, where the body may catch it. |
| `value instanceof Generator` | Whether the value is a generator. |

Generators implement the iteration protocol (section 11.3), so `for...of`,
list spread and call spread consume them. Leaving a `for...of` early with
`break`, `return` or an exception **closes** the generator, running its
pending `finally` blocks.

```napalm
function* fibonacci() {
    let a = 0;
    let b = 1;
    while (true) {
        yield a;
        const next = a + b;
        a = b;
        b = next;
    }
}

for (const n of fibonacci()) {
    if (n > 20) break;          // closes the infinite generator
    System.print(n);
}

class Tree {
    constructor(public left: any, public value: number, public right: any) {}
    *walk(): Generator<number> {
        if (this.left) yield* this.left.walk();
        yield this.value;
        if (this.right) yield* this.right.walk();
    }
}
const tree = new Tree(new Tree(null, 1, null), 2, new Tree(null, 3, null));
System.print([...tree.walk()]); // [1, 2, 3]
```

Rules and semantics:

- The annotation `Generator<Yield, Return, Next>` (one to three arguments) is
  checked at runtime: every yielded value against `Yield`, every `return`
  (including falling off the end) against `Return`, and every `next(value)`
  argument against `Next`. `void`, `any` and omitted arguments leave that
  channel unchecked. `next()` without an argument sends `null`, so after the
  first call a generator with a non-nullable `Next` must be resumed with a
  value. A violation is a fatal contract error. A generator
  passed through `Generator<T>` keeps that contract even through `any` aliases.
- An exception thrown by the body propagates to the caller of `next()`,
  `return()` or `throw()`, and the generator is then done.
- `return()` on a generator that has not started finishes it without running
  the body; `throw()` on it rethrows the error to the caller.
- If the generator is suspended inside `try...finally`, `return()` runs the
  `finally` block. If that block yields, `return()` returns that value with
  `done: false`, and the return completes on a later resumption.
- Resuming a generator from inside its own body (directly or through another
  generator) is a runtime error: "generator is already running".
- `yield` is only valid directly in the generator's own body; it cannot appear
  in a nested function or arrow function. Async, concurrent and
  getter/setter/constructor generators are not supported.
- `typeof gen` is `Generator`, and its text form is `<generator#N>`.
- Array destructuring requires a List, so `const [a, b] = gen()` is rejected;
  use `const [a, b] = [...gen()]` for a finite generator.

Generators are built on Fibers: each one owns a separate resumable stack, and
its hot loops are compiled by the JIT like any other code. A generator is not
a `Fiber`, though: `call`, `transfer` and the other Fiber operations are not
available on it.

## 23. Modules, Imports, and Exports

### 23.1 Exporting and Importing

Module files have their own scope.
Only explicitly exported names are public.
The executable example in the repository is
[module_import.napalm](scripts/module_import.napalm).

File `maths.napalm`:

```napalm fragment
export const Answer = 42;
export function add(a, b) { return a + b; }
export class Calculator {
    static multiply(a, b) { return a * b; }
}
export default Calculator;
```

File `main.napalm` in the same directory:

```napalm fragment
import DefaultCalculator, {Answer, add as sum} from "./maths";
System.print(Answer);
System.print(sum(20, 22));
System.print(DefaultCalculator.multiply(6, 7));
```

Supported forms:

| Form | Use |
| --- | --- |
| `import "./setup";` | Run module side effects. |
| `import {name, other as local} from "./dep";` | Named imports. |
| `import value from "./dep";` | Default import. |
| `import value, {name} from "./dep";` | Both in one declaration. |
| `export const/let/var ...` | Export declared bindings. |
| `export function/class/enum ...` | Export a declaration with a runtime value. |
| `export {local as publicName};` | Publish local bindings. |
| `export {name as renamed} from "./dep";` | Named re-export. |
| `export default expression;` | Default value. |
| `export default function (...) {...}` | Default function, which can be anonymous. |
| `export default class {...}` | Default class, which can be anonymous. |

Imports are non-reassignable bindings.
Do not assume ECMAScript live bindings: values are read when the import is
executed; imported objects retain shared identity, but a later reassignment by
the exporter does not necessarily update the imported binding.

`import * as ns`, `export *`, dynamic imports `import(...)`, `require`, and a
`node_modules` resolver are not implemented.
Specifiers are static strings, not interpolated templates.

### 23.2 Path Resolution

For `import "./util"`, the following are searched in this order:

1. `util.napalm`.
2. `util/index.napalm`.

An explicit extension must be `.napalm`.
A name without a prefix such as `"util"` is relative to the importer, not an
npm package. Relative paths are resolved from the importing file, not from
the process working directory.

| Prefix | Roots |
| --- | --- |
| `@lib/package[/subpath]` | `.napalm/packages`, paths in `NAPALM_LIB_PATH`, and `NAPALM_HOME/packages`. |
| `@std/name[/subpath]` | `.napalm/std`, paths in `NAPALM_STD_PATH`, `NAPALM_HOME/std`, and locations beside the executable. |

Project-root discovery recognizes `napalm.toml`, `Napalm.toml`, `.napalm`, or
`.git`. These markers do not imply that a package manager exists to install
dependencies automatically.
Path environment variables accept lists according to the operating system
(`;` on Windows).

For `@lib` and `@std`, `..` and escaping the configured root through symbolic
links are rejected. Modules are identified by canonical path and initialized
once per execution. Cyclic graphs can be discovered, but do not depend on
values that a module in the cycle has not yet initialized.

### 23.3 Types Across Modules

Type-only import/export syntax is accepted and erased.
An interface or alias contract is not imported to check another module.
Declare locally any schemas that should apply to received data.
A `declare` is neither an import nor a call to a native library.

## 24. Strings

### 24.1 Unicode and Indices

Public length and indexing operations work with **Unicode scalar values**,
not UTF-8 bytes or UTF-16 code units.
An emoji represented by one scalar value occupies one position.
A grapheme composed of multiple scalar values can occupy several positions.

```napalm
const text = "caf\u00e9";
System.print(text.length);          // 4
System.print(text[-1]);             // e with acute accent
System.print(text.indexOf("\u00e9"));// 3
System.print(text.slice(3));        // e with acute accent
System.print("a\u{1f600}".length);   // 2
System.print(text[99]);             // null
```

Strings are immutable. Transforming one produces another string.
Negative indices in `text[index]` count from the end.

### 24.2 Complete Installed API

| Member | Use |
| --- | --- |
| `count`, `length` | Length in Unicode scalar values. |
| `isEmpty` | Whether it is empty. |
| `[index]` | One-scalar string, or null out of range. |
| `+ value` | Concatenation through textual conversion. |
| `contains(text)`, `includes(text)` | Searches for a literal substring; both names are aliases. |
| `startsWith(text, position?)` | Prefix from the specified position. |
| `endsWith(text, endPosition?)` | Suffix up to the specified endpoint. |
| `indexOf(text, start?)` | First index or `-1`. |
| `lastIndexOf(text, position?)` | Last index no later than the position. |
| `padStart(targetLength, padding?)`, `padEnd(targetLength, padding?)` | Pads to a scalar length; default padding is a space. |
| `split(separator?, limit?)` | List of substrings separated by literal text, optionally limited in length. |
| `repeat(count)` | Repeats the string a non-negative integer number of times. |
| `replace(search, replacement)` | Replaces the first literal match. |
| `replaceAll(search, replacement)` | Replaces every non-overlapping literal match. |
| `slice(start, end?)` | Segment with an exclusive end; negatives are relative to the end. |
| `substring(start, end?)` | Clamps negatives to zero and swaps reversed endpoints. |
| `trim()`, `trimStart()`, `trimEnd()` | Removes Unicode whitespace at the corresponding ends. |
| `toUpperCase()`, `toLowerCase()` | Unicode uppercase/lowercase conversion. |
| `toString`, `toString()` | The string itself. |
| `iterate`, `iteratorValue` | Iteration by scalar value. |
| `String.fromCodePoint(number)` | String from a valid Unicode code point. |

```napalm
const text = "napalm";
System.print([text.slice(-3), text.slice(4, 1)]); // [alm, ]
System.print(text.substring(4, 1));              // apa
System.print(text.contains("pal"));              // true
System.print(" \tNapalm\n ".trim().toLowerCase());// napalm
System.print([..."abc"]);                        // [a, b, c]
```

Explicit positions must be integers; there is no general argument conversion
as in JS. `slice`/`substring` operations clamp their endpoints according to
their rules; `indexOf` returns `-1` if no match is found.
Uppercase conversion can change the length.

`padStart`/`padEnd` repeat and truncate the padding at scalar boundaries to
reach the target length. A target no larger than the current length (including
a negative target), or empty padding, leaves the text unchanged.

`split()` returns a one-element List containing the original string.
`split('')` returns individual Unicode scalar values; splitting an empty string
this way returns an empty List. Other separators preserve empty parts at either
end and between adjacent separators. The optional limit is a non-negative
integer: zero returns an empty List, and any excess parts are discarded.

Replacement strings are literal: `$&`, `$1` and other `$` patterns have no
special meaning. An empty search inserts the replacement at the start for
`replace`, or at every scalar boundary (including both ends) for `replaceAll`.
Replacements are not searched again. Text arguments must be strings; lengths,
counts and limits must be finite integers. Invalid arguments and unrepresentable
result sizes produce catchable runtime errors.

```napalm
System.print('7'.padStart(3, '0'));       // 007
System.print('a,b,c'.split(',', 2));      // [a, b]
System.print('é😀'.split(''));            // [é, 😀]
System.print('ha'.repeat(3));             // hahaha
System.print('aba'.replace('a', 'x'));    // xba
System.print('aba'.replaceAll('a', 'x')); // xbx
```

`concat`, regular expressions, and regular-expression methods are not provided.

## 25. Numbers and Randomness

### 25.1 Math Getters

The following are **properties**, not calls:

| Getter | Result |
| --- | --- |
| `abs` | Absolute value. |
| `floor`, `ceil` | Rounding toward negative/positive infinity. |
| `round` | Rounds to nearest integer; ties are rounded away from zero. |
| `sqrt` | Square root. |
| `sin`, `cos`, `tan` | Trigonometry in radians. |
| `log` | Natural logarithm. |
| `exp` | Exponential with base e. |
| `isInteger` | Finite and without a fractional part. |
| `isNan` | Whether it is NaN; this is the exact capitalization. |
| `isInfinity` | Whether it is positive or negative infinity. |
| `toString` | Text representation; also `toString()`. |

```napalm
const value = -2.5;
System.print([value.abs, value.floor, value.ceil, value.round]);
System.print((9).sqrt);       // 3
System.print(Num.pi);
System.print(Num.tau);
System.print(Num.fromString(" 42.5 ")); // 42.5
System.print(Num.fromString("no"));    // null
System.print((0 / 0).isNan);          // true
System.print((1 / 0).isInfinity);     // true
```

Num static API: `fromString(text)`, `pi`, `tau`, `infinity`, `nan`.
`fromString` uses textual numeric conversion; it is not a radix-based
`parseInt`.
There is no standard `Math` namespace, `parseInt`, `parseFloat`, `BigInt`, or
exact decimal arithmetic.
Division by zero and invalid mathematical domains can produce Infinity/NaN
instead of throwing an exception.
Arbitrary integers above `2^53 - 1` can lose precision.

### 25.2 Random

```napalm
const random = new Random();
const digit = random.int(10); // 0 <= digit < 10
const die = random.int(1, 7); // 1 <= die < 7
System.print([digit, die]);
```

`new Random()` takes no arguments.
`int(upper)` uses zero as the lower bound.
`int(lower, upper)` includes lower and excludes upper.
Bounds must be finite safe integers, with lower less than upper.
There is no public seed API or cryptographic guarantee for these numbers.

## 26. System and Output

| API | Behavior |
| --- | --- |
| `System.print(value)` | Prints with a line break; uses the text protocol and returns value. |
| `System.print()` | Prints an empty line; returns null. |
| `System.write(value)` | Prints without a line break and flushes stdout; returns value. |
| `print(value)` | Global one-argument shortcut. |
| `System.clock` | Monotonic seconds elapsed since host creation. |
| `System.os()` | Operating system as a String, e.g. `"windows"`, `"linux"`, `"macos"`. See [section 36](#36-platform-detection). |
| `System.arch()` | CPU architecture as a String, e.g. `"x86_64"` or `"aarch64"`. See [section 36](#36-platform-detection). |
| `System.gc()` | Requests explicit garbage collection. |

```napalm
const started = System.clock;
System.write("Hola");
System.print(" Napalm");
System.print([System.os(), System.arch()]);
System.print(`Transcurridos: ${(System.clock - started) * 1000} ms`);
```

**`System.clock` is in seconds**, not milliseconds, despite some benchmark
labels. For wall-clock time in milliseconds, use `Date.now()`.

`System.print` participates in user-defined `toString()`; the `print` shortcut
and `System.write` use the native formatter. To ensure custom conversion,
use `System.print(value)` or prepare a string through interpolation.
`console.log`, stdin, filesystem, and environment variables are not provided
as general language APIs.

Napalm value memory is managed automatically. `System.gc()` does not close
servers, libraries, or memory obtained through a C API; those resources have
their own rules.

## 27. Dates

Date is a native builtin; it does not need a special module import.
It stores a timestamp in **milliseconds since the Unix epoch**.

### 27.1 Constructors and Static Methods

| API | Use |
| --- | --- |
| `new Date()` | Current time. |
| `new Date(milliseconds)` | Timestamp. |
| `new Date(text)` | Parses a supported date. |
| `new Date(otherDate)` | Copies the timestamp. |
| `new Date(year, month, day?, hours?, minutes?, seconds?, ms?)` | Components in the local time zone. |
| `Date.UTC(year, month, day?, hours?, minutes?, seconds?, ms?)` | Timestamp from UTC components. |
| `Date.parse(text)` | Timestamp or NaN. |
| `Date.now()`, `Date.now` | Current timestamp in ms. |
| `Date.nowMs()`, `Date.nowMs` | Alias in ms. |
| `Date.sleep(ms)` | Blocks the current thread; returns null. |

Months start at **zero**; days of the month start at one.
Omitted components default to day 1 and hour/minute/second/ms 0.
Years 0..99 in component-based construction retain the historical adjustment
to 1900..1999. Component overflows are normalized.

```napalm
const date = new Date(Date.UTC(2024, 1, 29, 13, 45, 30, 250));
System.print(date.toISOString()); // 2024-02-29T13:45:30.250Z
System.print(date.getUTCMonth()); // 1
date.setUTCMonth(11, 25);
System.print(date.toISOString()); // 2024-12-25T13:45:30.250Z
```

### 27.2 Getters

All of the following are calls with no arguments:

| Local | UTC | Data |
| --- | --- | --- |
| `getFullYear()` | `getUTCFullYear()` | Full year. |
| `getMonth()` | `getUTCMonth()` | Month 0..11. |
| `getDate()` | `getUTCDate()` | Day of month. |
| `getDay()` | `getUTCDay()` | Day of week, Sunday 0. |
| `getHours()` | `getUTCHours()` | Hour. |
| `getMinutes()` | `getUTCMinutes()` | Minutes. |
| `getSeconds()` | `getUTCSeconds()` | Seconds. |
| `getMilliseconds()` | `getUTCMilliseconds()` | Milliseconds. |

Also: `getTime()` and `valueOf()` return the timestamp;
`getTimezoneOffset()` returns the offset in minutes according to the Date
convention; `getYear()` is the historical year minus 1900 form.

### 27.3 Setters

They return the resulting timestamp:

| Local / UTC | Arguments |
| --- | --- |
| `setFullYear` / `setUTCFullYear` | `year, month?, day?` |
| `setMonth` / `setUTCMonth` | `month, day?` |
| `setDate` / `setUTCDate` | `day` |
| `setHours` / `setUTCHours` | `hour, minute?, second?, ms?` |
| `setMinutes` / `setUTCMinutes` | `minute, second?, ms?` |
| `setSeconds` / `setUTCSeconds` | `second, ms?` |
| `setMilliseconds` / `setUTCMilliseconds` | `ms` |

`setTime(timestamp)` and the historical `setYear(year)` are also available.

### 27.4 Text, Parsing, and Invalid Dates

Available methods:
`toString()`, `toUTCString()`, `toGMTString()`, `toISOString()`,
`toDateString()`, `toTimeString()`, `toLocaleString()`,
`toLocaleDateString()`, `toLocaleTimeString()`, and `toJSON()`.
The `toString` getter is also retained.

Locale forms accept zero, one, or two arguments for compatibility, but do not
implement Intl: explicit locale/options objects do not constitute a complete
configurable formatting API. Local output depends on the host.
`toJSON` also accepts one ignored argument.

The parser supports, among others, ISO dates with UTC/offset, ISO dates without
a time (treated as UTC), and the HTTP/RFC date formats used by the example.
It does not promise to interpret every string accepted by a browser.
The TimeClip limit of `8.64e15` ms is applied.

```napalm
const invalid = new Date("not a date");
System.print(invalid.getTime().isNan); // true
System.print(invalid.toString());      // Invalid Date
System.print(invalid.toJSON());        // null
try {
    invalid.toISOString();
} catch (error) {
    System.print(error); // RangeError: Invalid time value
}
```

Compare and subtract dates through `getTime()`; do not assume implicit
conversion of Date in numeric operators.

## 28. Proxy

Proxy intercepts operations on an object. The supported practical case is a
Map target and a Map handler with synchronous traps.

```napalm
const target = {value: 10};
const events = [];
const proxy = new Proxy(target, {
    get: (object, key, receiver) => {
        events.add(`get:${key}`);
        return object[key];
    },
    set: (object, key, value, receiver) => {
        events.add(`set:${key}`);
        object[key] = value;
        return true;
    },
    has: (object, key) => key in object,
    deleteProperty: (object, key) => {
        return delete object[key];
    },
    ownKeys: (object) => object.keys()
});
System.print(proxy.value);
proxy.value = 20;
System.print(target.value);
System.print("value" in proxy);
System.print(proxy.entries());
System.print(events);
```

| Trap | Signature |
| --- | --- |
| `get` | `(target, key, receiver)` |
| `set` | `(target, key, value, receiver)` |
| `has` | `(target, key)` |
| `deleteProperty` | `(target, key)` |
| `ownKeys` | `(target)`; return a list of keys. |

An absent or null trap uses the default behavior.
A non-function or async/concurrent trap produces an error.
Bracket reads and writes also pass through get/set.
`keys()`, `values()`, `entries()`, and object spread use the traps.
`ownKeys` preserves the order returned by the handler; ordinary Maps sort the
results of their `keys()`/`values()`/`entries()` helpers.

Limitations compared with JS Proxy:

- There is no `Proxy.revocable`, `Reflect`, or complete list of traps
  (`apply`, `construct`, descriptors, prototypes, etc.).
- An intercepted assignment can return the result of `set`, not necessarily
  the assigned value.
- Fallbacks are Map-oriented; do not use Proxy as a universal transparent
  wrapper for classes, lists, and functions.
- Proxy builtin members should not be confused with intercepted keys.
- Do not promise that a Proxy satisfies a structural Map contract.
- A Proxy cannot cross a concurrent boundary.

## 29. HTTP Client and fetch

### 29.1 fetch

`fetch(url, options?)` returns a Promise<Response>. It supports the native
HTTP/HTTPS client, not a browser context.

Recognized options:

| Option | Values |
| --- | --- |
| `method` | String; defaults to GET. |
| `headers` | Map of strings, list of `[name, value]` pairs, or Headers from another response. |
| `body` | String, Map, List, or null. |
| `redirect` | `"follow"` by default, `"manual"`, or `"error"`. |

A String body defaults to `text/plain;charset=UTF-8`;
Map/List is serialized as JSON and uses `application/json`, unless Content-Type
was specified. GET and HEAD do not accept a body.
CONNECT, TRACE, and TRACK are rejected.

Self-contained example using a temporary local server:

```napalm
const server = listen "127.0.0.1:0" {
    route "POST" ["/echo"] (request) => ({
        body: request.body,
        headers: {"Content-Type": "application/json"}
    });
};
try {
    const response = await fetch(`http://${http.address(server)}/echo`, {
        method: "POST",
        body: {name: "Ada", active: true}
    });
    System.print(response.status); // 200
    System.print(response.ok);     // true
    const data = await response.json();
    System.print(data.name);       // Ada
} finally {
    http.close(server);
}
```

HTTP error statuses resolve to a normal Response with `ok == false`;
transport errors or invalid requests reject.
Automatic redirect following is limited to 20.
The response is fully buffered, with a limit of **64 MiB**.
The returned Promise can be canceled.

JSON bodies must be serializable data: scalars, lists, and Maps with string
keys; not functions or cyclic graphs. Depth is limited.
The global [`JSON`](#38-json) utility also exposes `parse` and `stringify`.

### 29.2 Response and Headers

| Response member | Use |
| --- | --- |
| `status` | Numeric code. |
| `statusText` | Status text. |
| `ok` | Status from 200 to 299. |
| `url` | Final URL. |
| `redirected` | Whether a redirect was followed. |
| `headers` | Stable Headers object. |
| `bodyUsed` | Whether consumption/decoding has started. |
| `data` | Decoded and cached view; JSON according to Content-Type or text. |
| `json()` | Promise with decoded JSON. |
| `text()` | Promise with text. |

`headers.get(name)` returns a string or null.
`headers.has(name)` returns Bool. Name lookup is case-insensitive.
There is no general Headers constructor or complete mutation API.

The body is consumed **only once**:

- After `json()` or `text()`, another consumption call rejects.
- `data` consumes the body on first access and caches it; subsequent reads
  reuse the value.
- If successful decoding has already occurred, `data` exposes that result.
- After reading `data`, do not call `json()` expecting a second consumption.
- A JSON parsing error also consumes the body.
- `data` can throw synchronously; `json()`/`text()` report failure through
  their Promise.

There is no client ReadableStream, `arrayBuffer()`, `blob()`, `formData()`,
`clone()`, AbortController, or promise of complete WHATWG Fetch compatibility.
Undocumented options should not be assumed to be implemented.

### 29.3 HTTP Helpers

| Function | Expected result when awaited |
| --- | --- |
| `get(url, headers?)` | Map `{status, ok, body, headers}`. |
| `getJSON(url, headers?)` | Lo anterior mas `json`. |
| `postJSON(url, jsonText, headers?)` | POST and response with `json`. |

These helpers return **Maps**, not Response.
`postJSON` accepts an **already serialized** JSON string, unlike the Map/List
body used by fetch. Invalid response JSON rejects.
Response header names are normalized to lowercase.
The helpers buffer the entire body, including a streaming response.

## 30. HTTP Servers, Streaming, and SSE

### 30.1 listen and route

```napalm
const server = listen "127.0.0.1:0" {
    route "GET" ["/", "/hello"] (req) => {
        return {status: 200, body: "Hello!"};
    };
    route "GET" ["/users/:id"] async (req) => {
        await Timer.delay(1);
        return {body: `User ${req.params.id}`};
    };
    route "POST" ["/echo"] (req) => ({
        body: req.body,
        headers: [["X-Echo", "1"]]
    });
};
try {
    const result = await get(`http://${http.address(server)}/users/42`);
    System.print(result.body); // User 42
} finally {
    http.close(server);
}
```

`listen address { ... }` is an expression that returns a numeric handle.
It can also be used as a statement.
Port zero requests an available port; `http.address(handle)` reports the
actual `host:port`. `0.0.0.0` listens on all IPv4 interfaces; use
`127.0.0.1` if it should be accessible only locally.
Bind errors are recoverable with try/catch.

The equivalent API for dynamically constructed routes is:

```napalm
const handler = (req) => ({body: "ok"});
const server = http.listen("127.0.0.1:0", [
    ["GET", ["/"], handler]
]);
try {
    System.print((await get(`http://${http.address(server)}/`)).body);
} finally {
    http.close(server);
}
```

### 30.2 Route Selection

- Declaration order: the first match wins.
- Methods are normalized to uppercase; `"*"` accepts any method.
- Paths are case-sensitive and do not include the query.
- `:name` captures a non-empty segment.
- A final `*` captures the remainder as `req.params.wildcard`.
- Duplicate method/path pairs are rejected.
- No matching path: 404.
- Existing path with the wrong method: 405 and `Allow`.
- HEAD and OPTIONS require an explicit route or `"*"`.
- There is no automatic HEAD->GET fallback or automatic CORS.

Declare specific routes before parameters and wildcards.

### 30.3 Request

| Field | Contents |
| --- | --- |
| `method` | HTTP method. |
| `path` | Path without query. |
| `url` | Request target with query. |
| `query` | Raw query without `?`. |
| `params` | Path captures; not percent-decoded. |
| `headers` | Map of lowercase names to strings; the last repetition wins. |
| `body` | Buffered text, UTF-8 with invalid bytes replaced. |
| `serverId`, `reqId` | Handles used for streaming. |

The body is not automatically parsed as JSON, and `query` is not a parameter
Map. There is no middleware, session, or authentication framework here.

### 30.4 Handler Response

A string can be returned, equivalent to status 200, or an object:

```napalm fragment
return {
    status: 201,
    body: "{\"created\":true}",
    headers: [["Content-Type", "application/json"]]
};
```

- Omitted status: 200; omitted body: empty string.
- Status must be an integer from 200 to 599.
- Body and header values must be strings.
- Headers accepts a Map or list of pairs; pairs preserve repetitions.
- Content-Type must be set explicitly for JSON, HTML, or SSE.
- A Map is not automatically serialized as the response body.
- Thrown errors, rejections, and invalid responses produce 500 without
  revealing the exception to the client.
- Other requests continue to be served after a handler error.

Returning null or zero is valid only after starting a stream.
Once started, normal returns are ignored; an exception terminates that stream.

### 30.5 Streaming and SSE

```napalm
const server = listen "127.0.0.1:0" {
    route "GET" ["/events"] (req) => {
        http.startStream(req, 200, [
            ["Content-Type", "text/event-stream"],
            ["Cache-Control", "no-cache"]
        ]);
        setTimeout(() => {
            http.streamWrite(req.serverId, req.reqId, "data: hello\n\n");
            http.streamEnd(req.serverId, req.reqId);
        }, 1);
        return 0;
    };
};
try {
    const result = await get(`http://${http.address(server)}/events`);
    System.print(result.body); // data: hello and two line breaks
} finally {
    http.close(server);
}
```

| API | Effect |
| --- | --- |
| `http.startStream(req, status, headers)` | Sends headers once. |
| `http.streamWrite(serverId, reqId, chunk)` | Queues a UTF-8 chunk. |
| `http.streamEnd(serverId, reqId)` | Closes; returns whether it was open. |
| `http.close(serverId)` | Stops accepting connections; true once, false if already closed/unknown. |
| `http.address(serverId)` | Effective address. |

A chunk is limited to **64 KiB**, and each stream allows up to **64 queued
chunks**. A full queue throws a recoverable error: retry later using a
timer/await, rather than spinning in a blocking loop.
Empty chunks are ignored; writing to a closed/disconnected stream fails.

The server manages Content-Length, Transfer-Encoding, Connection, Trailer, and
Upgrade; do not supply those headers from the handler.
Closing the listener does not force already-dispatched handlers or streams to
finish.
Also close streams and cancel timers that should terminate.

### 30.6 Transport Limits

HTTP/1.0 and HTTP/1.1 over TCP, one request per connection.
There is no HTTPS listener, HTTP/2 server, or WebSockets.

| Limit | Value |
| --- | --- |
| Buffered request body and normal response | 8 MiB. |
| Active connections across VM listeners | 256. |
| Headers per request | 100. |
| Header/body read and handler wait | 30 s. |
| Maximum connection lifetime, including streams | 5 min. |

An oversized body receives 413, a body timeout 408, and a handler response
timeout 504. A long synchronous handler blocks dispatch of other handlers and
timers; async does not automatically make that computation parallel.

## 31. FFI and External Functions

### 31.1 Loading a C Library

FFI allows calling C symbols from shared libraries.
It does not import headers or infer signatures.

Example portable across common Windows/Linux/macOS installations
(requires that library to exist on the system):

```napalm
const os = System.os();
const libraryName = os == "windows" ? "ucrtbase.dll"
    : os == "macos" ? "/usr/lib/libSystem.B.dylib" : "libc.so.6";
const library = FFI.open(libraryName);
const length = library.function("strlen", "usize", ["cstring"]);
System.print(length.call("Napalm")); // 6
```

Do not copy an approximate function signature: **an incorrect ABI signature
can corrupt memory or terminate the process**. Napalm type validation cannot
detect that the actual C symbol has a different signature.

### 31.2 API

| Object | Member |
| --- | --- |
| FFI | `open(path)`, getter `null`, getter `lastError`. |
| Library | `name` getter; `function(symbol, resultType, argumentTypes)`. |
| Native function | `name` getter; `call(...)` with 0 to 16 arguments. |
| Pointer | `isNull` getter; `offset(bytes)`; `readCString()`; `toString`/`toString()`. |

A library remains loaded until the interpreter is destroyed; there is no
manual library close. `FFI.null` is ordinary null.
A returned C NULL pointer is also converted to null, not a wrapper with
`isNull == true`.

`FFI.lastError` captures errno/GetLastError immediately after the call.
Check it only when that function's contract indicates an error occurred:
success does not necessarily clear the previous value.

### 31.3 Supported Types

| FFI type | Napalm value |
| --- | --- |
| `void` | Return only; produces null. |
| `bool` | Bool; one-byte C boolean. |
| `i8`, `u8`, `i16`, `u16`, `i32`, `u32` | In-range integer Num. |
| `i64`, `u64`, `isize`, `usize` | Integer Num, only if exactly representable. |
| `char`, `unsigned char`, `short`, `unsigned short` | Num with the corresponding ABI size. |
| `int`, `unsigned int`, `long`, `unsigned long` | Num with the corresponding ABI size. |
| `long long`, `unsigned long long` | Exact Num. |
| `f32`/`float`, `f64`/`double` | Num. |
| `pointer` | Opaque pointer or null. |
| `cstring` | Copied NUL-terminated C string, or null. |

Aliases such as `int32_t`, `uint64_t`, `size_t`, `ssize_t`,
`intptr_t`, `uintptr_t`, `c_int`, `c_long`, `ptr`, `void*`, `char*`, and
`const char*`.

Integer arguments must be finite, integral, in range, and exactly
representable. Integer results that would lose precision are rejected.
A pointer is not converted to a 64-bit Num.

### 31.4 Memory Lifetime and Strings

- For a cstring argument, temporary memory valid during the call is created.
- Strings containing embedded NUL are rejected.
- Do not retain a pointer returned by the native function into that temporary
  memory. Declare a cstring return to copy it immediately when appropriate.
- A cstring result is copied; ownership of native memory is not automatically
  transferred and `free` is not called.
- For memory the caller must free: receive a pointer, read/copy it, and call
  the deallocator provided by the same library.
- `offset(bytes)` only computes an address; it does not check whether valid
  memory exists there.
- `readCString()` copies up to NUL, with a 16 MiB limit. An invalid pointer
  remains unsafe.
- `defer` can organize normal deallocation, but cannot fix an incorrect ABI.

There are no C->Napalm callbacks, variadic calls, structs/unions by value,
direct access to the Napalm heap, header parser, C++ methods, or manual
unloading.
FFI calls are synchronous and are not automatically sent to a worker.

### 31.5 foreign Is a Separate Integration

`foreign class` and `foreign` methods describe extension points that can be
implemented by the **host embedding Napalm**. They do not automatically call
`FFI.open` or resolve a library symbol. In the ordinary CLI, a foreign
declaration without a supporting host does not provide an implementation.
See [README_FFI.md](README_FFI.md) for the complete C contract.

## 32. Limitations and Differences from JavaScript

### 32.1 Practical Summary

| Do not assume | In Napalm |
| --- | --- |
| Any `.ts` file is executable. | Only supported syntax; modules use `.napalm`. |
| `0`, `""`, or NaN are falsy. | Only null/false are falsy in conditions. |
| `!!value` converts any value. | `!` requires an operation defined on the receiver. |
| `===` is stricter than `==`. | They are the same operation. |
| `+text` converts to a number. | Prefix `+` does not convert. |
| `var` has function scope. | It is lexical/block-scoped. |
| `let x;` is a valid declaration. | Variables require an initializer. |
| No annotation means any. | Inference can be restrictive. |
| All types are erased. | Variables, parameters, fields, and contracts can be validated. |
| A return annotation guarantees the return value. | That annotation is not currently checked. |
| `as` converts or validates. | It does not change the value. |
| Structural `interface` and `type` are interchangeable. | Interface is open; object alias is closed. |
| `implements` lets an instance bind to an interface annotation. | It adds and checks fields; interface/type annotations still require Maps. Test instances with `instanceof`. |
| Interfaces and types can declare methods. | They are data-only; use a class for methods. |
| Class `const` or private/readonly freezes a value. | No; structural readonly is a different case. |
| `Array(n)` creates zeros. | It creates null; `Array<Number>(n)` creates zeros. |
| `map.keys` and `map.keys()` are the same. | Live insertion-order view vs. sorted list. |
| Map iterates `[key, value]`. | It iterates MapEntry; `entries()` produces pairs. |
| All JS array/string methods exist. | Only those listed in this manual. |
| An empty derived constructor is equivalent to JS. | Use an explicit constructor and super. |
| A closure has dynamic this. | `this` is lexical. |
| async means parallel CPU work. | Use launch with concurrent functions for that. |
| Promise has then/catch/executor. | Use async/await and installed factories/combinators. |
| EventEmitter or MessageChannel must be imported from Node/browser APIs. | Napalm provides native global Event/Emitter and Channel/Receiver endpoints. |
| `System.clock` measures ms. | It measures seconds; timers and Date use ms. |
| A broken contract is caught. | Contract errors are fatal. |

### 32.2 Unsupported or Non-Public Syntax

Do not use the following as though they were available features:

- `switch`, async generators and `for await...of`, regex literals,
  JSX, executable namespaces, and `const enum`. (Synchronous generators are
  supported; see [section 22.2](#222-generators).)
- Generator methods in object literals (`{ *items() {} }`); object literals
  do not have method shorthand at all.
- `void` as an operator, and `>>>`.
- Rest parameters and full default-initializer parameters.
- The general comma operator and `matrix[x, y]` subscripts.
- Namespace imports, export-star, dynamic import, and require.
- `#private` fields, object descriptors, and the JS prototype API.
- Wren syntax `construct`, `is`, `..`/`...` ranges, trailing block closures
  on calls, and imports such as `import "x" for Name`.
- `fn` declarations, `else unless`, and Deferred APIs from old scripts.

The presence of a token or internal test does not mean the public parser
accepts it as a Napalm program.

### 32.3 Structural Limits

| Limit | Value or rule |
| --- | --- |
| Normal parameters and arguments | 16. |
| Ordinary arguments to a decorated method | 15, plus explicit receiver. |
| Parsing depth | 256. |
| Interpolation nesting | Frontend-limited; depth constant 8. |
| Concurrent transfer | Transferable acyclic graphs, depth 256. |
| Recursive interfaces/aliases | Not resolved as recursive contracts. |
| Interface imports as contracts | Not implemented. |
| User generics | No complete substitution/reification. |
| Browser/Node runtime | Not provided. |
| SQLite/UI | No public API is installed merely because files or internal dependencies are present. |

The names `Object`, `Class`, `Bool`, `Null`, etc. are part of the environment's
builtin values, but do not imply constructors/converters or every static
function on the JavaScript object with the same name.
Do not rely on helpers prefixed with `__` as a public API.

## 33. Complete Recipes

### 33.1 Transforming Data with a Closed Schema

```napalm
type Person = {name: string; age: number};
const people: Person[] = [
    {name: "Alice", age: 22},
    {name: "Bob", age: 40},
    {name: "Charlie", age: 55},
    {name: "Dave", age: 14}
];
const adults = [
    {name: person.name, age: person.age}
    for person in people
    if person.age >= 30
];
System.print(adults.map(person => person.name)); // [Bob, Charlie]
```

### 33.2 Capturing a Value per Iteration

```napalm
function capture(value) {
    return () => value;
}
const callbacks = [];
for (let i = 0; i < 3; i++) {
    callbacks.add(capture(i));
}
System.print(callbacks.map(callback => callback())); // [0, 1, 2]
```

### 33.3 Deduplicating While Preserving Order

```napalm
function unique(values) {
    const seen = new Set();
    for (const value of values) seen.add(value);
    return [...seen];
}
System.print(unique(["b", "a", "b", "c"])); // [b, a, c]
```

### 33.4 Recoverable Input Validation

A contract annotation is fatal if violated. To report a recoverable business
error, perform explicit validation and throw a value:

```napalm
function requireName(input) {
    if (input.name == null || input.name == "") {
        throw {code: "MISSING_NAME", message: "Falta el nombre"};
    }
    return input.name;
}
try {
    System.print(requireName({}));
} catch (error) {
    System.print(error.code); // MISSING_NAME
}
```

This function assumes an input that supports object access; it is not a
universal validator for every value. Keep business validation, data contracts,
and transport errors separate.

### 33.5 Bounded Asynchronous Retry

```napalm
let attempts = 0;
async function operation() {
    attempts++;
    if (attempts < 3) throw "temporal";
    return 42;
}
async function retry() {
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            return await operation();
        } catch (error) {
            if (attempt == 2) throw error;
            await Timer.delay(1);
        }
    }
}
System.print(await retry()); // 42
```

Do not indiscriminately catch errors to return fictitious success.
Fatal contract errors do not fit this retry pattern.

## 34. Example and Source Map

### 34.1 Useful Scripts as Guides

The scripts were reviewed for their language content. A benchmark is neither a
semantic specification nor a universal performance test.

| File | Content and notes |
| --- | --- |
| [basic.napalm](scripts/basic.napalm) | Variables, addition, and output. |
| [bool.napalm](scripts/bool.napalm) | Booleans, equality, and control flow. |
| [builtins.napalm](scripts/builtins.napalm) | System, interpolation, and closures. |
| [complete.napalm](scripts/complete.napalm) | Tour of functions, classes, ranges, async, and Fiber; its opening comment about erased types is historical. |
| [comprehensions.napalm](scripts/comprehensions.napalm) | Filtering people with a comprehension and structural alias. |
| [concurrent.napalm](scripts/concurrent.napalm) | launch and Promise.all benchmark; its ms labels do not convert System.clock from seconds to ms. |
| [date.napalm](scripts/date.napalm) | Constructors, getters/setters, parsing, and invalid dates; Date is already a builtin, not an auto-imported module. |
| [decorators.napalm](scripts/decorators.napalm) | Executable tests of all decorator targets and application order. |
| [native_decorators.napalm](scripts/native_decorators.napalm) | Tests for caching, TTL, once, delay, debounce, and throttle. |
| [destructuring_spread.napalm](scripts/destructuring_spread.napalm) | Patterns, defaults, rest, spread, and calls. |
| [enums.napalm](scripts/enums.napalm) | Numeric and string enums, reverse mapping, and local enum. |
| [generators.napalm](scripts/generators.napalm) | Self-checking tour of generators: laziness, `next(value)`, `yield*`, `return()`/`throw()`, cleanup, and `for...of` closing. |
| [interfaces.napalm](scripts/interfaces.napalm) | Open contracts, aliases, and valid mutation. |
| [types.napalm](scripts/types.napalm) | Classes implementing interfaces and types, instance `keys()`, and the `instanceof`/`implements`/`inherits` operators. |
| [intrinsics.napalm](scripts/intrinsics.napalm) | Arrays, lambdas, Unicode, strings, nullish operations, and optional chaining. |
| [map.napalm](scripts/map.napalm) | Map, Set, and native generics. |
| [proxy.napalm](scripts/proxy.napalm) | Traps and enumeration/spread. |
| [reactive.napalm](scripts/reactive.napalm) | Accessors and cached computed properties. |
| [spread.napalm](scripts/spread.napalm) | Object destructuring/rest/spread. |
| [import.napalm](scripts/import.napalm) | Named and side-effect imports. |
| [module_import.napalm](scripts/module_import.napalm) | Default import combined with named imports. |
| [module_exports.napalm](scripts/module_exports.napalm) | Auxiliary module with exports. |
| [math.napalm](scripts/math.napalm), [other.napalm](scripts/other.napalm) | Auxiliary modules with functions and classes. |
| [server.napalm](scripts/server.napalm) | Service with routes, local clients, and SSE; continues listening. |
| [server_simple.napalm](scripts/server_simple.napalm) | Minimal server; continues listening. |
| [fetch.napalm](scripts/fetch.napalm) | POST example; the external endpoint is illustrative, not a reproducible local test. |
| [lev.napalm](scripts/lev.napalm) | Algorithm using strings, arrays, and loops. |
| [trees.napalm](scripts/trees.napalm) | Recursion and nested objects. |
| [fib.napalm](scripts/fib.napalm) | Recursion and timing; benchmark workload, not a minimal example. |
| [runtime.napalm](scripts/runtime.napalm) | Loops of list/string operations; benchmark workload. |
| [gc.napalm](scripts/gc.napalm) | Mutation through aliases and GC request; learning the collector is not required to use the language. |

You do not need to examine [jit.napalm](scripts/jit.napalm) or
[jit_entries.napalm](scripts/jit_entries.napalm) to learn language features;
they are engine-specific benchmarks.

### 34.2 Historical Material Not to Copy Literally

| File | Observed difference |
| --- | --- |
| [index.napalm](scripts/index.napalm) | Mixes `fn`, `else unless`, uninstalled JS methods, Promise executor, and Deferred. Its use of `|>` is now valid, but the other historical constructs still prevent the file from compiling unchanged. |
| [trycatch.napalm](scripts/trycatch.napalm) | Uses `fn` declarations; the manual's exception examples use `function`. |
| [defer.napalm](scripts/defer.napalm) | The `errdefer (error)` binding must be written with bars. |

Those programs have not been modified to hide differences. This manual
documents the available language and shows usable alternatives.

### 34.3 Semantic Sources

| Source | Surface consulted |
| --- | --- |
| [napalm_front_main.rs](src/napalm_front_main.rs) | Public parser: syntax, declarations, enums, types, optional chaining, classes, decorators, and limits. |
| [napalm_front.rs](src/napalm_front.rs) | Shared AST/limit definitions; its historical parser alone does not determine public syntax. |
| [napalm_lowerer.rs](src/napalm_lowerer.rs) | Relevant language rules: inference, parameters, fields, contracts, and launch restrictions. |
| [napalm_builtin.rs](src/napalm_builtin.rs) | Effective API catalog and primitive semantics. |
| [napalm_interpreter.rs](src/napalm_interpreter.rs) | Observable behavior: values, calls, errors, async/Fiber, Proxy, reactivity, transfers, and fetch. |
| [napalm_interfaces.rs](src/napalm_interfaces.rs) | Validation of open/closed schemas and mutation protection. |
| [napalm_collection_ops.rs](src/napalm_collection_ops.rs) | Collection copies and validation, destructuring, and spread. |
| [napalm_modules.rs](src/napalm_modules.rs) | Actual resolution of extensions, namespaces, and exports. |
| [napalm_native.rs](src/napalm_native.rs) | Native decorator semantics and validation. |
| [napalm_generators.rs](src/napalm_generators.rs), [napalm_generator_runtime.rs](src/napalm_generator_runtime.rs) | Generator desugaring (`function*`, `yield`, `yield*`) and the `next`/`return`/`throw` runtime. |
| [napalm_concurrent.rs](src/napalm_concurrent.rs) | Concurrent work boundary and cancellation. |
| [napalm_fetch.rs](src/napalm_fetch.rs) | Client transport, redirects, and limits. |
| [napalm_http.rs](src/napalm_http.rs), [napalm_http_vm.rs](src/napalm_http_vm.rs) | HTTP API and transport. |
| [napalm_ffi.rs](src/napalm_ffi.rs), [README_FFI.md](README_FFI.md) | Types and ownership at the C boundary. |
| [main.rs](src/main.rs) | CLI and exit codes. |
| [napalm_tests.rs](src/tests/napalm_tests.rs) | Public-language regressions. |
| [napalm_collections_tests.rs](src/tests/napalm_collections_tests.rs) | Collections and contracts. |
| [napalm_http_tests.rs](src/tests/napalm_http_tests.rs) | Routes and local clients. |

Historical tests in [src/tests](src/tests) and inside the interpreter use a
different frontend. Their programs were checked against the public parser
before bringing any syntax form into this manual.

Supplementary documents:
[interfaces](docs/INTERFACES.md), [collections](docs/COLLECTIONS.md),
[HTTP](docs/HTTP.md) and [platform](README_SYSTEM_PLATFORM.md).
Some describe earlier revisions: parameter types, closed aliases, extensions,
and `keys()` in particular should be read with the qualifications in this
reference.

### 34.4 Verification Scope

Standalone examples are checked against the local release executable; blocks
marked as errors must fail, not pass.
The manual's HTTP examples use loopback and ephemeral ports and close the
listener. The FFI example depends on the specified platform library.
The illustrative external POST is not run, and example servers are not left
open.

Useful filters for existing tests include:

```powershell
cargo test napalm_decorators -- --test-threads=1
cargo test native_decorators -- --test-threads=1
cargo test napalm_collections -- --test-threads=1
cargo test napalm_http -- --test-threads=1
cargo test napalm_generators
```

These commands are entry points to regression tests, not a claim that the
entire historical suite passes on every revision. A backend test using old
syntax does not replace checking a `.napalm` file with the current CLI.

## 35. Current Performance

Current engine, as of 02-Nov-2026, shows an impressive performance that can compete against V8 and JSC. Although there are not a lot of benchmarks right now, these are the numbers of the existing benchmarks as of today.


| Test | Napalm Performance | Node Performance | Bun Performance |
| --- | --- | --- | --- |
| `jit` | 0.172551s | 0.187731s | 0.223191s |
| `jit_entries` | 0.002747s |  0.005001s | 0.004753s |
| `lev` | 0.000217s | 0.000710s | 0.001525s |
| `trees` | 0.004445s | 0.005114s | 0.003302s |
| `fib` | 0.000122s | 69.66374s | 32.97240s |

(Note: the impressive results in Fibonacci test are due to one of Napalm HIR optimization: the auto-memoization of pure functions with simple arguments).

## 36. Platform Detection

`System.os()` and `System.arch()` tell a program which platform it is running
on, so it can choose platform-specific paths, library names or commands.

| API | Returns |
| --- | --- |
| `System.os()` | The operating system, as a lowercase String. |
| `System.arch()` | The CPU architecture, as a lowercase String. |

```napalm
System.print([System.os(), System.arch()]); // [windows, x86_64] on 64-bit Windows

const separator = System.os() == "windows" ? "\\" : "/";
const is64BitArm = System.arch() == "aarch64";
```

Possible values:

| `System.os()` | Platform |
| --- | --- |
| `"windows"` | Windows |
| `"linux"` | Linux |
| `"macos"` | macOS |
| `"freebsd"`, `"netbsd"`, `"openbsd"`, `"dragonfly"` | BSD systems |
| `"android"`, `"ios"` | Mobile systems |

| `System.arch()` | Architecture |
| --- | --- |
| `"x86_64"` | 64-bit Intel/AMD |
| `"aarch64"` | 64-bit ARM (Apple Silicon, ARM servers, Windows on ARM) |
| `"x86"` | 32-bit Intel/AMD |
| `"arm"` | 32-bit ARM |
| `"riscv64"` | 64-bit RISC-V |

The lists are not exhaustive: compare against the values you support and
handle anything else explicitly instead of assuming a default.

Details:

- **They are methods, not properties.** Write `System.os()`, unlike
  `System.clock`. `System.os` without parentheses, or passing any argument,
  throws a runtime error.
- **The values describe the Napalm runtime, not the machine's hardware or OS
  version.** A 32-bit or x86_64 build running under emulation (for example on
  Windows on ARM or with Rosetta on macOS) reports its own architecture. There
  is no API for the OS version or distribution.
- **The values are fixed when the Napalm runtime is built.** The same running
  program always gets the same values. They are never computed during
  `--build`, so a `.burning` artifact moved to another platform reports the
  platform of the runtime that runs it.
- **A standalone executable reports the platform it was compiled on.**
  `--compile` copies the Napalm runtime that ran it, so a program compiled on
  Windows x86_64 always reports `"windows"` and `"x86_64"`. It only runs on
  that platform anyway; to target another platform, run `--compile` with a
  Napalm runtime for that platform.

[Section 31.1](#311-loading-a-c-library) uses `System.os()` to choose the C
library name to load with FFI.

## 37. Napalm native FFI

Napalm includes a small typed C FFI built on `libloading` + `libffi`.
It is separate from Napalm's existing embedding/`foreign` mechanism:

- **`foreign` methods / `Host::call_foreign`** let the application embedding Napalm provide native methods.
- **`FFI`** lets Napalm code open a shared library and call an exported C ABI symbol directly.

### 37.1 Basic API

```typescript
const libc = FFI.open("libc.so.6")
const strlen = libc.function("strlen", "usize", ["cstring"])

System.print(strlen.call("hello")) // 5
```

`library.function(symbol, resultType, argumentTypes)` resolves the symbol once and returns a callable native-function object. Prepared libffi CIFs are cached by signature **per OS thread**, so functions sharing a signature reuse the same call-interface description without moving libffi's non-`Send` descriptors between scheduler workers.

A native function exposes `call(...)` overloads from 0 through 16 arguments, matching Napalm's existing maximum method arity.

### 37.2 Core objects

#### 37.2.1 `FFI`

```typescript
FFI.open(path)       // -> native library
FFI.null             // -> null (explicit spelling for pointer arguments)
FFI.lastError        // -> errno/GetLastError snapshot from the last FFI call
```

`FFI.lastError` is captured immediately after `ffi_call`, before Napalm allocation or formatting can overwrite the OS error state. As with C itself, inspect it only when the called function's contract says an error occurred; successful C functions are not required to clear an old errno/last-error value.

#### 37.2.2 Native library

```typescript
library.name
library.function(symbol, resultType, argumentTypes)
```

Libraries remain mapped until the interpreter is destroyed. Napalm deliberately does not expose `close()` in v1: bound function pointers must never dangle, and MMTk worker threads must not accidentally execute library unload/finalizer code.

#### 37.2.3 Native function

```typescript
function.name
function.call(...)
```

#### 37.2.4 Native pointer

A non-null `pointer` result becomes an opaque FFI pointer object. A native null pointer becomes ordinary Napalm `null`.

```typescript
pointer.isNull          // normally false; native NULL is normalized to Napalm null
pointer.offset(bytes)   // address arithmetic only; does not dereference memory
pointer.readCString()   // copies at most 16 MiB, stopping at NUL
pointer.toString        // hexadecimal address
```

Pointers can be passed back to arguments declared as `pointer`. Napalm never exposes a raw address as `Num`, because a 64-bit pointer cannot in general be represented exactly by IEEE-754 `f64`.

### 37.3 Supported types

Canonical spellings are shown first; several C aliases are accepted.

| Napalm FFI type | C / ABI meaning | Napalm value |
| --- | --- | --- |
| `void` | no result | `null` |
| `bool` | 1-byte C boolean | `Bool` |
| `i8`, `u8` | fixed 8-bit integer | `Num` |
| `i16`, `u16` | fixed 16-bit integer | `Num` |
| `i32`, `u32` | fixed 32-bit integer | `Num` |
| `i64`, `u64` | fixed 64-bit integer | `Num`, only when exact |
| `isize`, `usize` | pointer-sized integer | `Num`, only when exact |
| `char` | target C `char` | `Num` |
| `unsigned char` | C `unsigned char` | `Num` |
| `short`, `unsigned short` | C short | `Num` |
| `int`, `unsigned int` | C int | `Num` |
| `long`, `unsigned long` | C long | `Num`, only when exact |
| `long long`, `unsigned long long` | C long long | `Num`, only when exact |
| `f32` / `float` | C float | `Num` |
| `f64` / `double` | C double | `Num` |
| `pointer` | `void*` | opaque pointer or `null` |
| `cstring` | NUL-terminated `char*` | copied Napalm String or `null` |

Recognized aliases also include common names such as `int32_t`, `uint64_t`, `size_t`, `ssize_t`, `intptr_t`, `uintptr_t`, `c_int`, `c_long`, `ptr`, `void*`, `char*`, and `const char*`.

`void` is valid only as a return type.

### 37.4 Integer precision

Napalm numbers are IEEE-754 doubles. Napalm therefore refuses to silently corrupt wide integer values:

- integer arguments must be finite, integral, in range, and exactly representable by the source Napalm `Num`;
- integer results are converted to `Num` only if conversion is exact;
- otherwise the call reports a runtime error.

This still permits large exactly representable values (for example powers of two above `2^53`), while rejecting arbitrary 64-bit values whose low bits would be lost. Napalm can later grow first-class `i64/u64` source values without changing the native signature model.

### 37.5 C strings

For a `cstring` argument, Napalm creates an owned `CString` before entering native code. Therefore:

- moving/collecting the Napalm String cannot invalidate the pointer during the call;
- embedded NUL bytes are rejected;
- the pointer is valid only for the duration of that call unless the C API explicitly copies it.

If a C function returns a pointer *into* one of its `cstring` arguments (for example a `strchr`-style API), declare the result as `cstring` when you want an immediate copy. Declaring such an interior result as `pointer` and retaining it after the call would outlive Napalm's temporary CString and is therefore invalid.

For a `cstring` result, Napalm immediately copies the bytes into a Napalm String. The native pointer is not retained.

If the native API returns an allocated string that the caller owns, declare it as `pointer`, call `readCString()`, and then pass the pointer to the API's deallocator.

### 37.6 Example: caller-owned native memory

```typescript
const libc = FFI.open("libc.so.6")
const malloc = libc.function("malloc", "pointer", ["usize"])
const free = libc.function("free", "void", ["pointer"])

const p = malloc.call(4096)
if (p == null) Fiber.abort("malloc failed")

System.print(p.toString)
free.call(p)
```

### 37.7 Safety boundary

The FFI validates Napalm values against the **declared** signature. It cannot prove that the declaration matches the actual C symbol. A wrong native signature can still corrupt memory or crash the process; that is inherent to a low-level C FFI.

Likewise, opaque pointer lifetime is owned by the native API. `offset()` performs integer address arithmetic only, but passing an invalid/stale pointer to C or calling `readCString()` on one is unsafe.

### 37.8 Deliberately not in v1

- callbacks from C into Napalm;
- variadic functions;
- structs/unions passed by value;
- direct Napalm heap pointers;
- automatic header parsing;
- C++ name mangling / methods;
- asynchronous/blocking-native-call offload;
- manual library unload.

Callbacks deserve a separate design because a callback may arrive on an arbitrary OS thread and therefore must coordinate with Napalm's GC safepoints, isolate ownership and future N:M scheduler. Struct support should similarly be added with explicit ABI layout objects rather than guessed layouts.

### 37.9 Dependencies

```toml
libffi = "5.2.0"
libloading = "0.9.0"
```

`libffi` uses its bundled libffi build by default. If a deployment deliberately uses a system libffi, enable the crate's `system` feature instead.

## 38. JSON

`JSON` is a global utility; no import or constructor is needed. Both methods
take exactly one argument and run synchronously.

| Method | Result and behavior |
| --- | --- |
| `JSON.parse(input)` | Parses a String, Bytes, or ByteBuffer containing UTF-8 JSON. Returns null, Bool, Num, String, List, or Map according to the JSON value. Binary input is neither modified nor consumed. |
| `JSON.stringify(value)` | Returns a compact JSON String for null, Bool, Num, String, List, or Map. Map keys must be Strings; nested values must also be supported. |

Objects become Maps in source order. Duplicate keys keep their original
position but use the last value. Stringification preserves Map insertion order.
Numbers use Napalm's floating-point representation, so large JSON integers may
lose precision. `stringify` encodes non-finite numbers as `null` and negative
zero as `0`.

Invalid JSON, unsupported values (including Bytes, ByteBuffer, functions, and
class instances), non-String Map keys, and cycles throw errors. Shared children
without cycles are allowed. Nesting is limited to depth 256, with the root at
depth zero; input and output are limited to 256 MiB. There are no reviver,
replacer, indentation, or automatic `toJSON()` hooks.

```napalm
const value = JSON.parse('{"name":"Napalm","ready":true}');
System.print(value["name"]); // Napalm
System.print(JSON.stringify(value)); // {"name":"Napalm","ready":true}

const bytes = Bytes.fromUtf8('[1,2,3]');
System.print(JSON.parse(bytes)[1]); // 2
```

See [scripts/json.napalm](scripts/json.napalm) for further examples.

## 39. Bytes

`Bytes` is an immutable binary value, available globally. Lengths and indices
count bytes, not characters. Create values with the static factories below;
there is no public `new Bytes()` constructor.

### 39.1 Factories

| Method | Result |
| --- | --- |
| `Bytes.empty()` | Empty Bytes. |
| `Bytes.alloc(length)` | Bytes containing `length` zero bytes. |
| `Bytes.fromUtf8(text)` | Bytes copied from a String's UTF-8 encoding. |
| `Bytes.fromArray(values)` | Bytes copied from a List of integer byte values in `0..255`. |

### 39.2 Properties and Methods

| Member | Result and behavior |
| --- | --- |
| `length` | Number of bytes. |
| `isEmpty` | Whether length is zero. |
| `bytes[index]` | Numeric byte at a zero-based index. Out-of-bounds access throws. |
| `slice(start, end?)` | Bytes sharing the original storage for the half-open range `[start, end)`. Omitted `end` means length. Requires `0 <= start <= end <= length`. |
| `clone()` | New Bytes value sharing immutable storage. |
| `toUtf8()` | Decoded String; invalid UTF-8 throws. |
| `toArray()` | New List of numeric bytes; limited to 16,777,216 elements. |
| `toMutable()` | Independent mutable ByteBuffer copy. |
| `indexOf(pattern)` | First matching byte offset, or `-1`; accepts Bytes, ByteBuffer, or String. An empty pattern matches at `0`. |
| `toString`, `toString()` | Diagnostic String such as `<Bytes length=3>`. Use `toUtf8()` to decode text. |
| `iterate(state)` | Iteration protocol: pass `null` to start, then the previous state; returns the next index or `false`. |
| `iteratorValue(state)` | Numeric byte at the iteration index. |

`for...of` iterates numeric bytes. Byte values, lengths, and indices must be
finite integers; negative indices and invalid ranges throw rather than wrap or
clamp. Binary allocations are limited to 256 MiB. Slices and clones retain their
backing storage; changing a `toMutable()` copy does not affect the source.

```napalm
const data = Bytes.fromUtf8("Hello");
System.print(data.length); // 5
System.print(data[0]); // 72
System.print(data.slice(1, 4).toUtf8()); // ell

const edited = data.toMutable();
edited[0] = 104;
System.print(edited.toUtf8()); // hello
System.print(data.toUtf8()); // Hello
```

## 40. ByteBuffer

`ByteBuffer` is a mutable, growable binary buffer. `new ByteBuffer()` creates
an empty buffer; `new ByteBuffer(capacity)` reserves initial storage but still
starts with length zero. It is available globally, without an import.

### 40.1 Properties and Methods

| Member | Result and behavior |
| --- | --- |
| `length` | Current number of bytes. |
| `capacity` | Current storage capacity in bytes. |
| `isEmpty` | Whether length is zero. |
| `buffer[index]` | Reads a numeric byte at a zero-based index. |
| `buffer[index] = value` | Replaces an existing byte with an integer in `0..255`; does not grow the buffer. |
| `append(data)` | Appends Bytes, ByteBuffer, or a String's UTF-8 bytes. Self-append is supported. Returns the buffer. |
| `reserve(additional)` | Ensures room for at least `additional` bytes beyond the current length, without changing length. Returns the buffer. |
| `resize(length)` | Truncates or grows to the requested length, filling new bytes with zero. Returns the buffer. |
| `writeUInt8(value)` | Appends one unsigned byte. Returns the buffer. |
| `writeUInt16BE(value)`, `writeUInt16LE(value)` | Appends an unsigned 16-bit integer in big-/little-endian order. Returns the buffer. |
| `writeUInt32BE(value)`, `writeUInt32LE(value)` | Appends an unsigned 32-bit integer in big-/little-endian order. Returns the buffer. |
| `take(count)` | Removes the first `count` bytes and returns them as immutable Bytes. Returns `null` without consuming anything if insufficient bytes are available. |
| `peek(count)` | Copies the first `count` bytes into Bytes without consuming them; returns `null` if insufficient bytes are available. |
| `discard(count)` | Removes the first `count` bytes. Throws if insufficient bytes are available. Returns the buffer. |
| `freeze()` | Transfers all contents into immutable Bytes and leaves the buffer empty and reusable. |
| `clear()` | Removes all contents. Returns the buffer. |
| `fill(value)` | Replaces all existing bytes with an integer in `0..255`, preserving length. Returns the buffer. |
| `indexOf(pattern)` | First matching byte offset, or `-1`; accepts Bytes, ByteBuffer, or String. An empty pattern matches at `0`. |
| `toUtf8()` | Decodes current contents as a String without consuming them; invalid UTF-8 throws. |
| `toString`, `toString()` | Diagnostic String such as `<ByteBuffer length=3>`. |
| `iterate(state)` | Iteration protocol: pass `null` to start, then the previous state; returns the next index or `false`. |
| `iteratorValue(state)` | Numeric byte at the iteration index. |

`for...of` iterates numeric bytes. The same integer, bounds, and 256 MiB size
limits as Bytes apply. Integer writes require values in `0..255`, `0..65535`,
or `0..4294967295`, respectively; they append rather than overwrite. Bytes
returned by `take`, `peek`, or `freeze` remain unchanged when the buffer is
subsequently reused.

```napalm
const frame = new ByteBuffer(16);
frame.writeUInt16BE(4660).append("OK");
System.print(frame.length); // 4
System.print(frame.peek(2)[0]); // 18
frame.discard(2);
System.print(frame.take(2).toUtf8()); // OK
System.print(frame.take(1)); // null
frame.append("reused");
System.print(frame.freeze().toUtf8()); // reused
System.print(frame.isEmpty); // true
```

See [scripts/bytes.napalm](scripts/bytes.napalm) for Bytes and ByteBuffer examples.

## 41. Fs

`Fs` is a global utility for synchronous, blocking filesystem operations.
Paths must be UTF-8 Strings and follow the host operating system's filesystem
rules. Relative paths use the process working directory. Methods return their
results directly, not Promises; invalid arguments and I/O failures throw.

### 41.1 Methods

| Method | Result and behavior |
| --- | --- |
| `Fs.readText(path)` | Reads the whole file into a String; invalid UTF-8 throws. |
| `Fs.readBytes(path)` | Reads the whole file into immutable Bytes. |
| `Fs.writeText(path, text)` | Creates or overwrites a file with a String's UTF-8 contents. Returns `null`. |
| `Fs.writeBytes(path, data)` | Creates or overwrites a file from Bytes or ByteBuffer without consuming the input. Returns `null`. |
| `Fs.exists(path)` | Bool indicating whether the path exists, following symlinks. Returns `false` for missing paths or dangling links; other lookup errors can throw. |
| `Fs.stat(path)` | FileInfo instance containing a metadata snapshot for the path itself, without following a final symlink. |
| `Fs.open(path, mode)` | Opens a regular file and returns a File instance. Directories, symlinks (including dangling links), and other non-regular files are rejected. See the modes below. |
| `Fs.readDir(path)` | List of entry Maps sorted by name; does not recurse or follow entry symlinks for type flags. |
| `Fs.realPath(path)` | Canonical absolute String path, resolving symlinks; the target must exist. |
| `Fs.mkdir(path)` | Creates one directory; its parent must exist. An existing target is an error. Returns `null`. |
| `Fs.mkdirAll(path)` | Creates missing parent directories too; succeeds for an existing directory. Returns `null`. |
| `Fs.removeFile(path)` | Removes a file or symlink. Returns `null`. |
| `Fs.rename(from, to)` | Renames or moves a file or directory using host filesystem rules. Returns `null`. |

Reads and writes are limited to 256 MiB per operation, and `readDir` to 100,000
entries. `writeText` and `writeBytes` truncate existing files and do not create
parent directories. Use `open` for incremental I/O or append mode. There is no
directory-removal API here.
`realPath` may return a Windows extended-length path such as `\\?\C:\...`.

### 41.2 FileInfo

`Fs.stat(path)` returns a native `FileInfo`, not a Map. Its properties are
read-only and describe the instant at which `stat` was called. Call `stat`
again to refresh them. `FileInfo` is a global type usable in annotations and
`instanceof`, but has no public constructor.

| Property | Meaning |
| --- | --- |
| `path` | The supplied filesystem path as a String. |
| `name` | Final filename component as a String, or `""` for a root. |
| `size` | Filesystem metadata size in bytes, represented as Num. |
| `isFile`, `isDirectory`, `isSymlink` | Bool type flags for the path itself. |
| `modifiedAt`, `createdAt` | Date values for modification and creation, or `null` if unavailable on the filesystem. Each access returns an independent Date. |

Use `info.size` and `info.modifiedAt`, replacing the previous Map fields
`info["size"]` and `info["modifiedMs"]`. The previous `readonly` and `accessedMs`
fields are no longer exposed.

### 41.3 Directory Entries

Each `readDir` entry has `name`, `path`, `isFile`, `isDirectory`, and `isSymlink`.
`path` includes the supplied directory path and is not necessarily absolute.
Directory names and returned paths must be representable as UTF-8.

This example reads an existing repository file and performs no writes:

```napalm
const path = "scripts/json.napalm"; // Run from the repository root.
if (Fs.exists(path)) {
    const text = Fs.readText(path);
    const metadata: FileInfo = Fs.stat(path);
    System.print(metadata.isFile); // true
    System.print(metadata.size === Bytes.fromUtf8(text).length); // true
    System.print(Fs.realPath(path)); // Host-specific absolute path.
}
```

### 41.4 Opening Files

`Fs.open(path, mode)` requires an explicit mode String. Only regular files can
be opened; the final path component cannot be a symlink. Existing parent
directories are required. The opened resource is checked before truncating it.

| Mode | Read | Write | Missing file | Existing contents |
| --- | --- | --- | --- | --- |
| `"r"` | Yes | No | Error | Preserved. |
| `"r+"` | Yes | Yes | Error | Preserved. |
| `"w"` | No | Yes | Created | Truncated. |
| `"w+"` | Yes | Yes | Created | Truncated. |
| `"a"` | No | Yes | Created | Preserved; writes always append. |
| `"a+"` | Yes | Yes | Created | Preserved; writes always append. |

The cursor initially starts at zero, including for reads in `a+` mode. Append
writes go to the current end of the file regardless of an earlier `seek`.
Reads and writes otherwise advance the shared cursor by the bytes transferred.

### 41.5 File Properties and Methods

`File` is a native global type with no public constructor; obtain an instance
through `Fs.open`. It supports type annotations and `instanceof`.

| Read-only property | Meaning |
| --- | --- |
| `path` | The path supplied to `Fs.open`. Remains readable after closing. |
| `name` | Final filename component. Remains readable after closing. |
| `modifiedAt`, `createdAt` | Fresh Date values from the open handle's current metadata, or `null` if unavailable. Access after closing throws. |

| Method | Result and behavior |
| --- | --- |
| `read(count)` | Returns Bytes containing up to `count` bytes from the current cursor. EOF and a zero count return empty Bytes. Requires a readable mode. |
| `write(bytes)` | Writes all supplied Bytes or ByteBuffer contents at the cursor (or end in append mode), without consuming the input. Returns the number of bytes written. Requires a writable mode; Strings are not accepted. |
| `seek(offset)` | Sets the cursor to the absolute byte offset from the beginning and returns that offset. Seeking beyond EOF is allowed and does not itself extend the file. |
| `flush()` | Synchronizes file data and metadata to storage. Returns `null`; filesystem errors throw. |
| `close()` | Releases the OS file handle and returns `null`. Repeated calls are harmless. |

Counts must be finite nonnegative integers at most 256 MiB; writes have the
same per-call size limit. Offsets must be finite nonnegative integers no greater
than `9007199254740991`. `read`, `write`, `seek`, and `flush` throw after closing.
I/O failures throw and can occur after partial transfer. Close explicitly with
`finally`; unreachable File objects also release their handles during heap
cleanup, whose timing is not deterministic. `close` does not replace `flush`
when storage synchronization is required. File and FileInfo cannot cross a
`launch` boundary.

```napalm
const file: File = Fs.open("scripts/json.napalm", "r"); // Run from repository root.
try {
    System.print(file.name); // json.napalm
    const prefix: Bytes = file.read(4);
    System.print(prefix.toUtf8()); // // E
    file.seek(0);
    System.print(file.read(4).toUtf8()); // // E
} finally {
    file.close();
}
```

See [scripts/fs.napalm](scripts/fs.napalm) for FileInfo, incremental File I/O,
file creation, directory listing, renaming, and removal examples.

## 42. Path

`Path` is a global utility for lexical path operations. It normalizes path
text without checking whether a file exists or resolving symlinks; use
`Fs.realPath` when filesystem canonicalization is needed. Inputs must be
UTF-8 Strings, at most 1 MiB each, without NUL characters.

| Method | Result and behavior |
| --- | --- |
| `Path.join(...parts)` | Joins and normalizes 0–16 String parts, skipping empty parts. With no nonempty parts, returns `"."`. Leading separators on later parts do not reset the accumulated path. |
| `Path.resolve(...parts)` | Resolves 0–16 String parts against the current working directory to an absolute path. A later absolute part replaces the preceding base. With no parts, returns the normalized working directory. |
| `Path.normalize(path)` | Normalizes separators, `.` and `..`; absolute paths cannot traverse above their root. An empty relative path becomes `"."`. |
| `Path.dirname(path)` | Parent of the normalized path; a root remains its own parent. |
| `Path.basename(path)` | Final component of the normalized path; an absolute root has an empty basename. |
| `Path.extension(path)` | Final extension including its dot, or `""`. A leading dot alone does not count: `.gitignore` has no extension. |
| `Path.isAbsolute(path)` | Bool indicating whether the parsed path is rooted. `C:work` is drive-relative; `C:\work` is absolute. |
| `Path.relative(from, to)` | Resolves both paths, then returns the path from `from` to `to`; identical paths produce `"."`. Different roots produce the absolute destination. |

POSIX and Windows drive/UNC syntax are recognized from the input, including
Windows paths on non-Windows hosts. Windows-style results use backslashes;
POSIX results use forward slashes. A leading `//server/share` is treated as UNC
on Windows. Windows component comparisons in `relative` are ASCII
case-insensitive.

`isAbsolute` also treats a Windows root-relative path such as `\work` as rooted,
but resolving it requires an active drive. Resolving `C:work` requires a base
on that drive. Incomplete UNC roots and Windows device paths throw; ordinary
extended-length drive and UNC prefixes are normalized.

```napalm
System.print(Path.join("src", "lib", "..", "main.napalm")); // src/main.napalm
System.print(Path.normalize("/a/../../b")); // /b
System.print(Path.extension("/src/main.napalm")); // .napalm
System.print(Path.relative("/a/b", "/a/c/d")); // ../c/d
System.print(Path.normalize("C:\\a\\..\\b")); // C:\b
```

## 43. Uri

`Uri` is a global utility for parsing URI text and converting file URIs.
String inputs must be valid UTF-8 and at most 1 MiB. These operations do not
access the filesystem.

### 43.1 Methods

| Method | Result and behavior |
| --- | --- |
| `Uri.parse(text)` | Parses a URI with a scheme and returns a Map of its components. Percent-encoded components stay encoded. |
| `Uri.fromFilePath(path)` | Converts an absolute POSIX, Windows drive, or UNC path to a normalized `file:` URI String, percent-encoding spaces, reserved characters, and UTF-8 bytes as needed. Relative paths and Windows rooted paths without a drive throw. |
| `Uri.toFilePath(uri)` | Converts a `file:` URI String, or a Map returned by `parse`, to a normalized filesystem path String. For a Map, reparses its `text` field; edits to other fields do not affect the result. |

### 43.2 Parsed Fields and Validation

| Field | Meaning |
| --- | --- |
| `text` | Original URI String. |
| `scheme` | Scheme before `:`, retaining its original spelling. |
| `authority` | Text after `//` and before the path, or `""`. |
| `path` | Raw path with percent encoding preserved. |
| `query` | Text after `?`, excluding the fragment, or `""`. |
| `fragment` | Text after `#`, or `""`. |
| `hasAuthority`, `hasQuery`, `hasFragment` | Bool flags distinguishing an absent component from a present but empty one. |

`parse` rejects missing or invalid schemes, malformed percent escapes,
unescaped spaces, and control characters. It does not decode query parameters
or split the authority into hostname, port, or credentials.

`toFilePath` requires the `file` scheme, an absolute URI path, and no query or
fragment (even an empty `?` or `#` is rejected). Empty authority and `localhost`
denote local paths; other supported authorities become UNC server names and
require a share. Encoded slashes, backslashes, NUL, and decoded invalid UTF-8
are rejected. Windows drive/UNC URIs produce Windows paths even on other hosts.

```napalm
const parsed = Uri.parse("https://example.com/a%20b?x=1#part");
System.print(parsed["authority"]); // example.com
System.print(parsed["path"]); // /a%20b
System.print(parsed["query"]); // x=1

const uri = Uri.fromFilePath("/tmp/hello world.txt");
System.print(uri); // file:///tmp/hello%20world.txt
System.print(Uri.toFilePath(Uri.parse(uri))); // /tmp/hello world.txt
```

See [scripts/path.napalm](scripts/path.napalm) for Path and Uri examples.

// Class contracts: `implements` field synthesis and required initialization,
// data-only interfaces/types, and the `instanceof`/`implements`/`inherits`
// operators.

/// Parse and lower `source`, returning every diagnostic message.
fn contract_diagnostics(source: &str) -> Vec<String> {
    let parsed = crate::napalm_front_main::parse("contracts.napalm", source);
    let mut messages: Vec<String> = parsed.diagnostics.iter().map(|d| d.message.clone()).collect();
    if !parsed.has_errors() {
        let mut hir = Program::new();
        let _ = napalm_builtin::install_hir_builtins(&mut hir);
        let output = lower_module(&mut hir, "contracts.napalm", source, "contracts", &parsed.module);
        messages.extend(output.diagnostics.iter().map(|d| d.message.clone()));
    }
    messages
}

#[test]
fn napalm_contracts_implements_synthesizes_fields_with_null_optional_defaults() {
    run_napalm(r#"
        interface Named { name: string; nickname?: string; }
        interface Aged extends Named { age: number; }
        type Point = { x: number; y: number; label: string | null };
        type Box<T> = { value: T };
        class Animal implements Named {
            constructor(name: string) { this.name = name; }
        }
        class Dog extends Animal implements Aged {
            constructor(name: string, age: number) { super(name); this.age = age; }
        }
        class P implements Point, Box<number> {
            constructor() { this.x = 1; this.y = 2; this.value = 3; }
        }
        const d = new Dog("Rex", 3);
        const p = new P();
        const result = [d.name, d.age, d.nickname, d.keys(), d.size, p.label, p.entries()];
    "#, "[Rex, 3, null, [name, nickname, age], 3, null, [[x, 1], [y, 2], [label, null], [value, 3]]]");
}

#[test]
fn napalm_contracts_fields_reuse_declared_and_inherited_storage() {
    run_napalm(r#"
        interface Named { name: string; }
        class Explicit implements Named {
            name = "explicit";
            count: number;
            constructor() { this.count = 1; }
        }
        class Param implements Named { constructor(public name: string) {} }
        class Base implements Named { constructor() { this.name = "base"; } }
        class Renamed extends Base implements Named {
            constructor() { super(); this.name = "renamed"; }
        }
        abstract class Deferred implements Named {}
        class Concrete extends Deferred { constructor() { super(); this.name = "concrete"; } }
        class Plain { constructor() { this.v = 1; } }
        class PlainChild extends Plain { constructor() { super(); this.v = 2; } }
        const result = [new Explicit().keys(), new Param("p").name, new Renamed().entries(),
            new Concrete().keys(), new Concrete().name, new PlainChild().entries()];
    "#, "[[name, count], p, [[name, renamed]], [name], concrete, [[v, 2]]]");
}

#[test]
fn napalm_contracts_synthesized_fields_keep_their_declared_type() {
    // Writes go through the synthesized setter, which checks the field's
    // declared contract type at runtime.
    let (program, entry, manifest) = compile_typescript_module(r#"
        interface Named { name: string; }
        class A implements Named { constructor(value) { this.name = value; } }
        const result = new A(5);
    "#);
    let mut vm = source_vm(&program, &manifest);
    let error = vm.run_module(entry).unwrap_err();
    assert!(format!("{error:?}").contains("String"), "{error:?}");
}

#[test]
fn napalm_contracts_instanceof_implements_and_inherits() {
    run_napalm(r#"
        interface Named { name: string; }
        interface Aged extends Named { age: number; }
        type Point = { x: number; y: number };
        class Animal implements Named { constructor() { this.name = "a"; } }
        class Dog extends Animal implements Aged { constructor() { super(); this.age = 1; } }
        class Puppy extends Dog {}
        const a = new Animal();
        const d = new Puppy();
        const plain = { name: "plain" };
        const instance = [d instanceof Puppy, d instanceof Dog, d instanceof Animal,
            d instanceof Named, d instanceof Aged, a instanceof Aged, a instanceof Point];
        const contracts = [d implements Named, d implements Aged, a implements Aged,
            Dog implements Aged, Dog instanceof Aged, 3 implements Named, null instanceof Named];
        const ancestry = [d inherits Dog, d inherits Animal, d inherits Puppy,
            Puppy inherits Animal, Animal inherits Animal, plain inherits Animal, d inherits 3];
        const structural = [plain implements Named, plain instanceof Named, plain implements Aged,
            { x: 1, y: 2 } instanceof Point, plain instanceof Animal];
        function nested(value: any) {
            const check = () => [value instanceof Named, value implements Aged, value inherits Animal];
            return check();
        }
        let hits = 0;
        for (let i = 0; i < 50; i++) {
            const value = i % 2 == 0 ? d : plain;
            if (value instanceof Aged) { hits++; }
            if (value implements Named) { hits++; }
            if (value inherits Dog) { hits++; }
        }
        const result = [instance, contracts, ancestry, structural, nested(d), nested(plain), hits];
    "#, "[[true, true, true, true, true, false, false], \
[true, true, false, true, false, false, false], \
[true, true, false, true, false, false, false], \
[true, true, false, true, false], [true, true, true], [true, false, false], 100]");
}

#[test]
fn napalm_contracts_reject_methods_in_interfaces_and_types() {
    for source in [
        "interface Shape { name: string; area(): number; }",
        "interface Shape { map<T>(value: T): T }",
        "type Shape = { name: string; area(): number };",
        "const s: { area(): number } = { area: 1 };",
    ] {
        let messages = contract_diagnostics(source);
        assert_eq!(
            messages,
            vec!["interfaces and types cannot declare methods; declare a class to provide methods"],
            "{source}"
        );
    }
    // Function-typed fields are data, not methods.
    assert!(contract_diagnostics("interface H { onDone: () => void; map?: (x: number) => number }").is_empty());
}

#[test]
fn napalm_contracts_require_constructor_initialization() {
    let missing = contract_diagnostics(r#"
        interface Named { name: string; age: number; nickname?: string; note: string | null; }
        class A implements Named { constructor() { this.name = "a"; } }
    "#);
    assert_eq!(missing, vec!["class `A` must initialize required field `age` of `Named` in its constructor"]);

    let deferred = contract_diagnostics(r#"
        interface Named { name: string; }
        abstract class Base implements Named {}
        class Child extends Base {}
    "#);
    assert_eq!(deferred, vec!["class `Child` must initialize required field `name` of `Named` in its constructor"]);

    assert!(contract_diagnostics(r#"
        interface Named { name: string; }
        class Initialized implements Named { name = "x"; }
        class Definite implements Named { name!: string; }
        class Inherited extends Initialized implements Named {}
    "#).is_empty());
}

#[test]
fn napalm_contracts_report_invalid_implements_clauses() {
    let cases = [
        ("class A implements Missing {}",
         "unknown interface or type `Missing`; a class can only implement interfaces and object types declared in the same module"),
        ("type Id = string | number;\nclass A implements Id {}",
         "`Id` is not an object type; a class can only implement interfaces and object types"),
        ("interface X { v: string }\ninterface Y { v: number }\nclass A implements X, Y { constructor() { this.v = 's'; } }",
         "field `v` has conflicting types in `X` and `Y`"),
        ("interface X { v: string }\nclass A implements X { v() { return 1; } }",
         "method `v` of class `A` conflicts with field `v` of `X`"),
        ("class A {}\nconst result = new A() implements A;",
         "`implements` expects an interface or object type name on its right-hand side"),
    ];
    for (source, expected) in cases {
        assert_eq!(contract_diagnostics(source), vec![expected.to_owned()], "{source}");
    }
}

#[test]
fn napalm_contracts_instances_satisfy_implemented_contracts_at_typed_boundaries() {
    run_napalm(r#"
        interface Named { name: string; }
        interface Aged extends Named { age: number; }
        type Point = { x: number; y?: number };
        class Dog implements Aged { constructor() { this.name = "rex"; this.age = 3; } }
        class Puppy extends Dog {}
        class P implements Point { constructor(x: number) { this.x = x; } }
        function greet(n: Named): string { return n.name; }
        function shift(p: Point) { p.x = p.x + 10; return p.x; }
        const typed: Named = new Puppy();
        const mixed: Named[] = [new Dog(), { name: "plain" }];
        const p = new P(1);
        const result = [typed.name, greet(new Puppy()), mixed[0].name, mixed[1].name, shift(p), p.x, p.y];
    "#, "[rex, rex, rex, plain, 11, 11, null]");

    // Contracts are nominal for classes: a class that does not declare
    // `implements` never satisfies the contract, even with matching fields.
    let (program, entry, manifest) = compile_typescript_module(r#"
        interface Named { name: string; }
        class Cat { constructor() { this.name = "tom"; } }
        function greet(n: Named): string { return n.name; }
        const result = greet(new Cat());
    "#);
    let mut vm = source_vm(&program, &manifest);
    let error = vm.run_module(entry).unwrap_err();
    assert!(error.message.starts_with(interfaces::TYPE_ERROR), "{error:?}");
}

#[test]
fn napalm_typeof_reports_runtime_kinds_contracts_and_generic_arguments() {
    run_napalm(r#"
        interface Named { name: string; }
        type Point = { x: number; y: number };
        class Animal implements Named { constructor() { this.name = "a"; } }
        class Dog extends Animal {}
        class Holder<T> { item: T; constructor(item: T) { this.item = item; } }
        const nums: number[] = [1, 2];
        const typedMap: Map<string, number> = new Map();
        const named: Named = { name: "n" };
        const point: Point = { x: 1, y: 2 };
        const emitter = new Emitter<number>();
        const timer = Timer.delay(60000, () => 1);
        const timerKind = typeof timer;
        timer.cancel();
        const result = [
            typeof 1, typeof true, typeof null, typeof "s", typeof nums, typeof [],
            typeof typedMap, typeof new Map(), typeof {}, typeof named, typeof point,
            typeof new Set(), typeof Named, typeof Point, typeof Animal, typeof new Dog(),
            typeof new Holder<number>(1), typeof new Holder(1), typeof (() => 1),
            typeof Promise.resolve(1), timerKind, typeof new Date(), typeof emitter, typeof emitter.event,
        ];
    "#, "[Number, Boolean, Null, String, Array<Number>, Array, Map<String, Number>, Map, Object, \
         Object implementation of Named, Object implementation of Point, Set, Interface, Type, Class, \
         Instance of Dog, Instance of Holder<Number>, Instance of Holder, Function, Promise, Timer, Date, \
         Emitter<Number>, Event<Number>]");
}

#[test]
fn napalm_generic_interfaces_and_types_are_instantiated_per_use() {
    run_napalm(r#"
        interface Box<T> { value: T; }
        type Pair<A, B> = { first: A; second: B };
        interface Bag<T> { items: T[]; }
        const box: Box<number> = { value: 1 };
        const loose: Box = { value: "anything" };
        const pair: Pair<number, string> = { first: 1, second: "x" };
        const bag: Bag<string> = { items: ["a"] };
        const result = [box.value, loose.value, pair.second, bag.items[0], typeof box, typeof pair];
    "#, "[1, anything, x, a, Object implementation of Box<number>, Object implementation of Pair<number,string>]");

    for source in [
        "interface Box<T> { value: T; } const v: any = 'x'; const b: Box<number> = { value: v };",
        "type Pair<A, B> = { first: A; second: B }; const v: any = 'x'; const p: Pair<number, string> = { first: v, second: 'y' };",
        "interface Bag<T> { items: T[]; } const b: Bag<number> = { items: [1] }; const v: any = 'x'; b.items.add(v);",
    ] {
        let (program, entry, manifest) = compile_typescript_module(source);
        let mut vm = source_vm(&program, &manifest);
        let error = vm.run_module(entry).unwrap_err();
        assert!(error.message.starts_with(interfaces::TYPE_ERROR), "{source}: {error:?}");
    }
    assert!(contract_diagnostics("interface Box<T> { value: T; } const b: Box<number, string> = { value: 1 };")
        .iter().any(|message| message.contains("expects 1 type argument")));
}

#[test]
fn napalm_generic_classes_check_fields_against_instance_arguments() {
    run_napalm(r#"
        class Holder<T> {
            item: T;
            constructor(item: T) { this.item = item; }
            set(value: T) { this.item = value; }
        }
        class Pair<A, B> { a: A; b: B; constructor(a: A, b: B) { this.a = a; this.b = b; } }
        function hot() {
            const acc = new Holder<number>(0);
            for (let i = 0; i < 1000; i++) { acc.item = acc.item + i; }
            return acc.item;
        }
        const typed = new Holder<number>(3);
        typed.set(4);
        const free = new Holder(3);
        free.item = "now a string";
        const annotated: Holder<string> = new Holder("s");
        const pair = new Pair<string, number[]>("k", [1]);
        const result = [typed.item, free.item, annotated.item, pair.b[0], hot()];
    "#, "[4, now a string, s, 1, 499500]");

    for (source, message) in [
        ("class H<T> { item: T; constructor(i: T) { this.item = i; } } const v: any = 'x'; const h = new H<number>(v);", "Structural type error"),
        ("class H<T> { item: T; constructor(i: T) { this.item = i; } } const h = new H<number>(1); const v: any = 'x'; h.item = v;", "Generic field type error"),
        ("class H<T> { item: T; constructor(i: T) { this.item = i; } set(v: T) { this.item = v; } } const h = new H<number>(1); const v: any = 'x'; h.set(v);", "Generic field type error"),
        ("class H<T> { item: T; constructor(i: T) { this.item = i; } } const h: H<string> = new H(1);", "Structural type error"),
        ("class P<A, B> { b: B; constructor(b: B) { this.b = b; } } const p = new P<string, number[]>([1]); const v: any = 'x'; p.b.add(v);", "Structural type error"),
        ("class H<T> { item: T; constructor(i: T) { this.item = i; } } function hot() { const h = new H<number>(0); for (let i = 0; i < 1000; i++) { h.item = h.item + i; } const v: any = 'x'; h.item = v; } hot();", "Generic field type error"),
    ] {
        let (program, entry, manifest) = compile_typescript_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            let error = vm.run_module(entry).unwrap_err();
            assert!(error.message.contains(message), "jit={jit} {source}: {error:?}");
        }
    }
}

#[test]
fn napalm_awaited_promise_values_are_checked_against_their_declared_type() {
    run_napalm("const p: Promise<number> = Promise.resolve(1); const result = await p;", "1");
    let (program, entry, manifest) = compile_typescript_module(
        "const v: any = 'x'; const p: Promise<number> = Promise.resolve(v); const result = await p;",
    );
    let mut vm = source_vm(&program, &manifest);
    assert!(vm.run_module(entry).is_err());
}

// Production frontend -> HIR -> optimizer -> bytecode -> interpreter/JIT.
fn run_napalm(source: &str, expected: &str) {
    let parsed = crate::napalm_front_main::parse("test.nap", source);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let output = lower_module(&mut hir, "test.nap", source, "test", &parsed.module);
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
    let entry = output.lowered.unwrap().entry;
    let program = compile_program(&mut hir).unwrap();
    for enabled in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(enabled.then(|| TracingJitOptions {
            hot_loop_threshold: 2,
            ..Default::default()
        }));
        let result = vm.run_module(entry).unwrap_or_else(|e| {
            panic!(
                "{e:?}\nMethods: {:?}",
                program
                    .metadata
                    .methods
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (i, program.metadata.string(s.name), s.kind))
                    .collect::<Vec<_>>()
            )
        });
        let value = *vm.modules.last().unwrap().variables.last().unwrap();
        let context = HostContext { heap: &mut vm.heap };
        assert_eq!(
            context.format_value(value),
            expected,
            "jit={enabled}, {result:?}"
        );
    }
}

#[test]
fn napalm_functions_types_closures_and_templates() {
    run_napalm(
        r#"
        type Numbers = Array<number>;
        interface Pair { left: number; right: number; }
        function twice(value: number): number { return value * 2; }
        const add: (a: number, b: number) => number = (a: number, b: number): number => a + b;
        let captured = 1;
        const next = () => ++captured;
        const values: Numbers = [next(), next(), add(2, 3), twice(4)];
        let result = `${values}: ${`nested ${twice(5)}`}: 50%`;
    "#,
        "[2, 3, 5, 8]: nested 10: 50%",
    );
}

#[test]
fn napalm_classes_constructor_inheritance_fields_accessors() {
    run_napalm(
        r#"
        const offset = 2;
        class Parent {
            value: number = 1;
            constructor(value: number) { this.value = value; }
            add(n: number): number { return this.value + n + offset; }
        }
        class Child extends Parent {
            extra = 3;
            static total: number = 10;
            constructor(value: number) { super(value); Child.total++; }
            get doubled(): number { return this.value * 2; }
            set doubled(n: number) { this.value = n / 2; }
            add(n: number) { return super.add(n) + this.extra; }
            callback() { return () => this.value; }
        }
        const item = new Child(4);
        item.doubled = 12;
        let result = [item.add(1), item.doubled, Child.total, item.callback()(), item instanceof Parent];
    "#,
        "[12, 12, 11, 6, true]",
    );
}

#[test]
fn napalm_for_continue_break_updates_and_capture() {
    run_napalm(
        r#"
        let sum = 0;
        for (let i = 0; i < 20; i++) {
            if (i === 2) continue;
            if (i === 10) break;
            sum += i;
        }
        for (const value of [1, 2, 3]) sum += value;
        let i = 1;
        let result = [sum, i++, ++i, i];
    "#,
        "[49, 1, 3, 3]",
    );
}

#[test]
fn napalm_assignment_evaluates_location_once_before_rhs() {
    run_napalm(
        r#"
        let order = '';
        const items = [10];
        function container() { order += 'a'; return items; }
        function index() { order += 'b'; return 0; }
        function value() { order += 'c'; return 2; }
        container()[index()] += value();
        const old = container()[index()]++;
        container()[index()] = value();
        const result = [order, old, items[0]];
    "#,
        "[abcababc, 12, 2]",
    );
}

#[test]
fn napalm_objects_and_hot_loop() {
    run_napalm(
        r#"
        const name = 'Napalm';
        const object = {name, score: 0, double: (x: number) => x * 2};
        for (let i = 0; i < 100; i++) object.score += object.double(i);
        let result = [object.name, object.score, object['score']];
    "#,
        "[Napalm, 9900, 9900]",
    );
}

#[test]
fn napalm_async_exceptions_and_cleanup() {
    run_napalm(
        r#"
        let log = '';
        async function compute(value: number): Promise<number> {
            defer { log += 'd'; }
            try { await Timer.delay(0); throw 'x'; }
            catch (error) { log += error; }
            finally { log += 'f'; }
            return value * 2;
        }
        const value = await compute(21);
        const task = async (n: number) => await Promise.resolve(n + 1);
        let result = [value, await task(1), log];
    "#,
        "[42, 2, xfd]",
    );
}

#[test]
fn napalm_const_shadowing_is_lexical() {
    run_napalm(
        "const x = 1; { let x = 2; x++; } const list = [1]; list[0]++; let result = [x, list];",
        "[1, [2]]",
    );
    for source in [
        "const x;",
        "const x = 1; x = 2;",
        "const x = 1; const f = () => x++;",
        "let x = 1; let x = 2;",
        "export { missing };",
        "let x: = 1;",
        "class Old { construct new() {} }",
        "for (x in [1]) {}",
        "let x = {|n| n};",
    ] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid source: {source}");
    }
}

#[test]
fn napalm_newlines_and_block_return_semantics() {
    run_napalm(
        r#"
        function empty() { 42; }
        function line() { return
            42;
        }
        const f = (x: number) => (
            x +
            2
        );
        const result = [empty(), line(), f(3), '50%', "a\u{1f600}"];
    "#,
        "[null, null, 5, 50%, a😀]",
    );
}

#[test]
fn napalm_nested_recursion_and_ranges() {
    run_napalm(
        r#"
        function outer() {
            function factorial(n: number): number { return n < 2 ? 1 : n * factorial(n - 1); }
            return factorial(6);
        }
        let sum = 0;
        for (const i of new Range(1, 5, true)) sum += i;
        const result = [outer(), sum, [0,1,2,3][new Range(1,3)]];
    "#,
        "[720, 15, [1, 2]]",
    );
}

#[test]
fn napalm_field_initializers_are_per_instance_and_function_bodies_are_statements() {
    run_napalm(
        r#"
        class Bag { items = []; callback = (n: number) => n + 1; }
        const first = new Bag();
        const second = new Bag();
        first.items.add(1);
        const result = [first.items, second.items, (first.callback)(41)];
    "#,
        "[[1], [], 42]",
    );
}

#[test]
fn napalm_const_imports_and_type_syntax_rejections() {
    for source in [
        "import { value } from './dep'; value++;",
        "const value = 1; function change() { value += 1; }",
        "let value = 1; export { value, value };",
        "const f = (a: number, a: number) => a;",
        "class A { constructor() {} constructor() {} }",
        "let x = 1; x + 2 = 3;",
        "let x = `unterminated ${1`",
    ] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid source: {source}");
    }
    let source = "type Matrix = Array<Array<number>>; interface A { f: (n: number) => number; } const matrix: Matrix = [[1]];";
    assert!(!crate::napalm_front_main::parse("types.nap", source).has_errors());
}

#[test]
fn napalm_object_property_stores_survive_gc_and_shape_changes() {
    run_napalm(
        r#"
        const object = {};
        System.gc();
        object.items = [1, 2, 3];
        object.text = 'retained';
        for (let i = 0; i < 80; i++) object[`key${i}`] = [i];
        System.gc();
        for (let i = 0; i < 30; i++) {
            object.items = [i, i + 1];
            System.gc();
        }
        const result = [object.items, object.text, object['key79']];
    "#,
        "[[29, 30], retained, [79]]",
    );
}

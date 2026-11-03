// Static types as optimization facts: declared result contracts, class-hierarchy
// devirtualization and trivial-accessor inlining (StaticTypeSpecialization).

/// Lower `source` and return the diagnostics without asserting success.
fn static_type_diagnostics(source: &str) -> Vec<crate::napalm_front::Diagnostic> {
    let parsed = crate::napalm_front_main::parse("static_types.napalm", source);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let mut hir = Program::new();
    let _ = napalm_builtin::install_hir_builtins(&mut hir);
    lower_module(&mut hir, "static_types.napalm", source, "static_types", &parsed.module).diagnostics
}

/// Opcodes of the `ordinal`-th source-level function declaration.
fn static_type_closure_opcodes(program: &BytecodeProgram, ordinal: usize) -> Vec<Opcode> {
    let module = program.functions.last().unwrap().module;
    let function = program
        .functions
        .iter()
        .filter(|function| function.module == module && function.kind == FunctionKind::Closure)
        .nth(ordinal)
        .expect("source function");
    crate::napalm_bytecode::Decoder::new(&function.chunk.code).map(|inst| inst.unwrap().opcode).collect()
}

/// Run on the interpreter only and with every native tier enabled.
fn static_type_results(program: &BytecodeProgram, entry: FuncId, manifest: &BuiltinManifest) -> [f64; 2] {
    [false, true].map(|native| {
        let mut vm = source_vm(program, manifest);
        if !native {
            vm.set_jit_options(None);
        }
        vm.run_module(entry).unwrap();
        source_result(&vm)
    })
}

#[test]
fn napalm_static_return_contract_rejects_statically_wrong_returns() {
    let diagnostics = static_type_diagnostics(r#"
        function name(): number { return "seven"; }
    "#);
    assert!(
        diagnostics.iter().any(|diagnostic| diagnostic.code == "NAPALM-T007"),
        "{diagnostics:#?}"
    );
    // Generic, void and async annotations never constrain `return`.
    for source in [
        "function id<T>(x: T): T { return x; } const result = id(\"a\");",
        "function nothing(): void { return 1; }",
        "async function later(): Promise<number> { return 1; }",
    ] {
        let diagnostics = static_type_diagnostics(source);
        assert!(
            diagnostics.iter().all(|diagnostic| diagnostic.code != "NAPALM-T007"),
            "{source}: {diagnostics:#?}"
        );
    }
}

#[test]
fn napalm_static_return_contract_validates_dynamic_returns() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function unchecked(x) { return x; }
        function checked(x): number { return unchecked(x); }
        const ok = checked(4);
        const result = checked("not a number");
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native {
            vm.set_jit_options(None);
        }
        assert!(vm.run_module(entry).is_err(), "a broken result contract must be fatal");
    }
}

#[test]
fn napalm_constant_loop_headers_keep_their_loop_carried_parameters() {
    // `while (true)` folds its header branch into a Jump; edge threading must
    // not bypass a header whose block parameters the loop body still reads.
    let (program, entry, manifest) = compile_typescript_module(r#"
        function returns(n: number): number {
            let i = 0;
            while (true) {
                i = i + 1;
                if (i >= n) { return i; }
            }
        }
        function breaks(n) {
            let i = 0;
            while (true) {
                i = i + 1;
                if (i >= n) { break; }
            }
            return i;
        }
        const result = returns(4) * 100 + breaks(7);
    "#);
    assert_eq!(static_type_results(&program, entry, &manifest), [407.0; 2]);
}

#[test]
fn napalm_static_types_devirtualize_monomorphic_class_calls() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Rect {
            width: number;
            height: number;
            constructor(width: number, height: number) {
                this.width = width;
                this.height = height;
            }
            function area(): number { return this.width * this.height; }
        }
        function measure(shape: Rect): number { return shape.area(); }
        function run(): number {
            let total = 0;
            const rect = new Rect(2, 3);
            for (let i = 0; i < 2000; i++) total = total + measure(rect);
            return total;
        }
        const result = run();
    "#);
    let opcodes = static_type_closure_opcodes(&program, 0);
    assert!(opcodes.contains(&Opcode::CallDirect), "{opcodes:?}");
    assert!(!opcodes.contains(&Opcode::Invoke), "{opcodes:?}");
    assert_eq!(static_type_results(&program, entry, &manifest), [12000.0; 2]);
}

#[test]
fn napalm_static_types_keep_dynamic_dispatch_for_overrides() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Shape {
            function area(): number { return 1; }
        }
        class Square extends Shape {
            function area(): number { return 4; }
        }
        function measure(shape: Shape): number { return shape.area(); }
        function dynamic(shape) { return shape.area(); }
        function run(): number {
            let total = 0;
            const shapes = [new Shape(), new Square()];
            for (let i = 0; i < 2000; i++) total = total + measure(shapes[0]) + dynamic(shapes[i % 2]);
            return total;
        }
        const result = run();
    "#);
    // `Shape.area` is overridden, so a `Shape`-typed receiver keeps dispatch.
    // (Runtime class contracts are currently exact; the pass does not rely on that.)
    let opcodes = static_type_closure_opcodes(&program, 0);
    assert!(!opcodes.contains(&Opcode::CallDirect), "{opcodes:?}");
    assert_eq!(static_type_results(&program, entry, &manifest), [2000.0 + 5000.0; 2]);
}

#[test]
fn napalm_static_types_devirtualize_inherited_methods_and_super_calls() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Base {
            function scale(): number { return 3; }
            function value(): number { return 10; }
        }
        class Derived extends Base {
            function value(): number { return super.value() + 1; }
        }
        function both(item: Derived): number { return item.scale() * item.value(); }
        function run(): number {
            let total = 0;
            const item = new Derived();
            for (let i = 0; i < 1000; i++) total = total + both(item);
            return total;
        }
        const result = run();
    "#);
    // Both method calls resolve statically. The remaining Invoke is the numeric
    // `*`: source operators stay semantic Invokes and the JIT specializes them.
    let opcodes = static_type_closure_opcodes(&program, 0);
    assert_eq!(opcodes.iter().filter(|op| **op == Opcode::CallDirect).count(), 2, "{opcodes:?}");
    assert_eq!(opcodes.iter().filter(|op| **op == Opcode::Invoke).count(), 1, "{opcodes:?}");
    assert_eq!(static_type_results(&program, entry, &manifest), [33000.0; 2]);
}

#[test]
fn napalm_static_types_inline_trivial_accessors_with_inherited_fields() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Base {
            first: number;
            constructor() { this.first = 1; }
        }
        class Point extends Base {
            private _x: number;
            constructor(x: number) { super(); this._x = x; }
            get x(): number { return this._x; }
            set x(value: number) { this._x = value; }
            function absorb(other: Point): number {
                other.x = other.x + 1;
                return other.x;
            }
        }
        function bump(point: Point): number {
            point.x = point.x + 1;
            return point.x + point.first;
        }
        function run(): number {
            const point = new Point(0);
            const helper = new Point(0);
            let total = 0;
            for (let i = 0; i < 500; i++) total = total + bump(point) + helper.absorb(point);
            return total;
        }
        const result = run();
    "#);
    // A free function may not touch fields: accessors become direct calls and
    // only the two numeric `+` stay dynamic.
    let opcodes = static_type_closure_opcodes(&program, 0);
    assert_eq!(opcodes.iter().filter(|op| **op == Opcode::Invoke).count(), 2, "{opcodes:?}");
    assert!(!opcodes.contains(&Opcode::LoadField), "{opcodes:?}");
    // Inside the owning class the accessors collapse into field accesses.
    let absorb = program
        .functions
        .iter()
        .find(|function| program.metadata.string(function.name).is_some_and(|name| name.contains("absorb")))
        .expect("absorb method");
    let absorb_ops: Vec<Opcode> = crate::napalm_bytecode::Decoder::new(&absorb.chunk.code)
        .map(|inst| inst.unwrap().opcode)
        .collect();
    assert!(absorb_ops.contains(&Opcode::LoadField), "{absorb_ops:?}");
    assert!(absorb_ops.contains(&Opcode::StoreField), "{absorb_ops:?}");
    assert!(!absorb_ops.contains(&Opcode::CallDirect), "{absorb_ops:?}");
    // Each iteration adds 2 to x; bump sees 2i+1 (+1 for `first`), absorb 2i+2.
    let expected: f64 = (0..500).map(|i| (2 * i + 2) as f64 + (2 * i + 2) as f64).sum();
    assert_eq!(static_type_results(&program, entry, &manifest), [expected; 2]);
}

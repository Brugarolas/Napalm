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

/// Run once on the interpreter only and once with every native tier, returning
/// both results plus the tracing-JIT statistics of the native run.
fn static_type_native_run(
    program: &BytecodeProgram,
    entry: FuncId,
    manifest: &BuiltinManifest,
) -> ([f64; 2], crate::napalm_tracing_jit::TracingJitStats) {
    let mut baseline = source_vm(program, manifest);
    baseline.set_jit_options(None);
    baseline.run_module(entry).unwrap();
    let mut native = source_vm(program, manifest);
    native.run_module(entry).unwrap();
    let stats = native.jit_stats().cloned().expect("tracing JIT enabled");
    ([source_result(&baseline), source_result(&native)], stats)
}

#[test]
fn napalm_bytecode_carries_static_register_and_result_types() {
    let (program, _entry, _manifest) = compile_typescript_module(r#"
        function sum(xs: number[]): number {
            let total = 0;
            for (let i = 0; i < xs.length; i = i + 1) total = total + xs[i];
            return total;
        }
        const result = sum([1, 2, 3]);
    "#);
    let module = program.functions.last().unwrap().module;
    let sum = program
        .functions
        .iter()
        .find(|function| function.module == module && function.kind == FunctionKind::Closure)
        .expect("sum");
    // The declared result contract becomes the bytecode-level result type.
    assert_eq!(sum.result_type, Type::Num);
    // Live registers carry their guaranteed WIR types for trace anchors.
    let typed_list = Type::List(Some(Box::new(Type::Num)));
    assert!(
        sum.chunk.live_registers.iter().any(|entry| entry.types.iter().any(|(_, ty)| *ty == typed_list)),
        "{:#?}",
        sum.chunk.live_registers
    );
    // Internal representations are never published as semantic facts.
    assert!(sum.chunk.live_registers.iter().all(|entry| entry
        .types
        .iter()
        .all(|(_, ty)| !matches!(ty, Type::Int32 | Type::F64 | Type::RawPtr | Type::Top | Type::Cell(_)))));
}

#[test]
fn napalm_typed_list_length_and_subscript_trace_natively_with_baseline_edge_cases() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function sum(xs: number[]): number {
            let total = 0;
            for (let i = 0; i < xs.length; i = i + 1) total = total + xs[i];
            return total;
        }
        function edges(xs: number[]): number {
            // Negative indices wrap and out-of-range reads yield null: both
            // must leave the native fast path and replay the baseline Invoke.
            let acc = 0;
            for (let i = 0; i < 400; i = i + 1) acc = acc + (xs[i % 7 - 2] ?? 1000);
            return acc;
        }
        function run(): number {
            const big: number[] = [];
            for (let i = 0; i < 3000; i = i + 1) big.push(i);
            const small: number[] = [1, 2, 3];
            return sum(big) + edges(small);
        }
        const result = run();
    "#);
    let (results, stats) = static_type_native_run(&program, entry, &manifest);
    // 0+..+2999 = 4498500; one 7-step cycle of `edges` adds 2+3+1+2+3+1000+1000,
    // 57 full cycles plus i = 399 (index -2 -> 2).
    assert_eq!(results, [4498500.0 + 57.0 * 2011.0 + 2.0; 2]);
    assert!(stats.traces_compiled > 0, "{stats:#?}");
}

#[test]
fn inferred_list_literal_type_never_becomes_a_runtime_contract() {
    // DefineArrayWithGenerics must only replace an explicit List<T> binding.
    // The inferred type of `[1]` is not a contract: an `any` variable stays
    // untyped. (Violations of real contracts are uncatchable by design and are
    // covered by the interface tests.)
    let (program, entry, manifest) = compile_typescript_module(r#"
        const loose: any = [1];
        loose[0] = 'a';
        loose.push('b');
        const declared: number[] = [];
        declared.push(4);
        const result = loose.length * 10 + declared[0];
    "#);
    assert_eq!(static_type_results(&program, entry, &manifest), [24.0; 2]);
}

#[test]
fn napalm_structural_arguments_are_scalar_replaced_after_direct_call_inlining() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        interface Point { x: number; y: number; }
        function dist2(a: Point, b: Point): number {
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            return dx * dx + dy * dy;
        }
        function points(n: number): number {
            let a: Point = { x: 1, y: 2 };
            let total = 0;
            for (let i = 0; i < n; i = i + 1) {
                let b: Point = { x: i, y: i + 1 };
                total = total + dist2(a, b) % 13;
            }
            return total;
        }
        // dist2 stays an ordinary function for other callers.
        const direct = dist2({ x: 0, y: 0 }, { x: 3, y: 4 });
        const result = points(2000) + direct;
    "#);
    // `points` is the second source function. Its literals no longer escape
    // into a call, so neither the maps nor their property traffic remain.
    let opcodes = static_type_closure_opcodes(&program, 1);
    // Include the specialized encodings: an object factory or shape access
    // would mean the literal was NOT scalar-replaced.
    for opcode in [Opcode::AllocMap, Opcode::StoreProperty, Opcode::LoadProperty, Opcode::CallDirect,
        Opcode::NewObjectEmpty, Opcode::NewObjectShape, Opcode::NewObjectWithProperties,
        Opcode::NewInterfaceOrTypeObject, Opcode::GetFieldShape, Opcode::SetFieldShape,
        Opcode::MapSet, Opcode::ImplementInterfaceOrType] {
        assert!(!opcodes.contains(&opcode), "{opcode:?} in {opcodes:?}");
    }
    let expected: f64 = (0..2000)
        .map(|i| {
            let (dx, dy) = (1.0 - i as f64, 2.0 - (i as f64 + 1.0));
            (dx * dx + dy * dy) % 13.0
        })
        .sum::<f64>()
        + 25.0;
    assert_eq!(static_type_results(&program, entry, &manifest), [expected; 2]);
}

#[test]
fn napalm_structural_contracts_still_reject_bad_arguments_after_inlining() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        interface Point { x: number; y: number; }
        function norm(p: Point): number { return p.x * p.x + p.y * p.y; }
        function run(): number {
            let total = 0;
            for (let i = 0; i < 10; i = i + 1) {
                const bad = { x: i, y: "not a number" };
                total = total + norm(bad);
            }
            return total;
        }
        const result = run();
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native {
            vm.set_jit_options(None);
        }
        assert!(vm.run_module(entry).is_err(), "the inlined Point contract must still fail");
    }
}

#[test]
fn napalm_typed_list_subscript_results_stay_nullable() {
    // `List<T>[i]` is `T | null`: an out-of-range read yields null, so neither
    // the static pass nor trace type seeding may treat the result as plain T.
    let (program, entry, manifest) = compile_typescript_module(r#"
        function probe(xs: number[], n: number): number {
            let hits = 0;
            for (let i = 0; i < n; i = i + 1) {
                const v = xs[i % 5];
                if (v == null) hits = hits + 100; else hits = hits + v;
            }
            return hits;
        }
        const result = probe([1, 2, 3], 1000);
    "#);
    // Per 5-step cycle: 1 + 2 + 3 + 100 + 100.
    assert_eq!(static_type_results(&program, entry, &manifest), [200.0 * 206.0; 2]);
}

#[test]
fn napalm_inlined_receiver_stores_replay_correctly_on_class_guard_failure() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Particle {
            x: number;
            v: number;
            constructor(x: number, v: number) { this.x = x; this.v = v; }
            function advance(dt: number): number {
                this.x = this.x + this.v * dt;
                this.v = this.v * 0.5 + 1;
                return this.x;
            }
        }
        class Heavy extends Particle {
            constructor(x: number, v: number) { super(x, v); }
        }
        function nudge(p: Particle): number { return p.advance(0.25); }
        function run(): number {
            const items = [];
            const typed: Particle[] = [];
            for (let i = 0; i < 8; i = i + 1) {
                items.push(new Particle(i, 1));
                typed.push(new Particle(i * 2, 3));
            }
            let total = 0;
            for (let s = 0; s < 300; s = s + 1) {
                // A new receiver class makes the inlined class guard fail; the
                // exit must replay the call before any of its stores happened.
                if (s == 200) items.push(new Heavy(100, 2));
                for (let i = 0; i < items.length; i = i + 1) total = total + items[i].advance(0.5);
                for (let i = 0; i < typed.length; i = i + 1) total = total + nudge(typed[i]);
            }
            for (let i = 0; i < items.length; i = i + 1) total = total + items[i].x * 7 + items[i].v;
            for (let i = 0; i < typed.length; i = i + 1) total = total + typed[i].x * 5 + typed[i].v;
            return total;
        }
        const result = run();
    "#);
    let (results, stats) = static_type_native_run(&program, entry, &manifest);
    assert!(results[0].is_finite() && results[0] > 0.0, "{results:?}");
    assert_eq!(results[0], results[1], "native stores must match the interpreter");
    assert!(stats.inlined_calls > 0, "{stats:#?}");
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
    // Either the generic or the specialized (super-instruction) field encoding.
    assert!(absorb_ops.iter().any(|op| matches!(op, Opcode::LoadField | Opcode::GetFieldClass)), "{absorb_ops:?}");
    assert!(absorb_ops.iter().any(|op| matches!(op, Opcode::StoreField | Opcode::SetFieldClass)), "{absorb_ops:?}");
    assert!(!absorb_ops.contains(&Opcode::CallDirect), "{absorb_ops:?}");
    // Each iteration adds 2 to x; bump sees 2i+1 (+1 for `first`), absorb 2i+2.
    let expected: f64 = (0..500).map(|i| (2 * i + 2) as f64 + (2 * i + 2) as f64).sum();
    assert_eq!(static_type_results(&program, entry, &manifest), [expected; 2]);
}

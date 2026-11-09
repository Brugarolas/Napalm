fn generator_results(source: &str, expected: f64) {
    let (program, entry, manifest) = compile_typescript_module(&format!("{source}\nvar generatorTestResult = result;"));
    for jit in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !jit { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), expected, "jit={jit}");
    }
}

#[test]
fn napalm_generators_lazy_resume_return_and_completion() {
    generator_results(r#"
        let effects = 0;
        function* sequence(n: number): Generator<number, number, number> {
            effects += 1;
            const received = yield n;
            yield received + 1;
            return 99;
        }
        const g = sequence(3);
        let result = effects == 0 ? 1 : 0;
        const a = g.next(1000);
        const b = g.next(20);
        const c = g.next(0);
        const d = g.next();
        if (a.value == 3 && !a.done && b.value == 21 && !b.done &&
            c.value == 99 && c.done && d.value == null && d.done && effects == 1) result += 10;
    "#, 11.0);
}

#[test]
fn napalm_generators_delegation_nested_values_and_iteration() {
    generator_results(r#"
        function* leaf(): Generator<number, number, number> {
            const n = yield 1;
            yield n;
            return 30;
        }
        function* tree() { const n = yield* leaf(); yield n + 1; yield* [7, 8]; }
        const g = tree();
        let result = g.next().value + g.next(2).value + g.next(0).value;
        for (const x of g) result += x;
        function* children(): Generator<Generator<number>> { yield leaf(); yield leaf(); }
        for (const child of children()) result += child.next().value;
        function* flat() { yield* [1, 2, 3, 4, 5]; }
        const spread = [...flat()];
        result += spread.length;
    "#, 56.0);
}

#[test]
fn napalm_generators_throw_return_and_finally_suspension() {
    generator_results(r#"
        let cleaned = 0;
        function* child() {
            try { yield 1; } catch (e) { yield e; }
            finally { yield 8; cleaned += 1; }
            return 9;
        }
        function* parent() { return yield* child(); }
        const a = parent();
        let result = a.next().value;
        result += a.throw(4).value;
        const pending = a.return(20);
        if (pending.value == 8 && !pending.done) result += 10;
        const done = a.next();
        if (done.value == 20 && done.done && cleaned == 1) result += 100;
        const untouched = child();
        if (untouched.return(2).done && cleaned == 1) result += 1000;
        try { untouched.throw(null); } catch (e) { if (e == null) result += 10000; }
    "#, 11115.0);
}

#[test]
fn napalm_generators_expressions_methods_and_reentrancy() {
    generator_results(r#"
        const make = function* (x) { yield x; return x + 1; };
        class Counter {
            value: number;
            constructor(n: number) { this.value = n; }
            *values(): Generator<number> { yield this.value; yield* [2, 3]; }
        }
        let result = make(4).next().value;
        const c = new Counter(5);
        for (const x of c.values()) result += x;
        let g: any = null;
        function* recursive() { try { g.next(); } catch (e) { yield 10; } }
        g = recursive();
        result += g.next().value;
        function* bad() { throw "failure"; }
        const broken = bad();
        try { broken.next(); } catch (e) { if (e == "failure" && broken.next().done) result += 100; }
    "#, 124.0);
}

#[test]
fn napalm_generators_reject_yield_outside_its_own_body() {
    for source in ["yield 1;", "function f() { yield 1; }", "function* f() { const g = () => yield 1; }",
        "function* f() { function g() { yield 1; } }", "async function* f() { yield 1; }"] {
        assert!(crate::napalm_front_main::parse("bad_generator.napalm", source).has_errors(), "{source}");
    }
}

#[test]
fn napalm_generators_close_on_break_return_and_exception() {
    generator_results(r#"
        let closed = 0;
        function* values() { try { yield 1; yield 2; } finally { closed += 1; } }
        for (const n of values()) { break; }
        function first() { for (const n of values()) return n; }
        first();
        try { for (const n of values()) throw "stop"; } catch (e) { }
        for (const n of values()) { continue; }
        var result = closed;
    "#, 4.0);
}

#[test]
fn napalm_generators_nested_cleanup_and_return_replacement() {
    generator_results(r#"
        let cleaned = 0;
        function* g() {
            try {
                try { yield 1; } finally { yield 2; cleaned += 10; }
            } finally { cleaned += 100; }
        }
        const a = g();
        a.next();
        a.return(8);
        const done = a.return(9);
        var result = done.done && done.value == 9 ? cleaned : -1;
    "#, 100.0);
}

#[test]
fn napalm_generators_contracts_check_yield_resume_return_and_aliases() {
    for source in [
        "function* f(): Generator<number> { yield \"wrong\"; } f().next();",
        "function* f(): Generator<number, number> { yield 1; return \"wrong\"; } const g=f(); g.next(); g.next();",
        "function* f(): Generator<number, any, number> { const n = yield 1; yield n + 1; } const g=f(); g.next(); g.next(\"wrong\");",
        "function* f() { yield \"wrong\"; } const g: Generator<number> = f(); const alias: any = g; alias.next();",
        "function* inner() { yield \"wrong\"; } function* outer(): Generator<Generator<number>> { yield inner(); } outer().next().value.next();",
    ] {
        let parsed = crate::napalm_front_main::parse("generator_contract.napalm", source);
        assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
        let mut hir = Program::new();
        let _ = napalm_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(&mut hir, "generator_contract.napalm", source, "generator_contract", &parsed.module);
        // A statically known wrong return may already be rejected by lowering.
        if lowered.has_errors() { continue; }
        let (program, entry, manifest) = compile_typescript_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            let error = vm.run_module(entry).expect_err(source);
            assert!(error.message.starts_with(interfaces::TYPE_ERROR), "{error:?}");
        }
    }
}

#[test]
fn napalm_generators_void_return_and_runtime_names() {
    generator_results(r#"
        function* once(): Generator<number, void, number> { yield 1; return; }
        function* returnsAnything(): Generator<number, void> { yield 1; return 5; }
        const g = once();
        let result = typeof g == "Generator" ? 1 : 0;
        if (("" + g).startsWith("<generator#")) result += 10;
        g.next();
        if (g.next(0).done) result += 100;
        const r = returnsAnything();
        r.next();
        if (r.next().value == 5) result += 1000;
        function plain(): void { return 1; }
        plain();
    "#, 1111.0);
}

#[test]
fn napalm_generators_iterator_close_only_guards_possible_generators() {
    // A protected region disables deforestation and loop fusion, so only an
    // iterable that may be a Generator pays for IteratorClose.
    let regions = |source: &str| {
        let parsed = crate::napalm_front_main::parse("iterator_close.napalm", source);
        assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
        let mut hir = Program::new();
        let _ = napalm_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(&mut hir, "iterator_close.napalm", source, "iterator_close", &parsed.module);
        assert!(!lowered.has_errors(), "{:#?}", lowered.diagnostics);
        hir.functions.iter().map(|function| function.exception_regions.len()).sum::<usize>()
    };
    let none = regions("function f(xs: number[]) { let t = 0; return t; }");
    assert_eq!(regions("function f(xs: number[]) { let t = 0; for (const x of xs) t += x; return t; }"), none);
    assert_eq!(regions("function f() { let t = 0; for (const x of [1, 2]) t += x; return t; }"), none);
    assert_eq!(regions("function f(xs) { let t = 0; for (const x of xs) t += x; return t; }"), none + 1);
    assert_eq!(regions("function* g() { yield 1; } function f() { let t = 0; for (const x of g()) t += x; return t; }"),
        regions("function* g() { yield 1; } function f() { let t = 0; return t; }") + 1);
}

#[test]
fn napalm_generators_close_skips_finished_and_unstarted_generators() {
    generator_results(r#"
        let started = 0;
        let cleaned = 0;
        function* values() { started += 1; try { yield 1; yield 2; } finally { cleaned += 1; } }
        let total = 0;
        const g = values();
        for (const n of g) total += n;
        // Exhausted: the exit-path close must be a no-op, not a second resume.
        for (const n of g) total += 100;
        var result = total * 100 + started * 10 + cleaned;
    "#, 311.0);
}

#[test]
fn napalm_generators_typed_hot_loop_executes_native_code_and_survives_gc() {
    let source = r#"
        function* sums(n: number): Generator<number, void, number> {
            let total: number = 0;
            for (let round = 0; round < 6; round++) {
                for (let i = 0; i < n; i++) total += i;
                System.gc();
                const received = yield total;
                total += received;
            }
        }
        const g = sums(1000);
        let sum = 0;
        for (let i = 0; i < 6; i++) { sum += g.next(0).value; System.gc(); }
        g.next(0);
        var result = sum;
    "#;
    let (program, entry, manifest) = compile_typescript_module(source);
    let (results, stats) = static_type_native_run(&program, entry, &manifest);
    assert_eq!(results, [10_489_500.0; 2]);
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(program.functions.iter().any(|f| f.chunk.live_registers.iter().any(|entry|
        entry.types.iter().any(|(_, ty)| *ty == Type::Num))));
}

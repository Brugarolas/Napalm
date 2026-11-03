#[test]
fn napalm_jit_xorshift_merges_numeric_diamonds_without_side_traces() {
    let definitions = include_str!("../../scripts/jit.napalm")
        .split("let pi = JitBench.integratePi(1000000)").next().unwrap();
    let source = format!("{definitions}\nconst result = JitBench.xorshift(10000);");
    let (program, entry, manifest) = compile_typescript_module(&source);
    for optimized in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions { optimize_trace_ir: optimized, ..Default::default() }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 695495781.0);
        if optimized {
            let stats = vm.jit_stats().unwrap();
            assert_eq!(stats.side_traces_compiled, 0, "{stats:?}");
            assert!(stats.native_iterations > 9900, "{stats:?}");
            assert!(stats.promoted_f64_loop_phis >= 2, "{stats:?}");
        }
    }
}

#[test]
fn napalm_jit_numeric_diamonds_preserve_untaken_arms_and_effects() {
    for source in [
        r#"
            class Test {
                static run(n) {
                    let a = 0.125;
                    let b = -0.75;
                    let i = 0;
                    while (i < n) {
                        if ((i & 1) == 0) {
                            a = b * 2;
                            b = a + i;
                        } else {
                            b = a / 2;
                            a = b - i;
                        }
                        i += 1;
                    }
                    return a + b;
                }
            }
            const result = Test.run(500);
        "#,
        r#"
            class Test {
                static run(n, bad) {
                    let total = 0;
                    let i = 0;
                    while (i < n) {
                        if (i < n) { total = total + i; }
                        else { total = bad * 2; }
                        i += 1;
                    }
                    return total;
                }
            }
            const result = Test.run(500, "not a number");
        "#,
        r#"
            class Test {
                static run(n) {
                    let total = 0;
                    const calls = [];
                    let i = 0;
                    while (i < n) {
                        if ((i & 7) < 4) {
                            calls.add(i);
                            total = total + i * 3;
                        } else {
                            total = total - i % 17;
                        }
                        i += 1;
                    }
                    return total + calls.count;
                }
            }
            const result = Test.run(500);
        "#,
    ] {
        let (program, entry, manifest) = compile_typescript_module(source);
        let mut baseline = source_vm(&program, &manifest);
        baseline.set_jit_options(None);
        baseline.run_module(entry).unwrap();
        let mut vm = source_vm(&program, &manifest);
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), source_result(&baseline));
    }
}

#[test]
fn napalm_jit_inlines_typed_numeric_entry_kernel() {
    let definitions = include_str!("../../scripts/jit_entries.napalm")
        .split("EntryBench.run(1000)").next().unwrap();
    let source = format!("{definitions}\nconst result = EntryBench.run(2000);");
    let (program, entry, manifest) = compile_typescript_module(&source);
    let mut baseline = source_vm(&program, &manifest);
    baseline.set_jit_options(None);
    baseline.run_module(entry).unwrap();
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), source_result(&baseline));
    let stats = vm.jit_stats().unwrap();
    assert!(stats.inlined_calls > 0, "{stats:?}");
    assert_eq!(stats.invoke_direct_calls, 0, "{stats:?}");
    assert!(stats.native_iterations > 1800, "{stats:?}");
}

#[test]
fn napalm_jit_inlined_primitive_bindings_replay_type_errors() {
    for (ty, valid, invalid) in [("number", "1", "'bad'"), ("boolean", "true", "1"), ("null", "null", "false")] {
        let source = format!(r#"
            class Test {{
                static check(x) {{
                    let value: {ty} = x;
                    return value;
                }}
                static run() {{
                    const items = [{valid}, {invalid}];
                    let result = {valid};
                    let i = 0;
                    while (i < 500) {{
                        result = this.check(items[i < 400 ? 0 : 1]);
                        i += 1;
                    }}
                    return result;
                }}
            }}
            const result = Test.run();
        "#);
        let (program, entry, manifest) = compile_typescript_module(&source);
        for native in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !native { vm.set_jit_options(None); }
            let error = vm.run_module(entry).expect_err("inlined binding must still fail");
            assert!(error.message.contains("Structural type error:"), "{ty}: {error}");
            if native {
                let stats = vm.jit_stats().unwrap();
                assert!(stats.native_entries > 0, "{ty}: {stats:?}");
                assert!(stats.inlined_calls > 0, "{ty}: {stats:?}");
            }
        }
    }
}

#[test]
fn napalm_jit_numeric_bindings_stay_native_in_mandelbrot() {
    let definitions = include_str!("../../scripts/jit.napalm")
        .split("let pi = JitBench.integratePi(1000000)").next().unwrap();
    let source = format!("{definitions}\nconst result = JitBench.mandelbrot(80, 60, 100);");
    let (program, entry, manifest) = compile_typescript_module(&source);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), 196065774.0);
    let jit = vm.jit.as_ref().unwrap();
    assert!(jit.stats().traces_compiled > 0);
    assert!(jit.stats().promoted_f64_loop_phis > 0, "{:?}", jit.stats());
    let mut doubled = false;
    for function in &vm.prepared.functions {
        for (index, &cached) in function.jit_trace_headers.iter().enumerate() {
            if cached == NO_INDEX { continue; }
            let trace = jit.trace(TraceId(cached)).unwrap();
            doubled |= trace.nodes.iter().any(|node| matches!(node.kind,
                crate::napalm_trace_ir::TraceInst::F64Binary { op: crate::napalm_ir::F64BinOp::Add, lhs, rhs } if lhs == rhs
            ));
            assert_eq!(trace.anchor.location, TraceLocation::new(function.id, BytecodePc(function.original_ops[index].raw_pc)));
            assert!(!trace.nodes.iter().any(|node| matches!(node.kind,
                crate::napalm_trace_ir::TraceInst::Runtime { .. }
                | crate::napalm_trace_ir::TraceInst::RuntimeCall { .. }
            )), "numeric loops must not redispatch primitive bindings");
        }
    }
    assert!(doubled, "the loop must retain the hoisted 2.0 literal and reduce multiplication to addition");
}

#[test]
fn napalm_jit_primitive_bindings_replay_type_errors_after_warmup() {
    for (ty, valid, invalid) in [("number", "1", "'bad'"), ("boolean", "true", "1"), ("null", "null", "false")] {
        let source = format!(r#"
            const items = [{valid}, {invalid}];
            function run() {{
                let value: {ty} = {valid};
                let i = 0;
                while (i < 500) {{
                    value = items[i < 400 ? 0 : 1];
                    i += 1;
                }}
                return value;
            }}
            const result = run();
        "#);
        let (program, entry, manifest) = compile_typescript_module(&source);
        for native in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !native { vm.set_jit_options(None); }
            let error = vm.run_module(entry).expect_err("invalid primitive must still fail");
            assert!(error.message.contains("Structural type error:"), "{ty}: {error}");
            if native { assert!(vm.jit_stats().unwrap().native_entries > 0, "{ty}"); }
        }
    }
}

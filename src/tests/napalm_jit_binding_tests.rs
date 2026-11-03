#[test]
fn napalm_jit_numeric_bindings_stay_native_in_mandelbrot() {
    let definitions = include_str!("../../scripts/jit.napalm")
        .split("JitBench.integratePi(1000000)").next().unwrap();
    let source = format!("{definitions}\nconst result = JitBench.mandelbrot(80, 60, 100);");
    let (program, entry, manifest) = compile_typescript_module(&source);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), 196065774.0);
    let jit = vm.jit.as_ref().unwrap();
    assert!(jit.stats().traces_compiled > 0);
    assert!(jit.stats().promoted_f64_loop_phis > 0, "{:?}", jit.stats());
    for function in &vm.prepared.functions {
        for (index, &cached) in function.jit_trace_headers.iter().enumerate() {
            if cached == NO_INDEX { continue; }
            let trace = jit.trace(TraceId(cached)).unwrap();
            assert_eq!(trace.anchor.location, TraceLocation::new(function.id, BytecodePc(function.original_ops[index].raw_pc)));
            assert!(!trace.nodes.iter().any(|node| matches!(node.kind,
                crate::napalm_trace_ir::TraceInst::Runtime { .. }
                | crate::napalm_trace_ir::TraceInst::RuntimeCall { .. }
            )), "numeric loops must not redispatch primitive bindings");
        }
    }
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

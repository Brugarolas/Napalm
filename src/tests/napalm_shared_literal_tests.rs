// Regressions for whole-program sharing of immutable constant map literals.
fn shared_literal_program(source: &str) -> (BytecodeProgram, FuncId, BuiltinManifest, u32) {
    let parsed = crate::napalm_front_main::parse("shared_literals.napalm", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "shared_literals.napalm", source, "test", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let entry = lowered.lowered.unwrap().entry;
    let program = compile_program(&mut hir).unwrap();
    let shared = hir.modules.iter().map(|module| module.hidden_variable_count).sum();
    (program, entry, manifest, shared)
}

fn shared_literal_result(source: &str) -> (String, u32) {
    let (program, entry, manifest, shared) = shared_literal_program(source);
    let mut outputs = Vec::new();
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        outputs.push(HostContext { heap: &mut vm.heap }.format_value(value));
    }
    assert_eq!(outputs[0], outputs[1], "interpreter and JIT tiers disagree");
    (outputs.pop().unwrap(), shared)
}

#[test]
fn napalm_shared_literals_hoist_read_only_tree_leaves() {
    let (result, shared) = shared_literal_result(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        let sum = 0;
        for (let i = 0; i < 20; i++) sum += count(make(8));
        const result = sum;
    "#);
    assert_eq!(result, "10220");
    assert_eq!(shared, 1);

    // Control for the negative cases below: read-only use of the same literal is shared.
    let (result, shared) = shared_literal_result(r#"
        function leaf() { return {v: 1}; }
        const result = leaf().v + leaf().v;
    "#);
    assert_eq!(result, "2");
    assert_eq!(shared, 1);
}

#[test]
fn napalm_shared_literals_keep_observable_literals_fresh() {
    let cases = [
        // Direct mutation.
        (r#"function leaf() { return {v: 1}; }
            const a = leaf(); a.v = 2;
            const result = leaf().v;"#, "1"),
        // Mutation inside a callee.
        (r#"function leaf() { return {v: 1}; }
            function bump(o) { o.v = o.v + 1; return o.v; }
            bump(leaf());
            const result = bump(leaf());"#, "2"),
        // Mutation through a containing literal.
        (r#"function leaf() { return {v: 1}; }
            function wrap() { return {inner: leaf()}; }
            wrap().inner.v = 9;
            const result = wrap().inner.v;"#, "1"),
        // Identity comparison.
        (r#"function leaf() { return {v: 1}; }
            const result = leaf() == leaf();"#, "false"),
        // Escape into a list.
        (r#"function leaf() { return {v: 1}; }
            const xs = [leaf(), leaf()];
            xs[0].v = 5;
            const result = xs[1].v;"#, "1"),
        // Escape into a module global.
        (r#"function leaf() { return {v: 1}; }
            let saved = leaf();
            saved.v = 3;
            const result = leaf().v;"#, "1"),
        // The producing function is used as a first-class value.
        (r#"function leaf() { return {v: 1}; }
            const make = leaf;
            make().v = 4;
            const result = make().v;"#, "1"),
        // Builtin methods that may mutate the receiver.
        (r#"function leaf() { return {v: 1}; }
            leaf().remove("v");
            const result = leaf().v;"#, "1"),
    ];
    for (source, expected) in cases {
        let (result, shared) = shared_literal_result(source);
        assert_eq!(result, expected, "{source}");
        assert_eq!(shared, 0, "{source}");
    }
}

#[test]
fn napalm_shared_literals_skip_programs_with_concurrent_workers() {
    // Workers never run the module body, so hidden slots would stay
    // uninitialized there. The program is only compiled: launching workers
    // while other tests collect garbage can deadlock the shared test process.
    let (_, _, _, shared) = shared_literal_program(r#"
        function leaf() { return {v: 1}; }
        function read(n) { return leaf().v + n; }
        concurrent function job(n) { return read(n); }
        let pending = launch job(1);
        let worker = await pending;
        let result = worker + read(0);
    "#);
    assert_eq!(shared, 0);
}

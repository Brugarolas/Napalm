// Automatic sharing must preserve fresh-object semantics whenever observable.
fn auto_memo_policy(program: &BytecodeProgram, ordinal: usize) -> (usize, MemoizationKind) {
    // Source declarations lower to anonymous closure names, in source order.
    let module = program.functions.last().unwrap().module;
    program.functions.iter().enumerate()
        .filter(|(_, function)| function.module == module && function.kind == FunctionKind::Closure)
        .nth(ordinal).map(|(index, function)| (index, function.memoization))
        .expect("source function")
}

#[test]
fn napalm_auto_memo_trees_share_by_depth_and_survive_gc() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        function forward(depth) { return make(depth); }
        function run() {
            let sum = count(forward(8));
            System.gc();
            for (let i = 0; i < 12; i++) sum += count(make(8));
            return sum;
        }
        const result = run();
    "#);
    let (make, policy) = auto_memo_policy(&program, 0);
    assert_eq!(policy, MemoizationKind::RecursiveShared);
    assert_eq!(auto_memo_policy(&program, 2).1, MemoizationKind::None);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 13.0 * 511.0);
        let table = vm.memo_tables[make].as_ref().unwrap();
        assert_eq!(table.entries.len(), 9, "one cached tree per depth");
        assert!(table.hits >= 20);
        let cached: Vec<_> = table.entries.values().map(|entry| Value::from_word(entry.value)).collect();
        // No source global retains a tree. Between VM entrypoints only the memo
        // cache owns these results; force GC without installing an active provider.
        vm.heap.collect_full();
        for value in cached {
            assert!(matches!(vm.heap.get(value), Some(HeapObject::Map(_))));
        }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 13.0 * 511.0);
    }
}

#[test]
fn napalm_auto_memo_objects_reject_observable_sharing() {
    let cases = [
        // Mutation, including recursive child aliases and another caller.
        ("function run() { let a = make(2); a.left.left = 9; return make(2).left.left == 9; } const result = run();", "false"),
        ("function mutate(a) { a.left = 7; } function run() { mutate(make(2)); return make(2).left == 7; } const result = run();", "false"),
        // Fresh calls and sibling subtrees have distinct identity.
        ("const result = make(2) == make(2);", "false"),
        ("function run() { let a = make(2); return a.left == a.right; } const result = run();", "false"),
        // Unanalyzed containers, globals and first-class function references.
        ("const xs = [make(2), make(2)]; xs[0].left = 7; const result = xs[1].left == 7;", "false"),
        ("const saved = make(2); saved.left = 7; const result = make(2).left == 7;", "false"),
        ("const alias = make; alias(2).left = 7; const result = make(2).left == 7;", "false"),
        // Mutating Map methods cannot be treated as field reads.
        ("function run() { make(2).remove('left'); return make(2).left == null; } const result = run();", "false"),
    ];
    for (use_site, expected) in cases {
        let source = format!(r#"
            function make(depth) {{
                if (depth > 0) return {{left: make(depth - 1), right: make(depth - 1)}};
                return {{left: null, right: null}};
            }}
            {use_site}
        "#);
        let (program, entry, manifest) = compile_typescript_module(&source);
        assert_eq!(auto_memo_policy(&program, 0).1, MemoizationKind::None, "{use_site}");
        for native in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !native { vm.set_jit_options(None); }
            vm.run_module(entry).unwrap();
            let result = last_source_variable(&vm);
            assert_eq!(HostContext { heap: &mut vm.heap }.format_value(result), expected, "{use_site}");
        }
    }
}

#[test]
fn napalm_auto_memo_objects_reject_effects_and_public_producers() {
    for source in [
        "let state = 1; function make(n) { return {v: state + n}; } const result = make(1).v;",
        "let state = 0; function make(n) { state += 1; return {v: n}; } const result = make(1).v;",
        "function make(n) { return {v: System.clock}; } const result = make(1).v;",
        "export function make(n) { return {v: n}; } const result = make(1).v;",
        "function make(n) { return {v: n}; } concurrent function work(n) { return make(n).v; } const result = make(1).v;",
    ] {
        let (program, _, _) = compile_typescript_module(source);
        assert_eq!(auto_memo_policy(&program, 0).1, MemoizationKind::None, "{source}");
    }
}

#[test]
fn napalm_auto_memo_keeps_nonrecursive_factories_and_callers_uncached() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function leaf(n) { return {value: n}; }
        function wrap(n) { return {child: leaf(n)}; }
        function forward(n) { return wrap(n); }
        const result = (forward(7).child == null ? 0 : 7) + (forward(7).child == null ? 0 : 7);
    "#);
    for ordinal in 0..3 {
        assert_eq!(auto_memo_policy(&program, ordinal).1, MemoizationKind::None);
    }
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), 14.0);
    let forward = auto_memo_policy(&program, 2).0;
    assert!(vm.memo_tables[forward].is_none());
}

#[test]
fn napalm_auto_memo_leaves_static_benchmark_kernels_uncached() {
    for (source, names) in [
        (include_str!("../../scripts/jit.napalm"), &["JitBench.integratePi", "JitBench.xorshift", "JitBench.mandelbrot"][..]),
        (include_str!("../../scripts/jit_entries.napalm"), &["EntryBench.kernel", "EntryBench.run"][..]),
    ] {
        let (program, _, _) = compile_typescript_module(source);
        let module = program.functions.last().unwrap().module;
        let methods: Vec<_> = program.functions.iter()
            .filter(|function| function.module == module && function.kind == FunctionKind::StaticMethod)
            .collect();
        for name in names {
            let prefix = format!("{name}(");
            let function = methods.iter().find(|function| {
                program.metadata.strings[function.name.0 as usize].starts_with(&prefix)
            }).expect("benchmark method");
            assert_eq!(function.memoization, MemoizationKind::None, "{:?}", function.id);
        }
    }
}

const SHARED_MEMO_FIB: &str = r#"
    function fib(n) {
        if (n < 2) return n;
        return fib(n - 1) + fib(n - 2);
    }
    const result = fib(30);
"#;

#[test]
fn napalm_auto_memo_shares_scalar_results_across_vms() {
    let (program, entry, manifest) = compile_typescript_module(SHARED_MEMO_FIB);
    let (fib, policy) = auto_memo_policy(&program, 0);
    assert_eq!(policy, MemoizationKind::Recursive);

    let mut parent = source_vm(&program, &manifest);
    parent.set_jit_options(None);
    parent.run_module(entry).unwrap();
    assert_eq!(source_result(&parent), 832040.0);
    let seeded = parent.memo_tables[fib].as_ref().unwrap().entries.len();
    assert!(seeded > 1);
    // The first launch seeds the domain from the parent's private results and attaches it.
    let shared = parent.shared_memo_for_workers().expect("memoized program has a shared domain");
    assert!(Arc::ptr_eq(parent.shared_memo.as_ref().unwrap(), &shared));
    assert_eq!(shared.len(fib as u32), seeded);

    // A joined VM reuses the published top-level result instead of recursing.
    let mut worker = source_vm(&program, &manifest);
    worker.set_jit_options(None);
    worker.set_shared_memo(Some(shared.clone()));
    assert!(worker.shared_memo.is_some());
    worker.run_module(entry).unwrap();
    assert_eq!(source_result(&worker), 832040.0);
    let table = worker.memo_tables[fib].as_ref().unwrap();
    assert_eq!((table.entries.len(), table.hits), (1, 1));
}

#[test]
fn napalm_auto_memo_builtin_rebinding_leaves_shared_domain() {
    let (program, entry, manifest) = compile_typescript_module(SHARED_MEMO_FIB);
    let fib = auto_memo_policy(&program, 0).0 as u32;
    let plus = program.metadata.methods.iter().position(|method| {
        program.metadata.string(method.name) == Some("+")
            && method.kind == crate::napalm_ir::MethodKind::Method
            && method.arity == 1
    }).map(|index| MethodSymbol(index as u32)).expect("Num + method");

    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(None);
    vm.run_module(entry).unwrap();
    let shared = vm.shared_memo_for_workers().unwrap();
    assert!(vm.shared_memo.is_some());

    vm.bind_builtin_primitive(BuiltinClass::Num, plus, PrimitiveId(0));
    assert!(vm.shared_memo.is_none());
    // Customized dispatch can neither rejoin nor seed a later workers-only domain.
    vm.set_shared_memo(Some(shared.clone()));
    assert!(vm.shared_memo.is_none());
    let workers_only = vm.shared_memo_for_workers().unwrap();
    assert!(!Arc::ptr_eq(&shared, &workers_only));
    assert_eq!(workers_only.len(fib), 0);
    assert!(vm.shared_memo.is_none());
}

#[test]
fn napalm_auto_memo_launch_workers_share_scalar_results() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function fib(n) {
            if (n < 2) return n;
            return fib(n - 1) + fib(n - 2);
        }
        concurrent function work(n: Number): Number {
            return fib(n);
        }
        let warm: Number = fib(30);
        let p1 = launch work(30);
        let p2 = launch work(31);
        let r1: Number = await p1;
        let r2: Number = await p2;
        let result = warm + r1 + r2;
    "#);
    let (fib, policy) = auto_memo_policy(&program, 0);
    assert_eq!(policy, MemoizationKind::Recursive);

    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(None);
    let ExecutionResult::Returned(_) = vm.run_module(entry).unwrap() else {
        panic!("expected module return after concurrent jobs settle")
    };
    assert_eq!(source_result(&vm), 832040.0 * 2.0 + 1346269.0);
    let shared = vm.shared_memo.as_ref().expect("parent joins its workers' domain");
    // Parent-seeded fib(0..=30) plus fib(31), published by a worker.
    assert!(shared.len(fib as u32) > vm.memo_tables[fib].as_ref().unwrap().entries.len());
}


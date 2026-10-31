// Regressions for whole-function native recursion and moving inline Map storage.
fn whole_function_program(source: &str) -> (BytecodeProgram, FuncId, BuiltinManifest) {
    let parsed = crate::napalm_front_main::parse("whole_function.napalm", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "whole_function.napalm", source, "test", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let entry = lowered.lowered.unwrap().entry;
    (compile_program(&mut hir).unwrap(), entry, manifest)
}

#[test]
fn napalm_whole_function_trees_survive_gc_and_shape_bailouts() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        let sum = 0;
        for (let i = 0; i < 40; i++) {
            const tree = make(10);
            System.gc();
            sum += count(tree);
            // Different shape and slot order inside an already-native walk.
            tree.left = {extra: 7, right: null, left: null};
            System.gc();
            sum += count(tree);
        }
        const result = sum;
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 40.0 * (2047.0 + 1025.0));
        assert!(vm.gc_statistics().collections > 0);
        if native {
            let stats = vm.whole_function_jit_stats().unwrap();
            assert!(stats.compiled_functions >= 2, "{stats:?}");
            assert!(stats.native_recursive_calls > 1000, "{stats:?}");
        }
        assert!(vm.native_shadow_top.is_null());
    }
}

#[test]
fn napalm_whole_function_identity_guards_preserve_other_equality() {
    let (program, entry, manifest) = whole_function_program(r#"
        function equal(left, right) {
            if (left == right) return {value: 1};
            return {value: 0};
        }
        const object = {};
        let sum = 0;
        for (let i = 0; i < 400; i++) {
            sum += equal(object, object).value + equal(null, null).value;
        }
        // Numeric and string equality must fall back from the identity PIC.
        const result = [sum, equal(0, -0).value, equal(0 / 0, 0 / 0).value,
            equal('a' + 'b', 'ab').value, equal(true, false).value, equal(object, {}).value];
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.run_module(entry).unwrap();
    let value = *vm.modules.last().unwrap().variables.last().unwrap();
    assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[800, 1, 0, 1, 0, 0]");
    assert!(vm.whole_function_jit_stats().unwrap().compiled_functions > 0);
}

#[test]
fn napalm_whole_function_recursion_replays_guards_and_depth_limit() {
    let (program, entry, manifest) = whole_function_program(r#"
        function chain(depth, value) {
            if (depth > 0) return {next: chain(depth - 1, value), value: depth + value};
            return {next: null, value: 0 + value};
        }
        let warm = chain(160, 1);
        let deep = chain(300, 2);
        let text = chain(20, 'x');
        System.gc();
        let count = 0;
        while (deep != null) { count++; deep = deep.next; }
        while (text.next != null) text = text.next;
        const result = [count, warm.value, text.value];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = *vm.modules.last().unwrap().variables.last().unwrap();
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[301, 161, 0x]");
        assert!(vm.native_shadow_top.is_null());
        if native { assert!(vm.whole_function_jit_stats().unwrap().native_recursive_calls > 0); }
    }
}

#[test]
fn napalm_inline_maps_grow_promote_and_keep_live_sidecars() {
    run_napalm_checked(r#"
        const map = {a: {value: 41}, b: 2};
        let sum = 0;
        for (let i = 0; i < 100; i++) {
            sum += map.a.value;
            if (i == 30) { map.c = 3; map.d = 4; map.e = 5; }
            if (i == 60) map[7] = 'seven';
            if (i % 10 == 0) System.gc();
        }
        const result = [sum, map.a.value, map.e, map[7]];
    "#, "[4100, 41, 5, seven]", true);
}

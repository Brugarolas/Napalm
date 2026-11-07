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
fn napalm_whole_function_tiering_avoids_redundant_recursive_entry_traces() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {left: make(depth - 1), right: make(depth - 1)};
            return {left: null, right: null};
        }
        function count(node) {
            if (node.left == null) return 1;
            return 1 + count(node.left) + count(node.right);
        }
        // Keep this a native-recursion test: observing mutation makes the
        // producer ineligible for automatic object memoization.
        make(1).left = null;
        const result = count(make(10));
    "#);
    for threshold in [0, 64, 96] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(Some(TracingJitOptions {
            hot_entry_threshold: threshold,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 2047.0);
        let stats = vm.whole_function_jit_stats().unwrap();
        assert_eq!(stats.compiled_functions, if threshold == 0 { 0 } else { 2 }, "{stats:?}");
        let traces = vm.jit_stats().unwrap();
        assert_eq!(traces.entry_traces_compiled, 0, "{traces:?}");
        if threshold == 0 {
            assert_eq!(stats.compile_attempts, 0, "{stats:?}");
        } else {
            assert!(stats.native_recursive_calls > 1000, "{stats:?}");
        }
        assert!(vm.native_shadow_top.is_null());
    }
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
            assert!(stats.inlined_recursive_sites >= 4, "{stats:?}");
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
    let value = last_source_variable(&vm);
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
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[301, 161, 0x]");
        assert!(vm.native_shadow_top.is_null());
        if native { assert!(vm.whole_function_jit_stats().unwrap().native_recursive_calls > 0); }
    }
}

#[test]
fn napalm_inline_maps_grow_promote_and_keep_live_sidecars() {
    // The loop lives in a function: module-scope variables use runtime opcodes
    // and would never let the trace reach native code.
    run_napalm_checked(r#"
        function run() {
            const map = {a: {value: 41}, b: 2};
            let sum = 0;
            for (let i = 0; i < 100; i++) {
                sum += map.a.value;
                if (i == 30) { map.c = 3; map.d = 4; map.e = 5; }
                if (i == 60) map[7] = 'seven';
                if (i % 10 == 0) System.gc();
            }
            return [sum, map.a.value, map.e, map[7]];
        }
        const result = run();
    "#, "[4100, 41, 5, seven]", true);
}

#[test]
fn napalm_whole_function_read_only_recursion_replays_deep_and_changed_shapes() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(depth) {
            if (depth > 0) return {next: make(depth - 1)};
            return {next: null};
        }
        function count(node) {
            if (node == null) return 0;
            return 1 + count(node.next);
        }
        const short = make(20);
        let total = 0;
        for (let i = 0; i < 100; i++) total += count(short);
        const deep = make(300);
        System.gc();
        const deepCount = count(deep);
        short.next.next = {extra: 42, next: {next: null}};
        System.gc();
        const result = [total, deepCount, count(short)];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[2100, 301, 4]");
        assert!(vm.native_shadow_top.is_null());
        if native {
            assert!(vm.whole_function_jit_stats().unwrap().compiled_functions >= 2);
            assert!(vm.whole_function_jit_stats().unwrap().native_recursive_calls > 1000);
            assert!(vm.whole_function_jit_stats().unwrap().inlined_recursive_sites >= 2);
        }
    }
}

#[test]
fn napalm_whole_function_literal_slots_preserve_duplicates_and_dynamic_keys() {
    let (program, entry, manifest) = whole_function_program(r#"
        function make(key, value) {
            return {b: value, [key]: value + 1, b: value + 2, c: value + 3};
        }
        let sum = 0;
        for (let i = 0; i < 500; i++) {
            const object = make('a', i);
            if (i % 25 == 0) System.gc();
            sum += object.a + object.b + object.c;
        }
        const changed = make('c', 10);
        System.gc();
        const result = [sum, changed.b, changed.c, changed.a];
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        let value = last_source_variable(&vm);
        assert_eq!(HostContext { heap: &mut vm.heap }.format_value(value), "[377250, 12, 13, null]");
        if native { assert!(vm.whole_function_jit_stats().unwrap().compiled_functions > 0); }
    }
}

#[test]
fn napalm_profiled_invoke_promotes_to_direct_method_and_compiles_natively() {
    // `shape` is untyped, so WIR cannot devirtualize `shape.area()`. The VM's
    // call-site profile promotes the monomorphic site to InvokeMethodDirect and
    // the whole-function JIT compiles `measure` (it used to reject it). A later
    // receiver of another class must fall back to the generic Invoke.
    let (program, entry, manifest) = whole_function_program(r#"
        class Square { side: number; constructor(s: number) { this.side = s; } area(): number { return this.side * this.side; } }
        class Circle { r: number; constructor(r: number) { this.r = r; } area(): number { return 3 * this.r * this.r; } }
        function measure(shape) { return shape.area() + 1; }
        const square = new Square(3);
        let sum = 0;
        for (let i = 0; i < 400; i++) sum += measure(square);
        sum += measure(new Circle(2));
        for (let i = 0; i < 10; i++) sum += measure(square);
        const result = sum;
    "#);
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        // Isolate the whole-function tier: no loop traces may inline `measure`.
        vm.set_jit_options(native.then(|| TracingJitOptions {
            hot_loop_threshold: u32::MAX,
            hot_entry_threshold: 64,
            ..Default::default()
        }));
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), 410.0 * 10.0 + 13.0);
        if native {
            let stats = vm.whole_function_jit_stats().unwrap();
            assert!(stats.compiled_functions >= 1, "{stats:?}");
            assert!(stats.native_entries > 0, "{stats:?}");
        }
    }
}

#[test]
fn inline_instance_fields_do_not_grow_managed_objects() {
    // Up to INSTANCE_INLINE_FIELDS fields live inside the managed object; the
    // fixed managed payload (every object pays it) must not grow for that.
    assert_eq!(std::mem::size_of::<HeapObject>(), std::mem::size_of::<NapalmMap>());
    assert_eq!(std::mem::size_of::<HeapObject>(), 48);
    assert!(InstanceObject::nulls(0, INSTANCE_INLINE_FIELDS).is_inline());
    assert!(!InstanceObject::nulls(0, INSTANCE_INLINE_FIELDS + 1).is_inline());
    assert_eq!(InstanceObject::nulls(0, 3).len(), 3);
    assert_eq!(InstanceObject::nulls(0, 7).len(), 7);
}

#[test]
fn inline_and_boxed_instance_fields_survive_moving_gc_in_native_code() {
    let (program, entry, manifest) = whole_function_program(r#"
        class Small { a: number; b: number; constructor(a: number, b: number) { this.a = a; this.b = b; } }
        class Wide {
            a: number; b: number; c: number; d: number; e: number; f: number;
            constructor(x: number) { this.a = x; this.b = x + 1; this.c = x + 2; this.d = x + 3; this.e = x + 4; this.f = x + 5; }
        }
        function run(n: number): number {
            const keep = [];
            let sum = 0;
            for (let i = 0; i < n; i++) {
                const s = new Small(i, 2);
                const w = new Wide(i);
                if (i % 50 == 0) { keep.push(s); keep.push(w); System.gc(); }
                s.b = s.b + w.f;
                sum = sum + s.a + s.b + w.a + w.f;
            }
            for (let j = 0; j < keep.length; j++) {
                const o = keep[j];
                sum = sum + o.a + o.b;
            }
            return sum;
        }
        const result = run(2000);
    "#);
    let mut expected = 0.0f64;
    let mut kept = 0.0f64;
    for i in 0..2000 {
        let i = i as f64;
        let b = 2.0 + i + 5.0;
        expected += i + b + i + (i + 5.0);
        if (i as i64) % 50 == 0 { kept += (i + b) + (i + (i + 1.0)); }
    }
    for native in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !native { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), expected + kept, "native={native}");
        assert!(vm.gc_statistics().collections > 0);
    }
}

#[test]
fn inline_instance_sidecar_uses_gc_relative_field_storage() {
    for count in [2usize, 9] {
        let entry = HandleEntry::new(FREE_HANDLE_NONE);
        let managed = ManagedObject {
            handle: 0,
            object: HeapObject::Instance(InstanceObject::nulls(3, count)),
        };
        let base = (&managed as *const ManagedObject) as usize;
        HandleTable::refresh_entry(&entry, base, &managed.object);
        let encoded = entry.data.load(Ordering::Relaxed);
        let fields = match &managed.object { HeapObject::Instance(fields) => fields.as_ptr() as usize, _ => unreachable!() };
        let relative = encoded & crate::napalm_tracing_jit::TRACE_HEAP_DATA_RELATIVE_MASK != 0;
        assert_eq!(relative, count <= INSTANCE_INLINE_FIELDS, "count {count}");
        // The branch-free JIT decode resolves both representations.
        let mask = crate::napalm_tracing_jit::TRACE_HEAP_DATA_RELATIVE_MASK;
        let decoded = (encoded & !mask) + (base & (encoded & mask).wrapping_neg());
        assert_eq!(decoded, fields, "count {count}");
        assert_eq!(entry.len.load(Ordering::Relaxed), count);
    }
}

#[test]
fn napalm_whole_function_small_maps_enable_tlab_allocation() {
    let mut heap = Heap::default();
    let info = heap.inline_map_fast_allocation_info()
        .expect("small managed objects must support direct TLAB allocation");
    assert!(!info.cursor.is_null() && !info.limit.is_null());
    assert!(info.bytes >= std::mem::size_of::<ManagedObject>());
    assert!(info.alignment.is_power_of_two());
}

#[test]
fn napalm_collections_instance_protocol_inheritance_overrides_and_gc() {
    run_napalm(r#"
        class Base { name = 'base'; }
        class Child extends Base { value = 42; }
        class Empty {}
        class Custom extends Child { keys() { return ['custom']; } }
        const item = new Child();
        const entries = item.entries();
        System.gc();
        const result = [item.keys(), item.values(), entries, item.size,
            new Empty().keys(), new Empty().size, new Custom().keys()];
    "#, "[[name, value], [base, 42], [[name, base], [value, 42]], 2, [], 0, [custom]]");
}

#[test]
fn napalm_collections_instance_subscript_reads_and_writes_fields_by_name() {
    run_napalm(r#"
        interface Point { x: number; y: number; }
        class Base { name = 'base'; }
        class P extends Base implements Point {
            constructor() { super(); this.x = 3; this.y = 40; }
        }
        function read(p: Point, k: string) { return p[k]; }
        function write(p: Point, k: string, v: number) { p[k] = v; }
        const p = new P();
        const parts = ['na', 'me'];
        const first = [p['x'], read(p, 'y'), p[parts[0] + parts[1]], p['missing'], p[parts[1]], p[7]];
        write(p, 'x', 5);
        p[parts[0] + parts[1]] = 'renamed';
        p['y'] += 2;
        let total = 0;
        const keys = ['x', 'y'];
        for (let i = 0; i < 200; i++) {
            const k = keys[i % 2];
            p[k] = p[k] + 1;
            total += read(p, k);
        }
        const result = [first, p.x, p.y, p.name, total, p.keys()];
    "#, "[[3, 40, base, null, null, null], 105, 142, renamed, 14800, [name, x, y]]");
}

#[test]
fn napalm_collections_instance_subscript_is_reactive() {
    run_napalm(r#"
        let evaluations = 0;
        class Counter {
            constructor() { this.hot = 1; this.cold = 100; this.key = 'hot'; }
            computed doubled() { evaluations++; return this[this.key] * 2; }
        }
        const c = new Counter();
        const first = [c.doubled, c.doubled];
        c['hot'] = 5;                    // subscript write invalidates
        const second = [c.doubled, c.doubled];
        c.hot = 7;                       // subscript read was tracked
        const third = c.doubled;
        c['cold'] = 200;                 // unrelated field: still cached
        const result = [first, second, third, c.doubled, evaluations];
    "#, "[[2, 2], [10, 10], 14, 14, 3]");
}

#[test]
fn napalm_collections_instance_subscript_keeps_field_types_and_layout() {
    for (source, message) in [
        ("class P { x: number = 1; } const p = new P(); const k = 'mis' + 'sing'; p[k] = 1;",
            "cannot add property 'missing' to a class instance"),
        ("class P { x: number = 1; } const p = new P(); const k = 'x'; p[k] = 'text';",
            "Class field type error"),
    ] {
        let (program, entry, manifest) = compile_typescript_module(source);
        for jit in [false, true] {
            let mut vm = source_vm(&program, &manifest);
            if !jit { vm.set_jit_options(None); }
            let error = vm.run_module(entry).unwrap_err();
            assert!(error.message.contains(message), "jit={jit} {source}: {error:?}");
        }
    }
}

#[test]
fn napalm_collections_map_set_api() {
    run_napalm(r#"
        const m = new Map();
        const same = m.set('name', 'Napalm') == m;
        m.set('nullable', null).set(123, 'number');
        const presentNull = m.has('nullable');
        const deletedNull = m.delete('nullable');
        const missing = m.delete('missing');
        const s = new Set();
        const sameSet = s.add('first') == s;
        s.add(42).add('first');
        const before = [m.get('name'), m.get(123), m.get('missing'), m.size,
            presentNull, deletedNull, missing, same, sameSet, s.size, s.has(42)];
        const removed = [s.delete(42), s.delete(42), s.has(42), s.size];
        m.clear(); s.clear(); s.add(7);
        const result = [before, removed, m.size, [...s], s.count, s.isEmpty];
    "#, "[[Napalm, number, null, 2, true, true, false, true, true, 2, true], [true, false, false, 1], 0, [7], 1, false]");
}

#[test]
fn napalm_collections_keys_identity_and_gc() {
    run_napalm(r#"
        const a = {value: 1}; const b = {value: 1};
        const s = new Set();
        s.add(a).add(b).add(a).add(null).add(0).add(-0).add(Num.nan).add(Num.nan);
        const m = new Map();
        m.set(a, 'a').set(b, 'b').set(Num.nan, 7).set(-0, 8);
        System.gc();
        const checks = [s.size, s.has(a), s.has({value: 1}), s.has(Num.nan),
            m.get(a), m.get(b), m.get(Num.nan), m.get(0)];
        const ordered = new Set(); ordered.add('a').add('b').add('a');
        ordered.delete('a'); ordered.add('a');
        const alias = ordered; System.gc();
        const result = [checks, alias.has('a'), alias.has('b'), [...ordered.keys()].length, [...ordered.values()].length];
    "#, "[[5, true, false, true, a, b, 7, 8], true, true, 2, 2]");
}

#[test]
fn napalm_collections_generic_constructors_annotations_and_nested_types() {
    run_napalm(r#"
        interface Item { value: number; }
        const m = new Map<string, Set<Item>>();
        const s: Set<Item> = new Set();
        s.add({value: 42}); m.set('items', s);
        const numbers: Map<string, number> = {one: 1};
        numbers.set('two', 2); numbers['three'] = 3;
        const mixed = new Set<string | number>(); mixed.add('a').add(2);
        const nested = new Map<string, Map<string, number>>(); nested.set('n', numbers);
        const list: Set<number>[] = [new Set<number>()]; list[0].add(9);
        System.gc();
        const result = [[...m.get('items')][0].value, numbers.get('two'), numbers['three'],
            [mixed.has('a'), mixed.has(2)], nested.get('n').size, [...list[0]]];
    "#, "[42, 2, 3, [true, true], 3, [9]]");
}

#[test]
fn napalm_collections_generic_contracts_enforce_all_alias_writes() {
    for source in [
        "const m = new Map<string, number>(); const a: any = m; a.set('bad', false);",
        "const m = new Map<string, number>(); m.set(1, 2);",
        "const m = new Map<string, number>(); const a: any = m; a['bad'] = false;",
        "const m = new Map<string, number>(); const a: any = m; a.bad = false;",
        "const s = new Set<number>(); const a: any = s; a.add('bad');",
        "const raw: any = {x: false}; const m: Map<string, number> = raw;",
        "const raw: any = new Set(); raw.add(false); const s: Set<number> = raw;",
        "interface I { value: number; } const s = new Set<I>(); const a: any = {value: 1}; s.add(a); a.value = false;",
        "interface I { value: number; } const m = new Map<I, number>(); const a: any = {value: 1}; m.set(a, 2); a.value = false;",
        "const m = new Map<string, Set<number>>(); const s = new Set(); m.set('s', s); s.add(false);",
        "const s = new Set<number>(); s.clear(); s.add(false);",
        "const m = new Map<string, number>(); m.clear(); m.set('x', false);",
        "const s = new Set<{value: number}>(); s.add({value: false});",
        "class Item {} class Other {} const s = new Set<Item>(); s.add(new Other());",
    ] {
        interface_error(source, false);
    }
}

#[test]
fn napalm_collections_reject_invalid_constructor_type_arity() {
    for source in ["new Map<string>();", "new Map<string, number, boolean>();", "new Set<string, number>();"] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid constructor: {source}");
    }
}

#[test]
fn napalm_collections_lookup_or_insert_bytecode() {
    use crate::napalm_bytecode::{BytecodeVerifier, Decoder, Opcode, Word};
    for constructor in ["Map", "Map<string, number>", "ConcurrentMap", "ConcurrentMap<string, number>", "Set", "Set<string>", "ConcurrentSet", "ConcurrentSet<string>"] {
        let args = if constructor.contains("Set") { "'key'" } else { "'key', 42" };
        let source = format!("const collection = new {constructor}(); const result = collection.set_if_not_exists({args});");
        let parsed = crate::napalm_front_main::parse("lookup.nap", &source);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let mut hir = Program::new();
        napalm_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(&mut hir, "lookup.nap", &source, "lookup", &parsed.module);
        assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
        let entry = lowered.lowered.unwrap().entry;
        let mut program = compile_program(&mut hir).unwrap();
        BytecodeVerifier::verify_program(&program).unwrap();
        let function = program.functions.iter_mut().find(|f| f.id == entry).unwrap();
        let pc = Decoder::new(&function.chunk.code).find_map(|i| {
            let i = i.unwrap(); (i.opcode == Opcode::MapLookupOrInsert).then_some(i.pc.0 as usize)
        }).unwrap_or_else(|| panic!("missing MapLookupOrInsert for {constructor}"));
        function.chunk.code[pc + 1] = Word(256);
        assert!(BytecodeVerifier::verify_program(&program).is_err());
    }
}

#[test]
fn napalm_collections_concurrent_api_and_generics() {
    run_napalm(r#"
        const m = new ConcurrentMap<string, number>();
        const s = new ConcurrentSet<number>();
        m.set('b', 2).set('a', 1); s.add(3).add(3);
        System.gc();
        const before = [m.get('a'), m.size, m.keys(), m.values(), m.entries(), s.has(3), s.size];
        m.delete('a'); s.delete(3); m.clear();
        const result = [before, m.size, s.size, typeof m, typeof s];
    "#, "[[1, 2, [a, b], [1, 2], [[a, 1], [b, 2]], true, 1], 0, 0, ConcurrentMap<String, Number>, ConcurrentSet<Number>]");
    for source in [
        "const m = new ConcurrentMap<string, number>(); m.set('bad', false);",
        "const s = new ConcurrentSet<number>(); s.add('bad');",
    ] { interface_error(source, false); }
}

#[test]
fn napalm_collections_lookup_or_insert_preserves_existing_values() {
    for constructor in ["Map", "ConcurrentMap"] {
        run_napalm(&format!(r#"
            const map = new {constructor}();
            const first = map.set_if_not_exists('answer', 42);
            const again = map.set_if_not_exists('answer', 99);
            map.set_if_not_exists('null', null);
            const presentNull = map.set_if_not_exists('null', 7);
            const object = {{value: 5}};
            const same = map.set_if_not_exists(object, object) == object;
            const retained = map.set_if_not_exists(object, {{value: 9}}) == object;
            map.set_if_not_exists(Num.nan, 11);
            map.set_if_not_exists(-0, 12);
            System.gc();
            const result = [first, again, presentNull, same, retained,
                map.set_if_not_exists(Num.nan, 20), map.set_if_not_exists(0, 30), map.size];
        "#), "[42, 42, null, true, true, 11, 12, 5]");
    }
    for constructor in ["Set", "ConcurrentSet"] {
        run_napalm(&format!(r#"
            const set = new {constructor}();
            const object = {{value: 3}};
            const first = set.set_if_not_exists(object) == object;
            const again = set.set_if_not_exists(object) == object;
            set.set_if_not_exists(null);
            const presentNull = set.set_if_not_exists(null);
            set.set_if_not_exists(Num.nan);
            set.set_if_not_exists(Num.nan);
            System.gc();
            const result = [first, again, presentNull, set.has(object), set.size];
        "#), "[true, true, null, true, 3]");
    }
    // Object literals exercise shaped storage and its ordered dictionary fallback.
    run_napalm(r#"
        const map: any = {a: null};
        const old = map.set_if_not_exists('a', 9);
        for (let i = 0; i < 8; i++) map.set_if_not_exists('k' + i, i);
        const result = [old, map.set_if_not_exists('k7', 99), map.k7];
    "#, "[null, 7, 7]");
}

#[test]
fn napalm_collections_lookup_or_insert_contracts() {
    for constructor in ["Map", "ConcurrentMap"] {
        run_napalm(&format!(r#"
            const map = new {constructor}<string, number>();
            const alias: any = map;
            alias.set_if_not_exists('key', 42);
            const result = alias.set_if_not_exists('key', false);
        "#), "42");
        interface_error(&format!(
            "const map = new {constructor}<string, number>(); const alias: any = map; alias.set_if_not_exists('bad', false);"
        ), false);
        interface_error(&format!(
            "const map = new {constructor}<string, number>(); const alias: any = map; alias.set_if_not_exists(1, 2);"
        ), false);
    }
    for constructor in ["Set", "ConcurrentSet"] {
        interface_error(&format!(
            "const set = new {constructor}<number>(); const alias: any = set; alias.set_if_not_exists('bad');"
        ), false);
    }
    interface_error(r#"
        interface Item { value: number; }
        const map = new Map<string, Item>();
        const item: any = {value: 1};
        map.set_if_not_exists('item', item);
        item.value = false;
    "#, false);
    interface_error(r#"
        function run() {
            const map = new Map<number, number>();
            for (let i = 0; i < 100; i++) {
                const value: any = i < 99 ? i : false;
                map.set_if_not_exists(i, value);
            }
        }
        run();
    "#, true);
}

#[test]
fn napalm_collections_lookup_or_insert_references_survive_gc() {
    run_napalm_checked(r#"
        function run() {
            const map = new Map();
            const set = new Set();
            let sum = 0;
            for (let i = 0; i < 512; i++) {
                const chosen = map.set_if_not_exists(i % 32, {value: i, items: [i]});
                set.set_if_not_exists(chosen);
                if (i % 64 == 0) System.gc();
                sum += chosen.value + chosen.items[0];
            }
            System.gc();
            return [sum, map.size, set.size, map.get(31).items[0]];
        }
        const result = run();
    "#, "[15872, 32, 32, 31]", true);
}

#[test]
fn napalm_collections_lookup_or_insert_is_atomic_across_workers() {
    run_napalm(r#"
        const shared map = new Map();
        const shared set = new Set();
        concurrent function fill(id) {
            let errors = 0;
            for (let key = 0; key < 64; key++) {
                const candidate = {id: id, log: []};
                const winner = map.set_if_not_exists(key, candidate);
                if (winner != map.get(key)) errors++;
                winner.log.push(id);
                if (set.set_if_not_exists(key) != key) errors++;
            }
            return errors;
        }
        const pending = [];
        for (let id = 0; id < 16; id++) pending.push(launch fill(id));
        let errors = 0;
        for (const task of pending) errors += await task;
        System.gc();
        let entries = 0;
        for (let key = 0; key < 64; key++) entries += map.get(key).log.length;
        const result = [errors, map.size, set.size, entries];
    "#, "[0, 64, 64, 1024]");
}

#[test]
fn napalm_collections_lookup_or_insert_in_native_tiers() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        function run() {
            const map = new Map<number, number>();
            const set = new Set<number>();
            let sum = 0;
            for (let i = 0; i < 4096; i++) {
                sum += map.set_if_not_exists(i % 32, i) + set.set_if_not_exists(i % 32);
            }
            return sum + map.size + set.size;
        }
        const result = run();
    "#);
    for native in [false, true] {
        for llvm in [false, true] {
            if llvm && (!native || !cfg!(feature = "llvm-jit")) { continue; }
            let mut vm = source_vm(&program, &manifest);
            vm.set_jit_options(native.then(|| TracingJitOptions {
                hot_loop_threshold: 2,
                hot_entry_threshold: 0,
                llvm_hot_threshold: if llvm { 1 } else { 0 },
                llvm_background_compile: false,
                native_loop_iteration_budget: 31,
                ..Default::default()
            }));
            vm.run_module(entry).unwrap();
            assert_eq!(source_result(&vm), 127040.0, "native={native}, llvm={llvm}");
            if native {
                let stats = vm.jit_stats().unwrap();
                assert!(stats.native_iterations > 3000, "{stats:?}");
                if llvm { assert!(stats.llvm_traces_installed > 0, "{stats:?}"); }
            }
        }
    }

    let (program, entry, manifest) = compile_typescript_module(r#"
        function lookup(map: Map<number, number>, key: number, value: number) {
            return map.set_if_not_exists(key, value);
        }
        const map = new Map<number, number>();
        let sum = 0;
        for (let i = 0; i < 4096; i++) sum += lookup(map, i % 32, i);
        const result = sum + map.size;
    "#);
    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2,
        hot_loop_threshold: u32::MAX,
        llvm_hot_threshold: 0,
        ..Default::default()
    }));
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), 63520.0);
    let stats = vm.whole_function_jit_stats().unwrap();
    assert!(stats.native_entries > 100, "{stats:?}");
}

#[test]
fn napalm_collections_shared_globals_and_local_captures() {
    run_napalm(r#"
        const shared m = new ConcurrentMap<number, number>();
        const shared s = new ConcurrentSet<number>();
        concurrent function fill(start) {
            for (let i = start; i < start + 100; i++) { m.set(i, i + 1); s.add(i); }
            System.gc();
            return 1;
        }
        const a = launch fill(0); const b = launch fill(100);
        await a; await b;
        async function local() {
            const shared localMap = new ConcurrentMap<string, number>();
            const worker = concurrent () => { localMap.set('answer', 42); return localMap.get('answer'); };
            const promise = launch worker();
            System.gc();
            const answer = await promise;
            return [answer, localMap.get('answer')];
        }
        const localResult = await local();
        const result = [m.size, s.size, m.get(199), s.has(150), localResult];
    "#, "[200, 200, 200, true, [42, 42]]");
}

#[test]
fn napalm_collections_shared_map_and_set_are_promoted() {
    run_napalm(r#"
        const shared m = new Map<number, number>();
        const shared s = new Set();
        const shared typed: Map<string, number> = new Map();
        const shared grouped: Set<string> = (new Set<string>());
        concurrent function fill(start) {
            for (let i = start; i < start + 50; i++) { m.set(i, i * 2); s.add(i); }
            typed.set('k' + start, start);
            grouped.add('g' + start);
            return 1;
        }
        const a = launch fill(0); const b = launch fill(50);
        await a; await b;
        const plain = new Map();
        const result = [typeof m, typeof s, typeof typed, typeof grouped, typeof plain,
            m.size, s.size, m.get(99), typed.get('k50'), grouped.has('g0')];
    "#, "[ConcurrentMap<Number, Number>, ConcurrentSet, ConcurrentMap<String, Number>, ConcurrentSet<String>, Map, 100, 100, 198, 50, true]");
    // Promoted shared collections keep the generic contracts.
    interface_error("const shared m = new Map<string, number>(); m.set('bad', false);", false);
}

#[test]
fn napalm_collections_shared_shapes_cross_worker_boundaries() {
    // Hidden classes are program-wide: shapes created by any VM stay valid in
    // every other VM that reaches the object through a shared collection.
    run_napalm(r#"
        const shared m = new Map();
        m.set('owner', {left: 40, right: 2});
        concurrent function put(i) {
            const owner = m.get('owner');
            m.set(i, {value: i, other: i * 2, extra: owner.left + owner.right});
            return 1;
        }
        const pending = [];
        for (let i = 0; i < 32; i++) pending.push(launch put(i));
        for (const p of pending) await p;
        System.gc();
        let sum = 0;
        for (let i = 0; i < 32; i++) { const o = m.get(i); sum += o.value + o.other + o.extra; }
        const fresh = {value: 1, other: 2, extra: 3};
        const result = [sum, m.get(31).extra, fresh.value + fresh.other + fresh.extra];
    "#, "[2832, 42, 6]");
}

#[test]
fn napalm_collections_shared_objects_are_locked_per_operation() {
    // Every push is one bytecode operation under the list's VmAdaptativeMutex:
    // concurrent workers can neither lose an element nor corrupt the List.
    run_napalm(r#"
        const shared m = new Map();
        m.set('list', []);
        m.set('counter', {hits: 0, log: []});
        concurrent function work(id) {
            const list = m.get('list');
            const counter = m.get('counter');
            for (let i = 0; i < 300; i++) {
                list.push(id * 1000 + i);
                counter.log.push(i);
            }
            return list.length > 0 ? 1 : 0;
        }
        const pending = [];
        for (let w = 0; w < 8; w++) pending.push(launch work(w));
        let started = 0;
        for (const p of pending) started += await p;
        const list = m.get('list');
        let sum = 0;
        for (const value of list) sum += value % 1000;
        const result = [started, list.length, m.get('counter').log.length, sum];
    "#, "[8, 2400, 2400, 358800]");
}

#[test]
fn napalm_collections_specialized_field_stores_publish_shared_values() {
    use crate::napalm_bytecode::{Decoder, Opcode};

    for (target, opcode) in [("shape", Opcode::SetFieldShape), ("object", Opcode::SetFieldClass)] {
        let source = format!(r#"
            interface Holder {{ items: number[]; }}
            class Box {{ items: number[] = []; }}
            const shared roots = new Map();
            const shape: Holder = {{items: []}};
            const object = new Box();
            roots.set('shape', shape);
            roots.set('object', object);
            concurrent function ready() {{ return 1; }}
            const pending = launch ready();
            await pending;
            const replacement: number[] = [];
            {target}.items = replacement;
        "#);
        let (program, entry, manifest) = compile_typescript_module(&source);
        assert!(program.functions.iter().any(|function| {
            Decoder::new(&function.chunk.code).any(|inst| inst.unwrap().opcode == opcode)
        }), "fixture must exercise {opcode:?}");
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(None);
        vm.run_module(entry).unwrap();
        // Inspect the last binding directly: reading it through another heap
        // opcode could itself mark the list shared and hide a missed barrier.
        assert!(vm.heap.is_shared(last_source_variable(&vm)),
            "{opcode:?} must publish values stored into an already shared object");
    }
}

#[test]
fn shared_trace_recording_and_steps_defer_before_mutation() {
    let mut asm = Assembler::new();
    asm.emit_ab(Opcode::ListAppend, 0, 1);
    asm.emit_a(Opcode::Return, 0);
    let mut chunk = asm.finish().unwrap();
    let mut references = ReferenceMap::default();
    references.set_general(Reg(0), true);
    references.set_general(Reg(1), true);
    chunk.safepoints.push(SafepointEntry {
        pc: BytecodePc(0), kind: SafepointKind::RuntimeMutation, references,
    });
    let program = simple_program(chunk, 2);
    let mut vm = Interpreter::new(&program).unwrap();
    vm.set_jit_options(Some(TracingJitOptions { hot_entry_threshold: 1, ..Default::default() }));
    let _roots = vm.gc_root_registration();
    vm.run_state = RunState::Running;
    vm.push_frame_direct(0, &[], None, None).unwrap();
    let list = vm.heap.alloc(HeapObject::List(Vec::new()));
    vm.add_root(list);
    vm.heap.shared_mode = true;
    vm.heap.mark_shared_deep(list);
    unsafe { vm.set_reg(0, list.word()); vm.set_reg(1, Value::TRUE.word()); }
    let location = vm.trace_current_location();

    let mut jit = vm.jit.take().unwrap();
    jit.preheat_entry(location);
    jit.on_entry(&mut vm).unwrap();
    vm.jit = Some(jit);
    assert_eq!(vm.trace_current_location(), location);
    assert!(matches!(vm.heap.get(list), Some(HeapObject::List(values)) if values.is_empty()));

    // Inlined/call recording also uses individual unfused steps. It must yield
    // without advancing the PC, so dispatch can acquire the operation's lock.
    let step = vm.trace_execute_unfused_step().unwrap();
    assert!(!step.running);
    assert_eq!(step.after, location);
    assert!(matches!(vm.run_state, RunState::Reschedule));
}

#[test]
fn shared_whole_function_helper_parks_until_the_lock_is_released() {
    use crate::napalm_bytecode::{IncomingSource, ParameterBinding, ArgReg};
    // Hand-built calls keep this test independent of HIR inlining/DCE.
    let mut caller = Assembler::new();
    caller.emit_call_direct(Reg(0), crate::napalm_ir::CallConv::Napalm, &[Reg(0)], FuncId(1)).unwrap();
    caller.emit_a(Opcode::Return, 0);
    let mut chunk = caller.finish().unwrap();
    let mut references = ReferenceMap::default();
    references.set_general(Reg(0), true);
    chunk.safepoints.push(SafepointEntry {
        pc: BytecodePc(0), kind: SafepointKind::Call, references: references.clone(),
    });
    let mut program = simple_program(chunk, 2);
    program.functions[0].kind = FunctionKind::Closure;
    program.functions[0].arity = 1;
    program.functions[0].frame.parameters.push(ParameterBinding {
        source: IncomingSource::Argument(ArgReg(0)), destination: Reg(0), kind: RegisterKind::Value,
    });
    let mut body = Assembler::new();
    body.emit_a(Opcode::LoadTrue, 1);
    body.emit_ab(Opcode::ListAppend, 0, 1);
    body.emit_a(Opcode::Return, 0);
    let mut callee = program.functions[0].clone();
    callee.id = FuncId(1);
    callee.chunk = body.finish().unwrap();
    callee.chunk.safepoints.push(SafepointEntry {
        pc: BytecodePc(1), kind: SafepointKind::RuntimeMutation, references,
    });
    program.push_function(callee).unwrap();
    program.rebuild_indexes().unwrap();
    let mut vm = Interpreter::new(&program).unwrap();
    let _roots = vm.gc_root_registration();
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2, hot_loop_threshold: u32::MAX, ..Default::default()
    }));
    let local = vm.heap.alloc(HeapObject::List(Vec::new()));
    vm.add_root(local);
    for _ in 0..128 { vm.run_function(FuncId(0), &[local]).unwrap(); }
    assert!(vm.whole_function_jit_stats().unwrap().native_entries > 0);
    let list = vm.heap.alloc(HeapObject::List(Vec::new()));
    vm.add_root(list);
    vm.heap.shared_mode = true;
    vm.heap.mark_shared_deep(list);
    vm.is_concurrent_worker = true;
    let handle = list.heap_handle().unwrap();
    assert!(vm_mutex::VmAdaptativeMutex::try_lock(&vm.heap.handle_table.entry(handle).sync));

    let native_entries = vm.whole_function_jit_stats().unwrap().native_entries;
    let result = vm.run_function(FuncId(0), &[list]).unwrap();
    assert!(vm.whole_function_jit_stats().unwrap().native_entries > native_entries);
    assert!(matches!(result, ExecutionResult::Suspended(_)));
    assert_eq!(vm.blocked_on_shared_lock(), Some(handle));
    assert!(matches!(vm.heap.get(list), Some(HeapObject::List(values)) if values.is_empty()));

    let woke = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let notification = woke.clone();
    assert!(vm.park_blocked_task(Box::new(move || {
        notification.store(true, Ordering::Release);
    })).is_ok());
    vm_mutex::VmAdaptativeMutex::unlock(handle, &vm.heap.handle_table.entry(handle).sync);
    assert!(woke.load(Ordering::Acquire));
    assert!(matches!(vm.resume_blocked_task().unwrap(), ExecutionResult::Returned(_)));
    assert!(matches!(vm.heap.get(list), Some(HeapObject::List(values)) if values == &[Value::TRUE]));
    assert!(vm.held_locks.is_empty());
    assert!(!vm.shared_op_active);
}

#[test]
fn napalm_collections_contended_shared_lock_parks_tasks() {
    // One long operation (spreading a large shared List) holds the lock for
    // longer than the 1 ms spin budget, so contending workers park their task
    // and are resumed when the lock is released.
    let before = crate::napalm_interpreter::vm_mutex_parked_tasks();
    run_napalm(r#"
        const shared m = new Map();
        const big = [];
        for (let i = 0; i < 200000; i++) big.push(i);
        m.set('big', big);
        concurrent function copier() {
            let total = 0;
            for (let i = 0; i < 6; i++) total += [...m.get('big')].length;
            return total;
        }
        concurrent function pusher(id) {
            const list = m.get('big');
            for (let i = 0; i < 50; i++) list.push(-1);
            return 50;
        }
        const pending = [launch copier()];
        for (let w = 0; w < 6; w++) pending.push(launch pusher(w));
        let pushed = 0;
        const copied = await pending[0];
        for (let i = 1; i < pending.length; i++) pushed += await pending[i];
        const result = [pushed, m.get('big').length, copied >= 6 * 200000];
    "#, "[300, 200300, true]");
    assert!(crate::napalm_interpreter::vm_mutex_parked_tasks() > before, "no task parked");
}

#[test]
fn napalm_collections_reject_invalid_shared() {
    for source in [
        "const shared x = 1;",
        "const shared x = {};",
        "const shared x = [];",
        "const shared x: any = 1;",
        "const shared x: Map<string, number> = {};",
        "const x = new ConcurrentMap(); concurrent function f() { return x.size; }",
        "function f() { const x = new ConcurrentMap(); const worker = concurrent () => x.size; }",
    ] {
        let parsed = crate::napalm_front_main::parse("shared.nap", source);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let mut hir = Program::new();
        napalm_builtin::install_hir_builtins(&mut hir);
        let output = lower_module(&mut hir, "shared.nap", source, "shared", &parsed.module);
        assert!(output.has_errors(), "accepted {source}");
    }
}

#[test]
fn napalm_collections_generic_class_elements_and_keys() {
    run_napalm(r#"
        class Item { value; constructor(n) { this.value = n; } }
        const item = new Item(42);
        const s = new Set<Item>(); s.add(item);
        const m = new Map<Item, Set<Item>>(); m.set(item, s);
        System.gc();
        const result = [m.get(item).has(item), [...s][0].value];
    "#, "[true, 42]");
}

#[test]
fn napalm_collections_intrinsics_native_loops_and_gc() {
    run_napalm_checked(r#"
        const m = new Map<number, number>(); const s = new Set<number>();
        let sum = 0;
        for (let i = 0; i < 100; i++) {
            m.set(i, i + 1); s.add(i); s.add(i);
            if (s.has(i) && m.has(i)) sum += m.get(i);
        }
        System.gc();
        for (let i = 0; i < 50; i++) { m.delete(i); s.delete(i); }
        let total = 0; for (const n of s) total += n;
        const result = [sum, m.size, s.size, total];
    "#, "[5050, 50, 50, 3725]", true);
}

#[test]
fn napalm_collections_generic_contract_survives_hot_alias_and_late_binding() {
    interface_error(r#"
        const raw: any = {x: 1};
        for (let i = 0; i < 100; i++) raw.x = i;
        const typed: Map<string, number> = raw;
        for (let i = 0; i < 100; i++) raw.x = i;
        raw.x = false;
    "#, true);
    interface_error(r#"
        const s = new Set<number>();
        function insert(value) { s.add(value); }
        for (let i = 0; i < 100; i++) insert(i);
        insert('bad');
    "#, true);
}

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

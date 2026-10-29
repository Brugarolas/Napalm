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
fn napalm_collections_keys_identity_order_and_gc() {
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
        const result = [checks, [...alias], [...ordered.keys()], [...ordered.values()]];
    "#, "[[5, true, false, true, a, b, 7, 8], [b, a], [b, a], [b, a]]");
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
            [...mixed], nested.get('n').size, [...list[0]]];
    "#, "[42, 2, 3, [a, 2], 3, [9]]");
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

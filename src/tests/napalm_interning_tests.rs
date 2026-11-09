fn interning_results(source: &str, expected: f64) {
    let (program, entry, manifest) = compile_typescript_module(&format!("{source}\nvar internTestResult = result;"));
    for jit in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        if !jit { vm.set_jit_options(None); }
        vm.run_module(entry).unwrap();
        assert_eq!(source_result(&vm), expected, "jit={jit}");
    }
}

#[test]
fn napalm_interning_mixes_interned_and_runtime_built_keys() {
    interning_results(r#"
        const o = {};
        const k = "na" + "me";              // runtime-built: not interned yet
        o[k] = 1;                            // becomes a shape key: interned
        let result = o.name + o["name"] + o[k];
        const p = { name: 5 };
        result += p["na" + "me"];           // unnamed probe vs interned shape key
        result += ("ab" + "c" == "abc") ? 10 : 0;
        result += ("abc" == "abc") ? 100 : 0;
        result += ("abd" == "abc") ? 0 : 1000;
        result += ("abc" != "ab" + "c") ? 0 : 10000;
        const d = {};
        for (let i = 0; i < 40; i++) d["k" + i] = i;   // dictionary mode
        result += d.k39 + d["k" + 39];
        let hits = 0;
        for (let i = 0; i < 500; i++) {
            const shaped = { alpha: i, beta: 2 };
            const key = i % 2 == 0 ? "alpha" : "be" + "ta";
            hits += shaped[key];
        }
        result += hits;
    "#, 3.0 + 5.0 + 10.0 + 100.0 + 1000.0 + 10000.0 + 78.0 + (0..500).step_by(2).sum::<i32>() as f64 + 500.0);
}

#[test]
fn napalm_interning_policy_names_script_text_and_shape_keys_only() {
    // Script text (literals and names) is interned when the VM starts.
    let (program, _entry, manifest) = compile_typescript_module("const greeting = \"interning_policy_literal\"; var result = 1;");
    let vm = source_vm(&program, &manifest);
    assert_eq!(vm.strings.len(), vm.string_names.len());
    for (&value, &name) in vm.strings.iter().zip(vm.string_names.iter()) {
        let Some(HeapObject::String(string)) = vm.heap.get(value) else { panic!("program string is not a String") };
        assert_eq!(string.name(), Some(name));
        assert_eq!(string.as_ref(), name.as_bytes());
        assert_eq!(string.hash, fast_hash_bytes(name.as_bytes()));
    }
    assert!(Name::lookup("interning_policy_literal").is_some());

    // Runtime-built strings are not interned on creation...
    let built = NapalmString::new(Arc::from(&b"interning_policy_runtime_value"[..]));
    assert_eq!(built.name(), None);
    assert_eq!(Name::lookup("interning_policy_runtime_value"), None);

    // ...but a key entering a hidden-class shape is.
    let shapes = MapShapeTable::default();
    let (child, slot) = shapes.transition(0, MapStringKey::new(Arc::from(&b"interning_policy_dynamic_key"[..]))).unwrap();
    let named = Name::lookup("interning_policy_dynamic_key").expect("shape key interned");
    assert_eq!(shapes.key_at(child, slot).name, Some(named));
    assert_eq!(shapes.slot(child, StrProbe::named(named)), Some(slot));
    assert_eq!(shapes.slot_hashed(child, fast_hash_bytes(b"interning_policy_dynamic_key"), b"interning_policy_dynamic_key"), Some(slot));
    assert_eq!(shapes.slot(child, StrProbe::named(Name::new("interning_policy_other"))), None);

    // Oversized keys never become shape keys and are never interned.
    let long = vec![b'q'; MAX_SHAPED_MAP_KEY_BYTES + 1];
    assert!(shapes.transition(0, MapStringKey::new(Arc::from(long.as_slice()))).is_none());
    assert_eq!(Name::lookup(std::str::from_utf8(&long).unwrap()), None);

    // Non-UTF-8 keys stay unnamed but are still found by bytes.
    let (raw_shape, raw_slot) = shapes.transition(0, MapStringKey::new(Arc::from(&[0xff, 0x01][..]))).unwrap();
    assert_eq!(shapes.key_at(raw_shape, raw_slot).name, None);
    assert_eq!(shapes.slot_hashed(raw_shape, fast_hash_bytes(&[0xff, 0x01]), &[0xff, 0x01]), Some(raw_slot));

    // Interned and plain keys are interchangeable in dictionaries.
    let plain = MapStringKey::new(Arc::from(&b"interning_policy_literal"[..]));
    let interned = MapStringKey::interned(Name::new("interning_policy_literal"));
    assert_eq!(plain, interned);
    assert_eq!(MapKey::String(plain.clone()), MapKey::String(interned.clone()));
    let mut table = hashbrown::HashMap::with_hasher(AesBuildHasher::new());
    table.insert(MapKey::String(plain), 7);
    assert_eq!(table.get(&MapKey::String(interned)), Some(&7));
}

/// Run `source` in the interpreter and with only the whole-function tier,
/// assert equal results, and return the WFJIT stats. Loop traces are off and
/// entry traces never install, so every hot call is the WFJIT's to take (with
/// default options, entry traces legitimately claim mid-sized methods first).
fn wfjit_this_run(source: &str) -> (f64, WholeFunctionJitStats) {
    let (program, entry, manifest) = compile_typescript_module(source);
    let mut baseline = source_vm(&program, &manifest);
    baseline.set_jit_options(None);
    baseline.run_module(entry).unwrap();
    let expected = source_result(&baseline);

    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(Some(TracingJitOptions {
        hot_entry_threshold: 2,
        hot_loop_threshold: u32::MAX,
        min_entry_trace_bytecodes: u32::MAX,
        llvm_hot_threshold: 0,
        ..Default::default()
    }));
    vm.run_module(entry).unwrap();
    assert_eq!(source_result(&vm), expected, "whole-function JIT changed the result");
    assert!(vm.native_shadow_top.is_null(), "native roots leaked after returning");
    (expected, vm.whole_function_jit_stats().unwrap())
}

#[test]
fn napalm_wfjit_compiles_methods_and_constructors_using_this() {
    let (result, stats) = wfjit_this_run(r#"
        class Vec2 {
            x: number;
            y: number;
            constructor(x: number, y: number) { this.x = x; this.y = y; }
            dot(o: Vec2): number { return this.x * o.x + this.y * o.y; }
            scaled(k: number): Vec2 { return new Vec2(this.x * k, this.y * k); }
            norm2(): number { return this.dot(this); }
        }
        let total = 0;
        const a = new Vec2(1, 2);
        for (let i = 0; i < 3000; i++) {
            const b = new Vec2(i, 1);
            total += a.dot(b) + b.scaled(2).x + b.norm2();
        }
        var result = total;
    "#);
    assert!(result > 0.0);
    assert!(stats.compiled_functions >= 4, "{stats:?}");
    assert_eq!(stats.rejected_functions, 0, "{stats:?}");
    assert!(stats.native_entries > 3000, "{stats:?}");
}

#[test]
fn napalm_wfjit_this_inheritance_chaining_and_mutation() {
    let (_, stats) = wfjit_this_run(r#"
        class Counter {
            count: number;
            step: number;
            constructor(step: number) { this.count = 0; this.step = step; }
            bump(): Counter { this.count = this.count + this.step; return this; }
            value(): number { return this.count; }
        }
        class Doubler extends Counter {
            constructor(step: number) { super(step * 2); }
            value(): number { return this.count + 1; }
        }
        let total = 0;
        for (let i = 0; i < 2000; i++) {
            const c = i % 2 == 0 ? new Counter(i) : new Doubler(i);
            total += c.bump().bump().value();
        }
        var result = total;
    "#);
    assert!(stats.compiled_functions > 0, "{stats:?}");
    assert_eq!(stats.rejected_functions, 0, "{stats:?}");
}

#[test]
fn napalm_wfjit_this_survives_bailouts_and_gc() {
    // `label` is compiled while `tag` holds numbers, then sees strings: the
    // native body must bail out at the exact PC and baseline must finish.
    let (_, stats) = wfjit_this_run(r#"
        class Item {
            tag: any;
            parts: any;
            constructor(tag: any) {
                this.tag = tag;
                this.parts = [tag, tag, { inner: tag }];
            }
            label(): any { return this.tag + this.tag; }
            static make(tag: any): Item { return new Item(tag); }
        }
        let total = 0;
        let text = "";
        for (let i = 0; i < 4000; i++) {
            const item = Item.make(i < 3000 ? i : "s");
            const label = item.label();
            if (i < 3000) { total += label + item.parts[2].inner; } else { text = label; }
            if (i % 500 == 0) System.gc();
        }
        var result = total + text.length;
    "#);
    assert!(stats.compiled_functions > 0, "{stats:?}");
    assert!(stats.native_entries > 0, "{stats:?}");
    assert!(stats.bailouts > 0, "receiver type change never exercised a bailout: {stats:?}");
}

#[test]
fn napalm_wfjit_this_callee_becomes_hot_inside_native_caller() {
    // drive reaches the threshold one call before add. Once drive is native,
    // calls to add must still count toward its own whole-function compilation.
    let (result, stats) = wfjit_this_run(r#"
        class Counter {
            value: number;
            constructor() { this.value = 0; }
            add(n: number): number { this.value += n; return this.value; }
        }
        function drive(counter: Counter, n: number): number {
            return counter.add(n) + 1;
        }
        const counter = new Counter();
        let total = 0;
        for (let i = 0; i < 400; i++) total += drive(counter, 1);
        var result = total + counter.value;
    "#);
    assert_eq!(result, 81000.0);
    assert!(stats.compiled_functions >= 2, "callee never compiled: {stats:?}");
    assert!(stats.native_direct_calls > 300, "callee stayed in baseline: {stats:?}");
}

#[test]
fn napalm_wfjit_recursive_this_preserves_mutation_at_native_depth_limit() {
    let (result, stats) = wfjit_this_run(r#"
        class Counter {
            value: number;
            constructor() { this.value = 0; }
            visit(n: number): number {
                this.value += 1;
                if (n <= 0) return this.value;
                return this.visit(n - 1);
            }
        }
        const counter = new Counter();
        for (let i = 0; i < 100; i++) counter.visit(2);
        var result = counter.visit(700) + counter.value;
    "#);
    assert_eq!(result, 2002.0);
    assert!(stats.compiled_functions > 0, "{stats:?}");
    assert!(stats.native_direct_calls > 500, "{stats:?}");
}

// Reactivity: computed properties are reactive-memoized getters validated by
// pulling (one global epoch, value-checked field edges, versioned computed
// edges); subscriptions exist only towards watchers.

fn reactive_vm<'a>(program: &'a BytecodeProgram, manifest: &BuiltinManifest, jit: bool) -> Interpreter<'a> {
    let mut vm = source_vm(program, manifest);
    vm.set_jit_options(jit.then(|| TracingJitOptions {
        hot_loop_threshold: 2,
        hot_entry_threshold: 2,
        ..Default::default()
    }));
    vm
}

#[test]
fn reactive_computed_follows_only_the_edges_of_its_last_run() {
    run_napalm(r#"
        let runs = 0;
        class Switch {
            flag = true; a = 1; b = 10;
            computed pick() { runs++; return this.flag ? this.a : this.b; }
        }
        const s = new Switch();
        const log = [s.pick, runs];
        s.b = 20;                 // not an edge of the last run
        log.push(s.pick); log.push(runs);
        s.a = 2;
        log.push(s.pick); log.push(runs);
        s.flag = false;           // the edge set changes: flag, b
        log.push(s.pick); log.push(runs);
        s.a = 3;                  // no longer an edge
        log.push(s.pick); log.push(runs);
        s.b = 30;
        log.push(s.pick); log.push(runs);
        const result = log;
    "#, "[1, 1, 1, 1, 2, 2, 20, 3, 20, 3, 30, 4]");
}

#[test]
fn reactive_computed_ignores_a_value_that_changes_back_and_equal_writes() {
    run_napalm(r#"
        let runs = 0;
        class Box {
            value = 1; other = 5;
            computed doubled() { runs++; return this.value * 2; }
            computed viaDoubled() { return this.doubled + this.other; }
        }
        const b = new Box();
        const log = [b.viaDoubled, runs];
        b.value = 7; b.value = 1;   // back to what the last run saw
        b.value = 1;                // equal write
        log.push(b.viaDoubled); log.push(b.doubled); log.push(runs);
        b.other = 6;                // only the outer computed runs again
        log.push(b.viaDoubled); log.push(runs);
        const result = log;
    "#, "[7, 1, 7, 2, 1, 8, 1]");
}

#[test]
fn reactive_computed_recovers_after_its_body_throws() {
    run_napalm(r#"
        let runs = 0;
        class Risky {
            fail = true; value = 4; extra = 100;
            computed inner() { runs++; if (this.fail) { throw "boom"; } return this.value; }
            // Catches the failure of a nested computed and keeps reading: the
            // read of `extra` must still become an edge of `outer`.
            computed outer() {
                let base = -1;
                try { base = this.inner; } catch (error) { base = 0; }
                return base + this.extra;
            }
        }
        const r = new Risky();
        const log = [];
        try { log.push(r.inner); } catch (error) { log.push(error); }
        try { log.push(r.inner); } catch (error) { log.push(error); }   // not "cyclic", and not cached
        log.push(runs);
        log.push(r.outer);
        r.extra = 200;
        log.push(r.outer);
        r.fail = false;
        log.push(r.inner); log.push(r.outer); log.push(r.outer); log.push(runs);
        const result = log;
    "#, "[boom, boom, 2, 100, 200, 4, 204, 204, 5]");
}

#[test]
fn reactive_computed_reports_cycles_and_survives_them() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Loop {
            seed = 1;
            computed a() { return this.b + 1; }
            computed b() { return this.seed > 0 ? this.a + 1 : 5; }
        }
        const loop = new Loop();
        let message = "";
        try { loop.a; } catch (error) { message = error; }
        loop.seed = 0;
        var result = loop.a;
    "#);
    for jit in [false, true] {
        let mut vm = reactive_vm(&program, &manifest, jit);
        vm.run_module(entry).unwrap_or_else(|error| panic!("jit={jit}: {error:?}"));
        assert_eq!(source_result(&vm), 6.0, "jit={jit}");
        assert!(vm.reactive_observers.iter().all(|observer| observer.flags & REACTIVE_EVALUATING == 0), "jit={jit}");
        assert!(!vm.reactive_tracking_active, "jit={jit}");
        assert!(vm.fiber(0).reactive_stack.is_empty(), "jit={jit}");
    }

    let (program, entry, manifest) = compile_typescript_module(
        "class Loop { computed a() { return this.a + 1; } } var result = new Loop().a;");
    let mut vm = source_vm(&program, &manifest);
    let error = vm.run_module(entry).unwrap_err();
    assert!(error.message.contains("cyclic computed property evaluation"), "{error:?}");
}

#[test]
fn reactive_computed_written_during_its_own_run_runs_again() {
    run_napalm(r#"
        let runs = 0;
        class Counter {
            count = 0;
            computed bump() { runs++; const seen = this.count; this.count = seen + 1; return seen; }
        }
        const c = new Counter();
        const result = [c.bump, c.bump, c.bump, runs, c.count];
    "#, "[0, 1, 2, 3, 3]");
}

#[test]
fn reactive_computed_state_is_per_instance_and_per_class_level() {
    run_napalm(r#"
        let baseRuns = 0;
        class Base {
            x = 1;
            computed total() { baseRuns++; return this.x * 10; }
            computed plain() { return this.x + 1; }
        }
        class Derived extends Base {
            y = 5;
            computed total() { return super.total + this.y; }
        }
        const a = new Base();
        const b = new Base();
        const d = new Derived();
        const log = [a.total, b.total, d.total, d.plain, baseRuns];
        b.x = 2;                    // only b's computed runs again
        log.push(a.total); log.push(b.total); log.push(d.total); log.push(baseRuns);
        d.y = 6;                    // the base level of d stays cached
        log.push(d.total); log.push(baseRuns);
        d.x = 3;
        log.push(d.total); log.push(d.plain); log.push(baseRuns);
        const result = [log, d.keys(), d.size, d['_reactive_computed_cache$total']];
    "#, "[[10, 10, 15, 2, 3, 10, 20, 15, 4, 16, 4, 36, 4, 5], [x, y], 2, null]");
}

#[test]
fn reactive_computed_tracks_static_fields_and_other_objects() {
    run_napalm(r#"
        class Config { static scale = 2; }
        class Wheel { radius = 1; }
        class Car {
            wheel = new Wheel();
            computed reach() { return this.wheel.radius * Config.scale; }
        }
        const car = new Car();
        const log = [car.reach];
        Config.scale = 3;
        log.push(car.reach);
        car.wheel.radius = 4;
        log.push(car.reach);
        car.wheel = new Wheel();
        log.push(car.reach);
        const result = log;
    "#, "[2, 3, 12, 3]");
}

#[test]
fn reactive_computed_survives_collections_and_recycled_handles() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Item { constructor(weight) { this.weight = weight; } }
        class Holder {
            item = new Item(1);
            computed weight() { return this.item.weight; }
            computed label() { return "w" + this.weight; }
        }
        const keep = new Holder();
        let total = 0;
        for (let round = 0; round < 40; round++) {
            // Short-lived owners: their observer records must be recycled.
            for (let i = 0; i < 25; i++) { const temp = new Holder(); total += temp.weight; }
            // A field edge that saw a heap reference: after a collection its
            // handle may name a different object.
            keep.item = new Item(round + 2);
            System.gc();
            total += keep.weight;
            if (keep.label != "w" + (round + 2)) { total = -1000000; }
        }
        var result = total;
    "#);
    let expected = (40 * 25) as f64 + (0..40).map(|round| (round + 2) as f64).sum::<f64>();
    for jit in [false, true] {
        let mut vm = reactive_vm(&program, &manifest, jit);
        vm.run_module(entry).unwrap_or_else(|error| panic!("jit={jit}: {error:?}"));
        assert_eq!(source_result(&vm), expected, "jit={jit}");
        vm.collect_garbage();
        let live = vm.reactive_observers.iter().filter(|observer| observer.kind != ReactiveObserverKind::Free).count();
        // `keep` owns two records; the 1000 temporaries reused a handful.
        assert_eq!(live, 2, "jit={jit}");
        assert!(vm.reactive_observers.len() < 200, "jit={jit}: {} records", vm.reactive_observers.len());
    }
}

#[test]
fn reactive_hot_loops_mix_cache_hits_writes_and_chains() {
    run_napalm(r#"
        class Graph {
            a = 1; b = 2; hot = 1;
            computed sum() { return this.a + this.b; }
            computed fan0() { return this.hot + this.a; }
            computed fan1() { return this.hot * 2 + this.b; }
            computed c0() { return this.hot + 1; }
            computed c1() { return this.c0 + 2; }
            computed c2() { return this.c1 + 3; }
            computed root() { return this.c2 + this.sum + this.fan0 + this.fan1; }
        }
        function run() {
            const g = new Graph();
            let hits = 0;
            for (let i = 0; i < 300; i++) { hits += g.sum; hits += g.sum; }
            let fan = 0;
            for (let i = 0; i < 300; i++) {
                g.hot = g.hot + 1; g.hot = g.hot + 1;
                fan += g.fan0; fan += g.fan1;
            }
            let chain = 0;
            for (let i = 0; i < 300; i++) { g.a = g.a + 1; chain += g.root; }
            // A read that hits while the loop is traced, then goes stale now
            // and then: the native hit must notice and recompute.
            let mixed = 0;
            for (let i = 0; i < 300; i++) {
                mixed += g.sum;
                if (i % 50 == 49) { g.b = g.b + 1; }
            }
            return [hits, fan, chain, g.hot, g.a, mixed];
        }
        const result = run();
    "#, "[1800, 272700, 815100, 601, 301, 91650]");
}

#[test]
fn reactive_watchers_follow_computed_edges_and_unsubscribe() {
    let (program, entry, manifest) = compile_typescript_module(r#"
        class Model {
            x = 1; y = 10; useY = false;
            runs = 0; seen = 0;
            computed doubled() { return this.x * 2; }
            computed total() { return this.doubled + 1; }
            watch observe() {
                this.runs = this.runs + 1;
                this.seen = this.useY ? this.y : this.total;
            }
        }
        const model = new Model();
        const log = [];
        async function step(action) {
            action();
            // The flush is one microtask and the watcher task another turn.
            await Timer.microtask(() => null);
            await Timer.microtask(() => null);
            log.push(model.runs);
            log.push(model.seen);
        }
        async function main() {
            await step(function () {});                      // first run: [1, 3]
            await step(function () { model.x = 2; model.x = 3; });   // coalesced: [2, 7]
            await step(function () { model.y = 11; });        // y is not an edge: [2, 7]
            await step(function () { model.useY = true; });   // switches to y: [3, 11]
            await step(function () { model.x = 4; });         // x no longer an edge: [3, 11]
            await step(function () { model.y = 12; });        // [4, 12]
        }
        main();
        // The same List the steps keep filling after the module body returns.
        var result = log;
    "#);
    for jit in [false, true] {
        let mut vm = reactive_vm(&program, &manifest, jit);
        vm.run_module(entry).unwrap_or_else(|error| panic!("jit={jit}: {error:?}"));
        let log = last_source_variable(&vm);
        assert_eq!(
            HostContext { heap: &mut vm.heap }.format_value(log),
            "[1, 3, 2, 7, 2, 7, 3, 11, 3, 11, 4, 12]",
            "jit={jit}",
        );
        // The watcher now depends on fields only: the computed chain it left
        // behind stopped subscribing to anything.
        let computed = vm.reactive_observers.iter()
            .filter(|observer| observer.kind == ReactiveObserverKind::Computed)
            .collect::<Vec<_>>();
        assert_eq!(computed.len(), 2, "jit={jit}");
        for observer in computed {
            assert!(observer.subscribers.is_empty(), "jit={jit}: {observer:?}");
            assert_eq!(observer.flags & REACTIVE_SUBSCRIBED, 0, "jit={jit}: {observer:?}");
        }
        // What the last run read: runs, useY and y.
        assert_eq!(vm.reactive_subscribed_nodes, 3, "jit={jit}");
    }
}

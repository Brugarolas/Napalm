// Production frontend -> HIR -> optimizer -> bytecode -> interpreter/JIT.
#[test]
fn napalm_destructuring_assignments_parameters_and_captures() {
    run_napalm(r#"
        function swap(pair) { let [a, b] = pair; [a, b] = [b, a]; return [a, b]; }
        const arrow = ({x: [first, ...rest]}, [other = 3]) => [first, rest, other];
        class Box {
            value;
            constructor({value}) { this.value = value; }
            add([a, b]) { return this.value + a + b; }
        }
        let a = 0; let rest = null; let x = 0;
        function change() { [a, ...rest] = [7, 8, 9]; ({missing: x = 10} = {}); }
        change();
        const object = {};
        const returned = ([object.value, x] = [11, 12]);
        let sum = 0;
        for (const [left, {right = 2}] of [[1, {}], [3, {right: 4}]]) sum += left + right;
        const result = [swap([1, 2]), arrow({x: [4, 5, 6]}, []), new Box({value: 1}).add([2, 3]), a, rest, object.value, x, returned, sum];
    "#, "[[2, 1], [4, [5, 6], 3], 6, 7, [8, 9], 11, 12, [11, 12], 10]");
}

#[test]
fn napalm_destructuring_rejects_invalid_patterns_and_const_writes() {
    for source in [
        "const [...rest, last] = [1, 2];",
        "const [...rest,] = [];",
        "let rest; [...rest,] = [];",
        "let rest; ({...rest,} = {});",
        "const {...rest, x} = {};",
        "const [a = ] = [];",
        "const x = 1; [x] = [2];",
        "const x = 1; const f = () => { ({x} = {x: 2}); };",
        "let x; [x] += [1];",
        "let x; [0] = [1];",
        "let x; const object = {x = 1};",
        "function f([x], {x}) {}",
    ] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid pattern: {source}");
    }
}

#[test]
fn napalm_decorators_apply_bottom_up_and_run_initializers() {
    run_napalm(
        r#"
        const events = []
        function mark(tag) {
            return function (value, context) {
                events.add(tag + ":" + context.kind + ":" + context.name)
                context.addInitializer(() => events.add("init:" + tag + ":" + context.name))
                return function (name) { return value(name) + "[" + tag + "]" }
            }
        }
        @mark("top")
        @mark("bottom")
        function greet(name) { return "hi " + name }
        const result = [greet("napalm"), events]
    "#,
        "[hi napalm[bottom][top], [bottom:function:greet, top:function:greet, init:bottom:greet, init:top:greet]]",
    );
}

#[test]
fn napalm_decorators_support_class_replacement_and_null_passthrough() {
    run_napalm(
        r#"
        const events = []
        class Wrapped {
            value;
            constructor() { this.value = "Sample:wrapped"; }
        }
        function keep(value, context) {
            events.add("keep:" + context.kind + ":" + context.name)
            context.addInitializer(() => events.add("init:" + context.name))
            return null
        }
        function replace(value, context) {
            return Wrapped
        }
        @keep
        @replace
        class Sample {
            value;
            constructor() { this.value = "original"; }
        }
        const instance = new Sample()
        const result = [instance.value, events]
    "#,
        "[Sample:wrapped, [keep:class:Sample, init:Sample]]",
    );
}

#[test]
fn napalm_decorators_report_runtime_contract_errors() {
    run_napalm(
        r#"
        let messages = []
        function badResult(value, context) { return 1 }
        let callback = () => null
        function delayed(value, context) {
            callback = context.addInitializer
            return null
        }
        try {
            @badResult
            function nope() { return 1 }
        } catch (e) {
            messages.add(e)
        }
        @delayed
        function ok() { return 2 }
        try {
            callback(() => 1)
        } catch (e) {
            messages.add(e)
        }
        const result = messages.count
    "#,
        "2",
    );
}

#[test]
fn napalm_decorators_example_covers_members_and_gc() {
    run_napalm_checked(include_str!("..\\..\\scripts\\decorators.napalm"), "decorators-ok", true);
}

#[test]
fn native_decorators_executable_example() {
    run_napalm(include_str!("..\\..\\scripts\\native_decorators.napalm"), "native-decorators-ok");
}

#[test]
fn native_decorators_cache_null_errors_and_full_argument_keys() {
    run_napalm(r#"
        let calls = 0;
        @memoize
        function empty(value) { calls++; return null; }
        empty(1); empty(1);
        @memoize
        function pair(a, b) { calls++; return a + b; }
        const values = [pair(1, 2), pair(2, 1), pair(1, 2), pair(0, -0), pair(-0, 0)];
        let failures = 0;
        @memoize
        function retry(value) {
            failures++;
            if (failures === 1) throw "retry";
            return value;
        }
        try { retry(5); } catch (error) {}
        const retried = [retry(5), retry(5)];
        const result = [calls, values, failures, retried];
    "#, "[4, [3, 3, 3, 0, 0], 2, [5, 5]]");
}

#[test]
fn native_decorators_cache_managed_arguments_results_and_recursive_calls() {
    run_napalm(r#"
        let calls = 0;
        @memoize
        function fib(value) {
            calls++;
            if (value < 2) return value;
            return fib(value - 1) + fib(value - 2);
        }
        const answer = fib(20);
        let objects = 0;
        @memoize
        function make(key) { objects++; return {key, values: [objects]}; }
        let key: any = {name: "alive"};
        const cached = make(key);
        System.gc();
        const again = make(key);
        again.values.add(2);
        key = null;
        System.gc();
        const result = [answer, calls, objects, cached.key.name, cached.values];
    "#, "[6765, 21, 1, alive, [1, 2]]");
}

#[test]
fn native_decorators_validate_durations_options_and_targets() {
    run_napalm(r#"
        let errors = 0;
        try { @delay(-1) function bad() {} } catch (error) { errors++; }
        try { @delay(0 / 0) function bad() {} } catch (error) { errors++; }
        try { @debounce("slow") function bad() {} } catch (error) { errors++; }
        try { @debounce(1, {leading: 1}) function bad() {} } catch (error) { errors++; }
        try { @debounce(1, {trailing: "yes"}) function bad() {} } catch (error) { errors++; }
        try { @debounce(1, {maxWait: -1}) function bad() {} } catch (error) { errors++; }
        try { @throttle(1, {leading: false, trailing: false}) function bad() {} } catch (error) { errors++; }
        try { @memoizeExpiring(-1) function bad() {} } catch (error) { errors++; }
        try { @memoize class Bad {} } catch (error) { errors++; }
        try { class Bad { @once value = 1; } } catch (error) { errors++; }
        const result = errors;
    "#, "10");
}

#[test]
fn native_decorators_debounce_throttle_edges_and_async_adoption() {
    run_napalm(r#"
        let calls = 0;
        @debounce(15, {leading: true, trailing: true})
        function single(value) { calls++; return value; }
        const first = single(1);
        await first;
        await Timer.delay(50);
        const singleCalls = calls;
        const next = [single(2), single(3), single(4)];
        const both = await Promise.all(next);
        await Timer.delay(50);
        @throttle(15, {leading: false})
        async function trailing(value) {
            await Timer.delay(1);
            return value * 2;
        }
        const pending = [trailing(2), trailing(4)];
        const adopted = await Promise.all(pending);
        let errors = 0;
        @debounce(1)
        function fail() { throw "failure"; }
        try { await fail(); } catch (error) { errors++; }
        const result = [singleCalls, calls, both, adopted, errors];
    "#, "[1, 3, [2, 4, 4], [8, 8], 1]");
}

#[test]
fn native_decorators_expiry_is_per_key_and_does_not_refresh_on_hit() {
    run_napalm(r#"
        let calls = 0;
        @memoizeExpiring(200)
        function value(key) { calls++; return calls; }
        const first = value("a");
        await Timer.delay(100);
        const second = value("a");
        const other = value("b");
        await Timer.delay(130);
        const expired = value("a");
        const stillLive = value("b");
        const result = [first, second, other, expired, stillLive, calls];
    "#, "[1, 1, 2, 3, 2, 3]");
}

#[test]
fn native_decorators_preserve_arity_stacking_and_rejected_promises() {
    run_napalm(r#"
        let calls = 0;
        @memoize
        function wide(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) {
            calls++;
            return a + p;
        }
        const values = [wide(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),
            wide(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)];
        let delayedCalls = 0;
        @once
        @delay(1)
        function initialized(value) {
            delayedCalls++;
            return value;
        }
        const delayed = await Promise.all([initialized(4), initialized(9)]);
        let rejectedCalls = 0;
        @memoize
        async function rejected(value) {
            rejectedCalls++;
            throw value;
        }
        let errors = 0;
        try { await rejected("error"); } catch (error) { errors++; }
        try { await rejected("error"); } catch (error) { errors++; }
        const result = [wide.arity, calls, values, delayed, delayedCalls, rejectedCalls, errors];
    "#, "[16, 1, [17, 17], [4, 4], 1, 1, 2]");
}

#[test]
fn native_decorators_scheduled_arguments_remain_rooted_without_the_wrapper() {
    run_napalm(r#"
        let pending: any = null;
        {
            @delay(10)
            function temporary(object) {
                System.gc();
                return object.value;
            }
            pending = temporary({value: ["retained", 42]});
        }
        System.gc();
        const result = await pending;
    "#, "[retained, 42]");
}

#[test]
fn native_decorators_zero_ttl_null_once_and_primitive_key_types() {
    run_napalm(r#"
        let nullCalls = 0;
        @once
        function empty() { nullCalls++; return null; }
        empty(); empty();
        let ttlCalls = 0;
        @memoizeExpiring(0)
        function fresh() { ttlCalls++; return ttlCalls; }
        const freshValues = [fresh(), fresh()];
        let keyCalls = 0;
        @memoize
        function identify(value) { keyCalls++; return keyCalls; }
        const values = [identify(1), identify("1"), identify(true), identify(null),
            identify(1), identify("1"), identify(true), identify(null)];
        const result = [nullCalls, ttlCalls, freshValues, keyCalls, values];
    "#, "[1, 2, [1, 2], 4, [1, 2, 3, 4, 1, 2, 3, 4]]");
}

#[test]
fn napalm_decorators_static_accessors_overloads_and_super() {
    run_napalm(r#"
        function method(value, context) {
            return (receiver, amount) => value(receiver, amount) + 10;
        }
        function getter(value, context) { return (receiver) => value(receiver) + 1; }
        function setter(value, context) { return (receiver, amount) => value(receiver, amount * 2); }
        function keep(value, context) { return null; }
        class Base {
            @method
            compute(amount) { return amount + 1; }
        }
        class Child extends Base {
            @method
            compute(amount) { return super.compute(amount) + 1; }
        }
        class Statics {
            static amount = 3;
            @getter
            static get value() { return Statics.amount; }
            @setter
            static set value(amount) { Statics.amount = amount; }
            @keep
            empty;
            @keep
            static missing;
            @keep
            compute() { return 7; }
            @method
            compute(amount) { return amount; }
        }
        const first = Statics.value;
        Statics.value = 5;
        const object = new Statics();
        const result = [new Child().compute(1), first, Statics.value,
            object.empty, Statics.missing, object.compute(), object.compute(2)];
    "#, "[23, 4, 11, null, null, 7, 12]");
}

#[test]
fn napalm_decorators_member_errors_and_initializer_lifetime() {
    run_napalm(r#"
        function bad(value, context) { return 42; }
        function badInitializer(value, context) { context.addInitializer(() => null); }
        let delayed = (callback) => null;
        function capture(value, context) {
            delayed = context.addInitializer;
            return null;
        }
        let errors = 0;
        try { class Bad { @bad method() {} } } catch (e) { errors++; }
        try { class Bad { @bad field = 1; } } catch (e) { errors++; }
        try { class Bad { @badInitializer get field() { return 1; } } } catch (e) { errors++; }
        class Valid { @capture field = 2; }
        try { delayed((receiver) => null); } catch (e) { errors++; }
        const first = new Valid();
        const second = new Valid();
        const result = [errors, first.field, second.field];
    "#, "[4, 2, 2]");
}

#[test]
fn napalm_decorators_local_factories_and_async_methods() {
    run_napalm(r#"
        let object: any = null;
        {
            const offset = 10;
            const namespace = {
                decorate: (value, context) => (receiver, amount) => value(receiver, amount) + offset
            };
            class Local {
                @namespace.decorate
                compute(amount) { return amount + 1; }
            }
            object = new Local();
        }
        function asyncDecorator(value, context) {
            return async (receiver, amount) => (await value(receiver, amount)) + 1;
        }
        class AsyncExample {
            amount = 5;
            @asyncDecorator
            async compute(amount) {
                await Timer.delay(0);
                return this.amount + amount;
            }
        }
        const result = [object.compute(2), await new AsyncExample().compute(3)];
    "#, "[13, 9]");
}

#[test]
fn napalm_decorators_throwing_decorator_closes_initializer_registration() {
    run_napalm(r#"
        let delayed = (callback) => null;
        let runs = 0;
        function fail(value, context) {
            delayed = context.addInitializer;
            context.addInitializer((receiver) => runs++);
            throw "failed decorator";
        }
        let errors = 0;
        try { class Invalid { @fail field; } } catch (e) { errors++; }
        try { delayed((receiver) => runs++); } catch (e) { errors++; }
        const result = [errors, runs];
    "#, "[2, 0]");
}

#[test]
fn napalm_decorators_exports_preserve_public_names() {
    let parsed = crate::napalm_front_main::parse("exports.napalm", r#"
        function keep(value, context) { return null; }
        @keep
        export default function () { return 1; }
        @keep
        export class Example {
            @keep
            value = 1;
        }
    "#);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let exports = parsed.module.exports.unwrap();
    let public: Vec<_> = exports.iter().map(|(_, name)| name.value.as_str()).collect();
    assert_eq!(public, ["default", "Example"]);
}

#[test]
fn napalm_decorators_reject_unsupported_member_targets() {
    for source in [
        "class Bad { @dec constructor() {} }",
        "class Bad { @dec static {} }",
        "class Bad { @dec abstract method(); }",
        "class Bad { @dec declare field; }",
        "class Bad { @dec foreign method(); }",
        "class Bad { @dec computed value() { return 1; } }",
        "class Bad { @dec watch value() {} }",
        "class Bad { @dec operator +(value) { return value; } }",
    ] {
        let parsed = crate::napalm_front_main::parse("invalid-decorator.napalm", source);
        assert!(parsed.has_errors(), "accepted unsupported decorator target: {source}");
    }
}

#[test]
fn napalm_spread_and_destructuring_use_dedicated_bytecode() {
    use crate::napalm_bytecode::{Decoder, Opcode};
    let source = "const [a, ...rest] = [1, ...[2, 3]]; const {x, ...other} = {...{x: 4, y: 5}}; const f = (x) => x; const result = f(...[a]);";
    let parsed = crate::napalm_front_main::parse("ops.nap", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let mut hir = Program::new();
    napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "ops.nap", source, "ops", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let mut program = compile_program(&mut hir).unwrap();
    let opcodes: crate::napalm_collections::HashSet<_> = program.functions.iter()
        .flat_map(|function| Decoder::new(&function.chunk.code).map(|op| op.unwrap().opcode)).collect();
    for opcode in [Opcode::ListSpread, Opcode::MapSpread, Opcode::DestructureCheck,
        Opcode::DestructureGet, Opcode::DestructureRest, Opcode::InvokeSpread] {
        assert!(opcodes.contains(&opcode), "missing {opcode:?}");
    }
    assert!(crate::napalm_bytecode::BytecodeVerifier::verify_program(&program).is_ok());
    // Corrupt a dynamic dispatch operand: the verifier must reject it before VM preparation.
    let function = program.functions.iter_mut().find(|f| Decoder::new(&f.chunk.code)
        .any(|op| op.unwrap().opcode == Opcode::InvokeSpread)).unwrap();
    let pc = Decoder::new(&function.chunk.code).find_map(|op| {
        let op = op.unwrap(); (op.opcode == Opcode::InvokeSpread).then_some(op.pc)
    }).unwrap();
    function.chunk.code[pc.word_index() + 1].0 = u32::MAX;
    assert!(crate::napalm_bytecode::BytecodeVerifier::verify_program(&program).is_err());
}

#[test]
fn napalm_destructuring_nested_defaults_rest_and_computed_keys() {
    run_napalm(r#"
        let calls = 0;
        function source() { calls++; return [{x: [1, 2, 3]}, 4, 5]; }
        const [{x: [first, , ...tail]}, second, ...rest] = source();
        let keys = 0;
        function key() { keys++; return 'x'; }
        const {[key()]: renamed, missing = ++calls, ...others} = {x: 7, y: 8};
        const [absent = 9, falseValue = 10] = [null, false];
        rest.add(6);
        const result = [first, tail, second, rest, renamed, missing, others.y, calls, keys, absent, falseValue];
    "#, "[1, [3], 4, [5, 6], 7, 2, 8, 2, 1, 9, false]");
}

#[test]
fn napalm_spread_collections_and_iterators() {
    run_napalm(r#"
        const original = [1, 2];
        const copied = [0, ...original, ...[], ...new Range(3, 5), ...'ab'];
        const base = {x: 1, y: [2]};
        const merged = {x: 0, ...base, ...null, x: 3};
        merged.y.add(4);
        original[0] = 9;
        const result = [copied, merged.x, base.x, base.y];
    "#, "[[0, 1, 2, 3, 4, a, b], 3, 1, [2, 4]]");
}

#[test]
fn napalm_spread_calls_methods_constructors_and_super() {
    run_napalm(r#"
        const sum = (a, b, c) => a + b + c;
        class Base { value; constructor(a, b) { this.value = a + b; } add(a, b) { return this.value + a + b; } }
        class Derived extends Base {
            constructor(xs) { super(...xs); }
            add(xs) { return super.add(...xs); }
        }
        const item = new Derived(...[[2, 3]]);
        const object = {sum};
        let skipped = 0;
        const missing = null;
        const ignored = missing?.(...[++skipped]);
        const result = [sum(1, ...[2, 3]), object.sum(...[4, 5, 6]), item.add([7, 8]), skipped, ignored];
    "#, "[6, 15, 20, 0, null]");
}

#[test]
fn napalm_destructuring_and_spread_report_catchable_errors() {
    run_napalm(r#"
        let errors = 0;
        try { const [] = null; } catch (e) { errors++; }
        try { const {} = null; } catch (e) { errors++; }
        try { const value = [...1]; } catch (e) { errors++; }
        try { const value = {...1}; } catch (e) { errors++; }
        try { const f = () => 1; f(...[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]); } catch (e) { errors++; }
        const result = errors;
    "#, "5");
}

fn run_napalm(source: &str, expected: &str) {
    run_napalm_checked(source, expected, false);
}

fn run_napalm_checked(source: &str, expected: &str, require_native: bool) {
    let parsed = crate::napalm_front_main::parse("test.nap", source);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let output = lower_module(&mut hir, "test.nap", source, "test", &parsed.module);
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
    let entry = output.lowered.unwrap().entry;
    let program = compile_program(&mut hir).unwrap();
    for enabled in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(enabled.then(|| TracingJitOptions {
            hot_loop_threshold: 2,
            native_loop_iteration_budget: if require_native { 7 } else { 10_000 },
            ..Default::default()
        }));
        let result = vm.run_module(entry).unwrap_or_else(|e| {
            panic!(
                "{e:?}\nMethods: {:?}",
                program
                    .metadata
                    .methods
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (i, program.metadata.string(s.name), s.kind))
                    .collect::<Vec<_>>()
            )
        });
        let value = last_source_variable(&vm);
        if enabled && require_native {
            assert!(vm.jit_stats().unwrap().native_entries > 0, "new collection opcodes never reached native code: {:?}", vm.jit_stats());
            assert!(vm.jit_stats().unwrap().native_budget_exits > 0, "collection loops must honor the native watchdog");
        }
        let context = HostContext { heap: &mut vm.heap };
        assert_eq!(
            context.format_value(value),
            expected,
            "jit={enabled}, {result:?}"
        );
    }
}

#[test]
fn napalm_spread_and_destructuring_in_native_loops_with_gc() {
    run_napalm_checked(r#"
        function add(a, b) { return a + b; }
        let total = 0;
        for (let i = 0; i < 100; i++) {
            const [first, ...tail] = [i, ...[1, 2]];
            const {x, ...rest} = {...{x: first, y: tail}};
            total += add(...[x, rest.y[0]]);
        }
        let source = {};
        for (let i = 0; i < 40; i++) source[`key${i}`] = [i];
        const copied = {...source};
        const {key0, ...rest} = copied;
        source = null;
        System.gc();
        const [a, ...tail] = [rest.key39, key0, rest.key1];
        System.gc();
        const result = [total, a, tail];
    "#, "[5050, [39], [[0], [1]]]", true);
}

#[test]
fn napalm_spread_preserves_custom_iteration_and_evaluation_order() {
    run_napalm(r#"
        let log = '';
        class Sequence {
            iterate(state) { log += 'i'; return state == null ? 0 : (state < 1 ? state + 1 : false); }
            iteratorValue(state) { log += 'v'; System.gc(); return state + 1; }
        }
        function part(n) { log += n; return n; }
        const f = (a, b, c, d) => [a, b, c, d];
        const items = f(part('a'), ...new Sequence(), part('b'));
        const result = [items, log];
    "#, "[[a, 1, 2, b], aivivib]");
}

#[test]
fn napalm_destructuring_defaults_merge_local_state_and_only_export_bindings() {
    run_napalm("function f() { let n = 0; const [a = ++n, b = ++n] = []; return [a, b, n]; } const result = f();", "[1, 2, 2]");
    run_napalm("function f() { let n = 0; let a; let b; [a = ++n, b = ++n] = []; return [a, b, n]; } const result = f();", "[1, 2, 2]");
    run_napalm(r#"
        function values() {
            let n = 0;
            const [a = ++n, b = ++n] = [];
            let c; let d;
            [c = ++n, d = ++n] = [];
            let sum = 0;
            for (let i = 0; i < 4; i++) [sum] = [sum + i];
            return [a, b, c, d, n, sum];
        }
        const result = values();
    "#, "[1, 2, 3, 4, 4, 6]");
    let parsed = crate::napalm_front_main::parse("exports.nap",
        "export const {a, nested: [b], ...rest} = {a: 1, nested: [2], c: 3};");
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let names: Vec<_> = parsed.module.exports.unwrap().into_iter().map(|(_, public)| public.value).collect();
    assert_eq!(names, ["a", "b", "rest"]);
}

#[test]
fn napalm_functions_types_closures_and_templates() {
    run_napalm(
        r#"
        type Numbers = Array<number>;
        interface Pair { left: number; right: number; }
        function twice(value: number): number { return value * 2; }
        const add: (a: number, b: number) => number = (a: number, b: number): number => a + b;
        let captured = 1;
        const next = () => ++captured;
        const values: Numbers = [next(), next(), add(2, 3), twice(4)];
        let result = `${values}: ${`nested ${twice(5)}`}: 50%`;
    "#,
        "[2, 3, 5, 8]: nested 10: 50%",
    );
}

#[test]
fn napalm_classes_constructor_inheritance_fields_accessors() {
    run_napalm(
        r#"
        const offset = 2;
        class Parent {
            value: number = 1;
            constructor(value: number) { this.value = value; }
            add(n: number): number { return this.value + n + offset; }
        }
        class Child extends Parent {
            extra = 3;
            static total: number = 10;
            constructor(value: number) { super(value); Child.total++; }
            get doubled(): number { return this.value * 2; }
            set doubled(n: number) { this.value = n / 2; }
            add(n: number) { return super.add(n) + this.extra; }
            callback() { return () => this.value; }
        }
        const item = new Child(4);
        item.doubled = 12;
        let result = [item.add(1), item.doubled, Child.total, item.callback()(), item instanceof Parent];
    "#,
        "[12, 12, 11, 6, true]",
    );
}

#[test]
fn napalm_for_continue_break_updates_and_capture() {
    run_napalm(
        r#"
        let sum = 0;
        for (let i = 0; i < 20; i++) {
            if (i === 2) continue;
            if (i === 10) break;
            sum += i;
        }
        for (const value of [1, 2, 3]) sum += value;
        let i = 1;
        let result = [sum, i++, ++i, i];
    "#,
        "[49, 1, 3, 3]",
    );
}

#[test]
fn napalm_assignment_evaluates_location_once_before_rhs() {
    run_napalm(
        r#"
        let order = '';
        const items = [10];
        function container() { order += 'a'; return items; }
        function index() { order += 'b'; return 0; }
        function value() { order += 'c'; return 2; }
        container()[index()] += value();
        const old = container()[index()]++;
        container()[index()] = value();
        const result = [order, old, items[0]];
    "#,
        "[abcababc, 12, 2]",
    );
}

#[test]
fn napalm_objects_and_hot_loop() {
    run_napalm(
        r#"
        const name = 'Napalm';
        const object = {name, score: 0, double: (x: number) => x * 2};
        for (let i = 0; i < 100; i++) object.score += object.double(i);
        let result = [object.name, object.score, object['score']];
    "#,
        "[Napalm, 9900, 9900]",
    );
}

#[test]
fn napalm_async_exceptions_and_cleanup() {
    run_napalm(
        r#"
        let log = '';
        async function compute(value: number): Promise<number> {
            defer { log += 'd'; }
            try { await Timer.delay(0); throw 'x'; }
            catch (error) { log += error; }
            finally { log += 'f'; }
            return value * 2;
        }
        const value = await compute(21);
        const task = async (n: number) => await Promise.resolve(n + 1);
        let result = [value, await task(1), log];
    "#,
        "[42, 2, xfd]",
    );
}

#[test]
fn napalm_const_shadowing_is_lexical() {
    run_napalm(
        "const x = 1; { let x = 2; x++; } const list = [1]; list[0]++; let result = [x, list];",
        "[1, [2]]",
    );
    for source in [
        "const x;",
        "const x = 1; x = 2;",
        "const x = 1; const f = () => x++;",
        "let x = 1; let x = 2;",
        "export { missing };",
        "let x: = 1;",
        "class Old { construct new() {} }",
        "for (x in [1]) {}",
        "let x = {|n| n};",
    ] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid source: {source}");
    }
}

#[test]
fn napalm_newlines_and_block_return_semantics() {
    run_napalm(
        r#"
        function empty() { 42; }
        function line() { return
            42;
        }
        const f = (x: number) => (
            x +
            2
        );
        const result = [empty(), line(), f(3), '50%', "a\u{1f600}"];
    "#,
        "[null, null, 5, 50%, a😀]",
    );
}

#[test]
fn napalm_nested_recursion_and_ranges() {
    run_napalm(
        r#"
        function outer() {
            function factorial(n: number): number { return n < 2 ? 1 : n * factorial(n - 1); }
            return factorial(6);
        }
        let sum = 0;
        for (const i of new Range(1, 5, true)) sum += i;
        const result = [outer(), sum, [0,1,2,3][new Range(1,3)]];
    "#,
        "[720, 15, [1, 2]]",
    );
}

#[test]
fn napalm_field_initializers_are_per_instance_and_function_bodies_are_statements() {
    run_napalm(
        r#"
        class Bag { items = []; callback = (n: number) => n + 1; }
        const first = new Bag();
        const second = new Bag();
        first.items.add(1);
        const result = [first.items, second.items, (first.callback)(41)];
    "#,
        "[[1], [], 42]",
    );
}

#[test]
fn napalm_const_imports_and_type_syntax_rejections() {
    for source in [
        "import { value } from './dep'; value++;",
        "const value = 1; function change() { value += 1; }",
        "let value = 1; export { value, value };",
        "const f = (a: number, a: number) => a;",
        "class A { constructor() {} constructor() {} }",
        "let x = 1; x + 2 = 3;",
        "let x = `unterminated ${1`",
    ] {
        let parsed = crate::napalm_front_main::parse("bad.nap", source);
        assert!(parsed.has_errors(), "accepted invalid source: {source}");
    }
    let source = "type Matrix = Array<Array<number>>; interface A { f: (n: number) => number; } const matrix: Matrix = [[1]];";
    assert!(!crate::napalm_front_main::parse("types.nap", source).has_errors());
}

#[test]
fn napalm_object_property_stores_survive_gc_and_shape_changes() {
    run_napalm(
        r#"
        const object = {};
        System.gc();
        object.items = [1, 2, 3];
        object.text = 'retained';
        for (let i = 0; i < 80; i++) object[`key${i}`] = [i];
        System.gc();
        for (let i = 0; i < 30; i++) {
            object.items = [i, i + 1];
            System.gc();
        }
        const result = [object.items, object.text, object['key79']];
    "#,
        "[[29, 30], retained, [79]]",
    );
}


#[test]
fn napalm_interfaces_valid_aliases_nested_lists_and_gc() {
    run_napalm(r#"
        interface Child { value: number; }
        interface Base { name: string; }
        interface Item extends Base { child: Child; values: number[]; label?: string; readonly id: number; }
        const child = {value: 1};
        const values = [2, 3];
        const raw: any = {name: 'item', child, values, id: 7, extra: true};
        let item: Item = raw;
        child.value = 4;
        values.add(5);
        item.label = 'ok';
        raw.name = 'changed';
        System.gc();
        function local() {
            const p: Child = {value: 9};
            p.value++;
            return p.value + p['value'];
        }
        const copy = {...item};
        copy.name = 123;
        const result = [item.name, item.child.value, item.values, item.label, copy.name, local()];
    "#, "[changed, 4, [2, 3, 5], ok, 123, 20]");
}

fn interface_error(source: &str, require_native: bool) {
    let parsed = crate::napalm_front_main::parse("interface.nap", source);
    assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
    let mut hir = Program::new();
    let manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "interface.nap", source, "interface", &parsed.module);
    if lowered.has_errors() {
        // A statically proven bad write is already rejected before any catch can
        // run. Tests explicitly requiring native execution must still reach the VM.
        assert!(!require_native, "native test rejected during lowering: {:?}", lowered.diagnostics);
        assert!(lowered.diagnostics.iter().all(|diagnostic| matches!(diagnostic.code, "NAPALM-T001" | "NAPALM-T003")),
            "unexpected compilation failure: {:?}", lowered.diagnostics);
        return;
    }
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let entry = lowered.lowered.unwrap().entry;
    let program = compile_program(&mut hir).unwrap();
    for enabled in [false, true] {
        let mut vm = source_vm(&program, &manifest);
        vm.set_jit_options(enabled.then(|| TracingJitOptions {
            hot_loop_threshold: 2, ..Default::default()
        }));
        let error = vm.run_module(entry).expect_err("interface violation did not terminate execution");
        assert!(error.message.starts_with(interfaces::TYPE_ERROR), "jit={enabled}: {error:?}");
        if require_native && enabled {
            assert!(vm.jit_stats().unwrap().native_entries > 0, "test did not reach native code");
        }
    }
}

#[test]
fn napalm_interfaces_reject_all_mutation_routes_and_cannot_be_caught() {
    let prefix = r#"
        interface Child { value: number; }
        interface Item { name: string; child: Child; values: number[]; optional?: boolean; readonly id: number; }
        const nested: any = {value: 1};
        const list: any = [1, 2];
        const raw: any = {name: 'ok', child: nested, values: list, id: 3};
        let item: Item = raw;
    "#;
    for mutation in [
        "item.name = 1;", "raw['name'] = false;", "raw[42] = true; raw.name = false;",
        "nested.value = 'bad';", "item.child = {value: 'bad'};",
        "list[0] = 'bad';", "list.add('bad');", "list.insert(0, 'bad');",
        "item.values = [1, 'bad'];", "raw.remove('name');", "raw.clear();",
        "item.optional = 'bad';", "item.id = 4;",
        "function bad(x) { x.name = 1; } bad(raw);",
        "item = {name: 'ok'};",
        "const replacement = {value: 5}; item.child = replacement; replacement.value = 'bad';",
        "const replacement = [5]; item.values = replacement; replacement.add('bad');",
        "const fiber = new Fiber(() => { raw.name = 1; }); fiber.try();",
        "async function bad() { raw.name = 1; } await bad();",
    ] {
        interface_error(&format!("{prefix} try {{ {mutation} }} catch (e) {{ }} const result = 1;"), false);
    }
}

#[test]
fn napalm_interfaces_reject_initial_shape_and_dynamic_rebinding() {
    for source in [
        "interface P { x: number; } const p: P = {};",
        "interface P { x: number; } const p: P = {x: 'bad'};",
        "interface P { x: number; } function f() { return 'bad'; } const p: P = f();",
        "interface P { x: number; } function f() { let p: P = {x: 1}; const g = () => { p = {x: 'bad'}; }; g(); } f();",
    ] { interface_error(source, false); }
}

#[test]
fn napalm_interfaces_jit_preexisting_alias_and_late_failure() {
    // Compile writes while untyped, then attach a contract through another alias.
    interface_error(r#"
        interface P { x: number; }
        function write(p, value) { for (let i = 0; i < 100; i++) p.x = value; }
        const raw: any = {x: 1};
        write(raw, 'warmup');
        raw.x = 1;
        const typed: P = raw;
        write(raw, 'bad');
    "#, true);
    interface_error(r#"
        interface P { xs: number[]; }
        function write(xs, value) { for (let i = 0; i < 100; i++) xs[0] = value; }
        const xs: any = [1];
        write(xs, 'warmup');
        xs[0] = 1;
        const typed: P = {xs};
        write(xs, 'bad');
    "#, true);
}

#[test]
fn napalm_interfaces_ir_retains_contract_and_typed_reads() {
    let source = "interface P { x: number; } function f() { const p: P = {x: 1}; return p.x; } const result = f();";
    let parsed = crate::napalm_front_main::parse("typed.nap", source);
    let mut hir = Program::new();
    napalm_builtin::install_hir_builtins(&mut hir);
    let lowered = lower_module(&mut hir, "typed.nap", source, "typed", &parsed.module);
    assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);
    let mut program = compile_program(&mut hir).unwrap();
    use crate::napalm_bytecode::{Decoder, Opcode};
    assert!(program.functions.iter().any(|f| Decoder::new(&f.chunk.code)
        .any(|op| op.unwrap().opcode == Opcode::BindInterface)));
    assert!(hir.functions.iter().any(|f| f.nodes.iter().any(|node| {
        matches!(node.kind, crate::napalm_ir::Inst::MapGet { .. })
            && f.value(node.results[0]).unwrap().ty == Type::Num
    })));
    // A corrupted descriptor must fail verification before VM preparation.
    let function = program.functions.iter_mut().find(|f| Decoder::new(&f.chunk.code)
        .any(|op| op.unwrap().opcode == Opcode::BindInterface)).unwrap();
    let pc = Decoder::new(&function.chunk.code).find_map(|op| {
        let op = op.unwrap(); (op.opcode == Opcode::BindInterface).then_some(op.pc)
    }).unwrap();
    function.chunk.code[pc.word_index() + 1].0 = u32::MAX;
    assert!(crate::napalm_bytecode::BytecodeVerifier::verify_program(&program).is_err());
}


#[test]
fn napalm_interfaces_unions_optional_removal_and_multiple_contracts() {
    run_napalm(r#"
        interface P { x: number | string; optional?: boolean; }
        let p: P | null = {x: 1};
        p.x = 'ok'; p.optional = true; p.remove('optional');
        const rows: P[] = [{x: 2}, {x: 'three'}];
        rows.add({x: 4});
        interface Q { child: { value: number; }; rows: { name: string; }[]; }
        const q: Q = {child: {value: 7}, rows: [{name: 'row'}]};
        const result = [p.x, p.optional, rows[2].x, q.child.value, q.rows[0].name];
    "#, "[ok, null, 4, 7, row]");
    interface_error("interface A { x: number; } interface B { x: number | string; } const raw = {x: 1}; const a: A = raw; const b: B = raw; b.x = 'bad';", false);
    interface_error("interface P { x: number; } const rows: P[] = [{x: 1}]; rows.add({x: 'bad'});", false);
    interface_error("interface Q { child: { value: number; }; } const q: Q = {child: {value: 1}}; q.child.value = 'bad';", false);
}

#[test]
fn napalm_interfaces_typed_mutations_remain_jittable() {
    run_napalm_checked(r#"
        interface P { x: number; values: number[]; }
        const p: P = {x: 0, values: [0]};
        for (let i = 0; i < 200; i++) { p.x++; p.values[0] = i; }
        System.gc();
        const result = [p.x, p.values[0]];
    "#, "[200, 199]", true);
}

#[test]
fn napalm_interfaces_jit_does_not_erase_a_failing_overwritten_store() {
    interface_error(r#"
        interface P { x: number; }
        function write(p) { for (let i = 0; i < 100; i++) { p.x = 'bad'; p.x = 1; } }
        const raw: any = {x: 1}; write(raw);
        const typed: P = raw; write(raw);
    "#, true);
}

#[test]
fn napalm_interfaces_preserve_map_builtin_precedence() {
    run_napalm(r#"
        interface P { count: string; }
        function f() { const p: P = {count: 'field'}; return [p.count, p['count']]; }
        const p: P = {count: 'global'};
        const result = [f(), p.count, p['count']];
    "#, "[[1, field], 1, global]");
}

#[test]
fn napalm_interfaces_reject_unresolved_and_conflicting_schemas() {
    for source in [
        "interface P { x: Unknown; }",
        "interface P { next: P; }",
        "interface P extends Missing { x: number; }",
        "interface A { x: number; } interface B extends A { x: string; }",
        "interface P { x: {nested: Unknown;}; }",
    ] {
        let parsed = crate::napalm_front_main::parse("invalid.nap", source);
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);
        let mut hir = Program::new();
        napalm_builtin::install_hir_builtins(&mut hir);
        let lowered = lower_module(&mut hir, "invalid.nap", source, "invalid", &parsed.module);
        assert!(lowered.has_errors(), "accepted {source}");
    }
}

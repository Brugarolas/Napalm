// reactive_benchmark_light.mjs
//
// Lightweight reactive benchmark for Node.js.
//
// Run with:
//
//   node reactive_benchmark_light.mjs
//
// This implements a deliberately small reactive system with:
//
//   - signals
//   - lazy computed values
//   - cached computed values
//   - automatic dependency tracking
//   - transitive invalidation
//
// It deliberately does NOT implement:
//
//   - batching
//   - debounce
//   - throttle
//   - effects
//   - watchers
//   - async scheduling
//   - graph compilation
//   - static dependency analysis
//   - version-based invalidation
//
// The benchmark mirrors the Napalm native-reactivity benchmark.

import { performance } from "node:perf_hooks";


const SCALE = 1;

const CACHE_ROUNDS = 2_000_000 * SCALE;
const FANOUT_ROUNDS = 250_000 * SCALE;
const CHAIN_ROUNDS = 400_000 * SCALE;
const GRAPH_ROUNDS = 250_000 * SCALE;


// ============================================================================
// Minimal reactive runtime
// ============================================================================

let activeComputation = null;


class Signal {
    constructor(value) {
        this._value = value;
        this._subscribers = new Set();
    }

    get value() {
        if (activeComputation !== null) {
            activeComputation._trackDependency(this);
        }

        return this._value;
    }

    set value(next) {
        if (Object.is(this._value, next)) {
            return;
        }

        this._value = next;

        // Invalidations are synchronous, but recomputation is lazy.
        //
        // A subscriber that is already dirty will immediately return from
        // _invalidate(), preventing repeated propagation through the graph.
        for (const subscriber of this._subscribers) {
            subscriber._invalidate();
        }
    }

    _subscribe(computation) {
        this._subscribers.add(computation);
    }

    _unsubscribe(computation) {
        this._subscribers.delete(computation);
    }
}


class Computed {
    constructor(compute) {
        if (typeof compute !== "function") {
            throw new TypeError("computed() expects a function");
        }

        this._compute = compute;

        this._cachedValue = undefined;
        this._dirty = true;
        this._computing = false;

        this._dependencies = new Set();
        this._subscribers = new Set();
    }

    get value() {
        // A computed can itself be a dependency of another computed.
        if (activeComputation !== null) {
            activeComputation._trackDependency(this);
        }

        if (this._dirty) {
            this._recompute();
        }

        return this._cachedValue;
    }

    _trackDependency(dependency) {
        // Reading the same dependency several times during one computation
        // should create only one dependency edge.
        if (this._dependencies.has(dependency)) {
            return;
        }

        this._dependencies.add(dependency);
        dependency._subscribe(this);
    }

    _subscribe(computation) {
        this._subscribers.add(computation);
    }

    _unsubscribe(computation) {
        this._subscribers.delete(computation);
    }

    _invalidate() {
        if (this._dirty) {
            return;
        }

        this._dirty = true;

        // Propagate dirtiness, but do not recompute anything yet.
        for (const subscriber of this._subscribers) {
            subscriber._invalidate();
        }
    }

    _recompute() {
        if (this._computing) {
            throw new Error(
                "Reactive dependency cycle detected while evaluating computed value",
            );
        }

        // Dependencies are discovered dynamically, so old edges have to be
        // removed before evaluating the computation again.
        for (const dependency of this._dependencies) {
            dependency._unsubscribe(this);
        }

        this._dependencies.clear();

        const previousComputation = activeComputation;

        this._computing = true;
        activeComputation = this;

        try {
            this._cachedValue = this._compute();
            this._dirty = false;
        } finally {
            activeComputation = previousComputation;
            this._computing = false;
        }
    }
}


function signal(value) {
    return new Signal(value);
}


function computed(compute) {
    return new Computed(compute);
}


// ============================================================================
// Benchmark graph
// ============================================================================

class ReactiveBenchmark {
    constructor() {
        // Shared reactive state.
        this.a = signal(1);
        this.b = signal(2);
        this.c = signal(3);
        this.d = signal(4);

        // Dedicated state for the fan-out and chain benchmarks.
        this.hot = signal(1);
        this.chainSeed = signal(1);


        // ====================================================================
        // Cached computed values
        // ====================================================================

        this.cached0 = computed(
            () => this.a.value + this.b.value,
        );

        this.cached1 = computed(
            () => this.b.value + this.c.value,
        );

        this.cached2 = computed(
            () => this.c.value + this.d.value,
        );

        this.cached3 = computed(
            () => this.a.value + this.d.value,
        );


        // ====================================================================
        // Wide fan-out
        // ====================================================================

        this.fan0 = computed(
            () => this.hot.value + this.a.value + 1,
        );

        this.fan1 = computed(
            () => this.hot.value * 2 + this.a.value + 2,
        );

        this.fan2 = computed(
            () => this.hot.value * 3 + this.a.value + 3,
        );

        this.fan3 = computed(
            () => this.hot.value * 4 + this.a.value + 4,
        );

        this.fan4 = computed(
            () => this.hot.value * 5 + this.b.value + 5,
        );

        this.fan5 = computed(
            () => this.hot.value * 6 + this.b.value + 6,
        );

        this.fan6 = computed(
            () => this.hot.value * 7 + this.b.value + 7,
        );

        this.fan7 = computed(
            () => this.hot.value * 8 + this.b.value + 8,
        );

        this.fan8 = computed(
            () => this.hot.value * 9 + this.c.value + 9,
        );

        this.fan9 = computed(
            () => this.hot.value * 10 + this.c.value + 10,
        );

        this.fan10 = computed(
            () => this.hot.value * 11 + this.c.value + 11,
        );

        this.fan11 = computed(
            () => this.hot.value * 12 + this.c.value + 12,
        );

        this.fan12 = computed(
            () => this.hot.value * 13 + this.d.value + 13,
        );

        this.fan13 = computed(
            () => this.hot.value * 14 + this.d.value + 14,
        );

        this.fan14 = computed(
            () => this.hot.value * 15 + this.d.value + 15,
        );

        this.fan15 = computed(
            () => this.hot.value * 16 + this.d.value + 16,
        );


        // ====================================================================
        // Deep computed chain
        // ====================================================================

        this.chain0 = computed(
            () => this.chainSeed.value + 1,
        );

        this.chain1 = computed(
            () => this.chain0.value + 2,
        );

        this.chain2 = computed(
            () => this.chain1.value + 3,
        );

        this.chain3 = computed(
            () => this.chain2.value + 4,
        );

        this.chain4 = computed(
            () => this.chain3.value + 5,
        );

        this.chain5 = computed(
            () => this.chain4.value + 6,
        );

        this.chain6 = computed(
            () => this.chain5.value + 7,
        );

        this.chain7 = computed(
            () => this.chain6.value + 8,
        );

        this.chain8 = computed(
            () => this.chain7.value + 9,
        );

        this.chain9 = computed(
            () => this.chain8.value + 10,
        );

        this.chain10 = computed(
            () => this.chain9.value + 11,
        );

        this.chain11 = computed(
            () => this.chain10.value + 12,
        );

        this.chain12 = computed(
            () => this.chain11.value + 13,
        );

        this.chain13 = computed(
            () => this.chain12.value + 14,
        );

        this.chain14 = computed(
            () => this.chain13.value + 15,
        );

        this.chain15 = computed(
            () => this.chain14.value + 16,
        );


        // ====================================================================
        // Multi-level reactive graph
        // ====================================================================

        this.leaf0 = computed(
            () => this.a.value + this.b.value + 1,
        );

        this.leaf1 = computed(
            () => this.a.value + this.c.value + 2,
        );

        this.leaf2 = computed(
            () => this.a.value + this.d.value + 3,
        );

        this.leaf3 = computed(
            () => this.a.value + this.b.value + this.c.value + 4,
        );

        this.leaf4 = computed(
            () => this.b.value + this.c.value + 5,
        );

        this.leaf5 = computed(
            () => this.b.value + this.d.value + 6,
        );

        this.leaf6 = computed(
            () => this.b.value + this.a.value + 7,
        );

        this.leaf7 = computed(
            () => this.b.value + this.c.value + this.d.value + 8,
        );

        this.leaf8 = computed(
            () => this.c.value + this.d.value + 9,
        );

        this.leaf9 = computed(
            () => this.c.value + this.a.value + 10,
        );

        this.leaf10 = computed(
            () => this.c.value + this.b.value + 11,
        );

        this.leaf11 = computed(
            () => this.c.value + this.d.value + this.a.value + 12,
        );

        this.leaf12 = computed(
            () => this.d.value + this.a.value + 13,
        );

        this.leaf13 = computed(
            () => this.d.value + this.b.value + 14,
        );

        this.leaf14 = computed(
            () => this.d.value + this.c.value + 15,
        );

        this.leaf15 = computed(
            () => this.d.value + this.a.value + this.b.value + 16,
        );


        this.middle0 = computed(
            () => this.leaf0.value + this.leaf1.value,
        );

        this.middle1 = computed(
            () => this.leaf2.value + this.leaf3.value,
        );

        this.middle2 = computed(
            () => this.leaf4.value + this.leaf5.value,
        );

        this.middle3 = computed(
            () => this.leaf6.value + this.leaf7.value,
        );

        this.middle4 = computed(
            () => this.leaf8.value + this.leaf9.value,
        );

        this.middle5 = computed(
            () => this.leaf10.value + this.leaf11.value,
        );

        this.middle6 = computed(
            () => this.leaf12.value + this.leaf13.value,
        );

        this.middle7 = computed(
            () => this.leaf14.value + this.leaf15.value,
        );


        this.upper0 = computed(
            () => this.middle0.value + this.middle1.value,
        );

        this.upper1 = computed(
            () => this.middle2.value + this.middle3.value,
        );

        this.upper2 = computed(
            () => this.middle4.value + this.middle5.value,
        );

        this.upper3 = computed(
            () => this.middle6.value + this.middle7.value,
        );


        this.graphRoot = computed(
            () =>
                this.upper0.value
                + this.upper1.value
                + this.upper2.value
                + this.upper3.value,
        );
    }
}


// ============================================================================
// Benchmark utilities
// ============================================================================

function printResult(name, elapsedMs, checksum) {
    console.log(name);

    console.log("Time (ms):");
    console.log(elapsedMs);

    console.log("Checksum:");
    console.log(checksum);

    console.log("");
}


// ============================================================================
// Benchmark
// ============================================================================

function runBenchmark() {
    console.log("========================================");
    console.log("Node lightweight reactivity benchmark");
    console.log("========================================");

    const benchmark = new ReactiveBenchmark();

    let totalChecksum = 0;

    const totalStart = performance.now();


    // ========================================================================
    // 1. Cached computed reads
    // ========================================================================

    console.log("");
    console.log("----------------------------------------");
    console.log("1. COMPUTED CACHE HITS");
    console.log("----------------------------------------");

    let cacheChecksum = 0;

    const cacheStart = performance.now();

    for (
        let i = 0;
        i < CACHE_ROUNDS;
        i++
    ) {
        cacheChecksum += benchmark.cached0.value;
        cacheChecksum += benchmark.cached1.value;
        cacheChecksum += benchmark.cached2.value;
        cacheChecksum += benchmark.cached3.value;
    }

    const cacheEnd = performance.now();
    const cacheMs = cacheEnd - cacheStart;

    totalChecksum += cacheChecksum;

    printResult(
        "Computed cache hits",
        cacheMs,
        cacheChecksum,
    );


    // ========================================================================
    // 2. Wide invalidation fan-out
    // ========================================================================

    console.log("----------------------------------------");
    console.log("2. WIDE INVALIDATION FAN-OUT");
    console.log("----------------------------------------");

    let fanoutChecksum = 0;

    const fanoutStart = performance.now();

    for (
        let i = 0;
        i < FANOUT_ROUNDS;
        i++
    ) {
        benchmark.hot.value =
            benchmark.hot.value + 1;

        benchmark.hot.value =
            benchmark.hot.value + 1;

        benchmark.hot.value =
            benchmark.hot.value + 1;

        benchmark.hot.value =
            benchmark.hot.value + 1;

        fanoutChecksum += benchmark.fan0.value;
        fanoutChecksum += benchmark.fan1.value;
        fanoutChecksum += benchmark.fan2.value;
        fanoutChecksum += benchmark.fan3.value;

        fanoutChecksum += benchmark.fan4.value;
        fanoutChecksum += benchmark.fan5.value;
        fanoutChecksum += benchmark.fan6.value;
        fanoutChecksum += benchmark.fan7.value;

        fanoutChecksum += benchmark.fan8.value;
        fanoutChecksum += benchmark.fan9.value;
        fanoutChecksum += benchmark.fan10.value;
        fanoutChecksum += benchmark.fan11.value;

        fanoutChecksum += benchmark.fan12.value;
        fanoutChecksum += benchmark.fan13.value;
        fanoutChecksum += benchmark.fan14.value;
        fanoutChecksum += benchmark.fan15.value;
    }

    const fanoutEnd = performance.now();
    const fanoutMs =
        fanoutEnd - fanoutStart;

    totalChecksum += fanoutChecksum;

    printResult(
        "Wide invalidation fan-out",
        fanoutMs,
        fanoutChecksum,
    );


    // ========================================================================
    // 3. Deep computed dependency chain
    // ========================================================================

    console.log("----------------------------------------");
    console.log("3. DEEP COMPUTED CHAIN");
    console.log("----------------------------------------");

    let chainChecksum = 0;

    const chainStart = performance.now();

    for (
        let i = 0;
        i < CHAIN_ROUNDS;
        i++
    ) {
        benchmark.chainSeed.value =
            benchmark.chainSeed.value + 1;

        chainChecksum +=
            benchmark.chain15.value;
    }

    const chainEnd = performance.now();
    const chainMs =
        chainEnd - chainStart;

    totalChecksum += chainChecksum;

    printResult(
        "Deep computed chain",
        chainMs,
        chainChecksum,
    );


    // ========================================================================
    // 4. Multi-level reactive graph
    // ========================================================================

    console.log("----------------------------------------");
    console.log("4. MULTI-LEVEL REACTIVE GRAPH");
    console.log("----------------------------------------");

    let graphChecksum = 0;

    const graphStart = performance.now();

    for (
        let i = 0;
        i < GRAPH_ROUNDS;
        i++
    ) {
        benchmark.a.value =
            benchmark.a.value + 1;

        benchmark.b.value =
            benchmark.b.value + 2;

        benchmark.c.value =
            benchmark.c.value + 3;

        benchmark.d.value =
            benchmark.d.value + 4;

        graphChecksum +=
            benchmark.graphRoot.value;
    }

    const graphEnd = performance.now();
    const graphMs =
        graphEnd - graphStart;

    totalChecksum += graphChecksum;


    printResult(
        "Multi-level reactive graph",
        graphMs,
        graphChecksum,
    );


    // ========================================================================
    // Results
    // ========================================================================

    const totalEnd = performance.now();
    const totalMs =
        totalEnd - totalStart;

    const totalReactiveMs =
        cacheMs
        + fanoutMs
        + chainMs
        + graphMs;


    console.log("========================================");
    console.log("RESULTS");
    console.log("========================================");

    console.log("Cache-hit time (ms):");
    console.log(cacheMs);

    console.log("Fan-out time (ms):");
    console.log(fanoutMs);

    console.log("Deep-chain time (ms):");
    console.log(chainMs);

    console.log("Reactive-graph time (ms):");
    console.log(graphMs);

    console.log("");

    console.log("Measured reactive work (ms):");
    console.log(totalReactiveMs);

    console.log("Whole benchmark time (ms):");
    console.log(totalMs);

    console.log("Final checksum:");
    console.log(totalChecksum);

    console.log("========================================");
}


runBenchmark();

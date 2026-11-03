// concurrency_benchmark.mjs
//
// CPU-bound concurrency benchmark for Node.js.
//
// Run with:
//   node concurrency_benchmark.mjs

import {
    Worker,
    isMainThread,
    parentPort,
} from "node:worker_threads";

import { availableParallelism } from "node:os";
import { performance } from "node:perf_hooks";


const TASK_COUNT = 5;
const WORK_ITERATIONS = 4_000_000;
const WARMUP_ITERATIONS = 250_000;


function heavyLoad(seed, iterations) {
    let state = seed;
    let accumulator = 0;

    for (let i = 0; i < iterations; i++) {
        state = (state * 48271 + 1) % 2147483647;

        accumulator += state % 1009;

        if ((i % 4096) === 0) {
            state = (state + (accumulator % 65521)) % 2147483647;
        }
    }

    return accumulator + state;
}


// ============================================================================
// Worker implementation
// ============================================================================

if (!isMainThread) {
    parentPort.on("message", ({ id, seed, iterations }) => {
        try {
            const result = heavyLoad(seed, iterations);

            parentPort.postMessage({
                type: "result",
                id,
                result,
            });
        } catch (error) {
            parentPort.postMessage({
                type: "error",
                id,
                error: error instanceof Error
                    ? error.stack ?? error.message
                    : String(error),
            });
        }
    });

    parentPort.postMessage({
        type: "ready",
    });
}


// ============================================================================
// Main-thread worker wrapper
// ============================================================================

class WorkerClient {
    #worker;
    #ready;
    #resolveReady;
    #rejectReady;
    #nextTaskId = 1;
    #pending = new Map();
    #closing = false;

    constructor() {
        this.#worker = new Worker(new URL(import.meta.url));

        this.#ready = new Promise((resolve, reject) => {
            this.#resolveReady = resolve;
            this.#rejectReady = reject;
        });

        this.#worker.on("message", (message) => {
            if (message.type === "ready") {
                this.#resolveReady();
                return;
            }

            const pending = this.#pending.get(message.id);

            if (pending === undefined) {
                return;
            }

            this.#pending.delete(message.id);

            if (message.type === "result") {
                pending.resolve(message.result);
                return;
            }

            pending.reject(new Error(message.error));
        });

        this.#worker.on("error", (error) => {
            this.#rejectReady(error);
            this.#rejectAll(error);
        });

        this.#worker.on("exit", (code) => {
            if (this.#closing) {
                return;
            }

            const error = new Error(
                `Worker unexpectedly exited with code ${code}`,
            );

            this.#rejectReady(error);
            this.#rejectAll(error);
        });
    }

    async ready() {
        await this.#ready;
    }

    async run(seed, iterations) {
        await this.#ready;

        const id = this.#nextTaskId++;

        return new Promise((resolve, reject) => {
            this.#pending.set(id, {
                resolve,
                reject,
            });

            this.#worker.postMessage({
                id,
                seed,
                iterations,
            });
        });
    }

    async close() {
        if (this.#closing) {
            return;
        }

        this.#closing = true;

        await this.#worker.terminate();
    }

    #rejectAll(error) {
        for (const pending of this.#pending.values()) {
            pending.reject(error);
        }

        this.#pending.clear();
    }
}


// ============================================================================
// Benchmark
// ============================================================================

async function runBenchmark() {
    console.log("========================================");
    console.log("Node.js CPU concurrency benchmark");
    console.log("========================================");
    const concurrentStart = performance.now();
    console.log("Tasks:");
    console.log(TASK_COUNT);
    console.log("Iterations per task:");
    console.log(WORK_ITERATIONS);
    console.log("Available parallelism:");
    console.log(availableParallelism());


    // ========================================================================
    // Worker creation
    // ========================================================================

    // Create the workers before the timed region so Worker construction itself
    // is not measured.
    //
    // However, no workload is executed in them before the concurrent benchmark.
    // Their V8 isolates and heavyLoad() machine code therefore start cold.
    const workers = Array.from(
        { length: TASK_COUNT },
        () => new WorkerClient(),
    );

    try {
        // Wait only until the Workers are alive and able to receive messages.
        // This does not execute heavyLoad() and does not warm its code.
        await Promise.all(
            workers.map((worker) => worker.ready()),
        );


        // ====================================================================
        // Concurrent benchmark
        // ====================================================================

        console.log("");
        console.log("----------------------------------------");
        console.log("CONCURRENT RUN");
        console.log("----------------------------------------");

        const concurrentPromises = [
            workers[0].run(101, WORK_ITERATIONS),
            workers[1].run(102, WORK_ITERATIONS),
            workers[2].run(103, WORK_ITERATIONS),
            workers[3].run(104, WORK_ITERATIONS),
            workers[4].run(105, WORK_ITERATIONS),
        ];

        const concurrentResults =
            await Promise.all(concurrentPromises);

        const concurrentEnd = performance.now();

        const concurrentMs =
            concurrentEnd - concurrentStart;

        const concurrentChecksum =
            concurrentResults.reduce(
                (sum, value) => sum + value,
                0,
            );

        console.log("Concurrent time (ms):");
        console.log(concurrentMs);

        console.log("Concurrent checksum:");
        console.log(concurrentChecksum);


        // ====================================================================
        // Sequential benchmark
        // ====================================================================

        console.log("");
        console.log("----------------------------------------");
        console.log("SEQUENTIAL RUN");
        console.log("----------------------------------------");

        const sequentialStart = performance.now();

        const s1 = heavyLoad(201, WORK_ITERATIONS);
        const s2 = heavyLoad(202, WORK_ITERATIONS);
        const s3 = heavyLoad(203, WORK_ITERATIONS);
        const s4 = heavyLoad(204, WORK_ITERATIONS);
        const s5 = heavyLoad(205, WORK_ITERATIONS);

        const sequentialEnd = performance.now();

        const sequentialMs =
            sequentialEnd - sequentialStart;

        const sequentialChecksum =
            s1 + s2 + s3 + s4 + s5;

        console.log("Sequential time (ms):");
        console.log(sequentialMs);

        console.log("Sequential checksum:");
        console.log(sequentialChecksum);


        // ====================================================================
        // Results
        // ====================================================================

        const speedup =
            sequentialMs / concurrentMs;

        const idealSpeedup =
            TASK_COUNT;

        const machineMaximum =
            Math.min(
                TASK_COUNT,
                availableParallelism(),
            );

        const parallelEfficiency =
            (speedup / idealSpeedup) * 100;


        console.log("");
        console.log("========================================");
        console.log("RESULTS");
        console.log("========================================");

        console.log("Concurrent time (ms):");
        console.log(concurrentMs);

        console.log("Sequential time (ms):");
        console.log(sequentialMs);

        console.log("Speedup:");
        console.log(speedup);

        console.log("Maximum theoretical speedup:");
        console.log(idealSpeedup);

        console.log("Machine-limited maximum:");
        console.log(machineMaximum);

        console.log("Parallel efficiency (%):");
        console.log(parallelEfficiency);

        console.log("");

        if (concurrentMs < sequentialMs) {
            console.log(
                "PASS: concurrent execution was faster.",
            );
        } else {
            console.log(
                "FAIL: concurrent execution was not faster.",
            );
        }

        if (speedup >= 4.0) {
            console.log(
                "EXCELLENT: near-linear scaling for five CPU-bound tasks.",
            );
        } else if (speedup >= 3.0) {
            console.log(
                "GOOD: substantial real parallelism detected.",
            );
        } else if (speedup >= 1.5) {
            console.log(
                "WARNING: parallelism works, but scaling is poor.",
            );
        } else {
            console.log(
                "ERROR: tasks may not actually be executing efficiently in parallel.",
            );
        }

        console.log("========================================");
    } finally {
        await Promise.all(
            workers.map((worker) => worker.close()),
        );
    }
}


if (isMainThread) {
    await runBenchmark();
}
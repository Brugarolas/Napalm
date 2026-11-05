// Builds Napalm in release mode and runs every Napalm program in ./scripts.
//
// Usage (Node >= 22.18 runs TypeScript natively):
//   node tools/run-scripts.ts [options] [name-filter ...]
//
// Options:
//   --no-build            Reuse the existing target/release binary.
//   --cargo-arg <arg>     Extra argument for `cargo build` (repeatable),
//                         e.g. --cargo-arg --no-default-features
//   --network             Also run scripts that need Internet access.
//   --skip-benches        Run only the functional scripts.
//   --timeout <seconds>   Default per-script timeout (default 120).
//   --verbose             Print the output of every script, not only failures.
//
// A script passes when it exits with code 0 within its timeout and prints no
// Rust panic. Long-running servers pass once their expected output appears
// (they are then stopped). Exit code: 0 if every script passed, 1 otherwise.

import { spawn, spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

type Kind = "test" | "bench" | "server";

interface ScriptSpec {
    kind?: Kind;
    /** Library module: only meaningful when imported by another script. */
    module?: boolean;
    /** Requires Internet access; skipped unless --network. */
    network?: boolean;
    /** Servers: stop and pass as soon as the output matches. */
    readyWhen?: RegExp;
    timeoutSeconds?: number;
}

const SPECS: Record<string, ScriptSpec> = {
    // Imported by import.napalm / module_import.napalm.
    math: { module: true },
    other: { module: true },
    module_exports: { module: true },
    // Talks to https://example.com.
    fetch: { network: true },
    // HTTP servers keep the event loop alive forever (port 3000).
    server: { kind: "server", readyWhen: /tick 2/, timeoutSeconds: 30 },
    server_simple: { kind: "server", readyWhen: /Hello!/, timeoutSeconds: 30 },
    // Benchmarks.
    concurrent: { kind: "bench", timeoutSeconds: 300 },
    fib: { kind: "bench" },
    gc: { kind: "bench" },
    jit: { kind: "bench" },
    jit_entries: { kind: "bench" },
    lev: { kind: "bench" },
    reactive_bench: { kind: "bench" },
    trees: { kind: "bench" },
};

interface Options {
    build: boolean;
    cargoArgs: string[];
    network: boolean;
    skipBenches: boolean;
    timeoutSeconds: number;
    verbose: boolean;
    filters: string[];
}

interface Result {
    name: string;
    kind: Kind;
    status: "pass" | "fail" | "skip";
    reason: string;
    millis: number;
    output: string;
}

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scriptsDir = join(root, "scripts");
const exe = join(root, "target", "release", process.platform === "win32" ? "napalm.exe" : "napalm");

function parseArgs(argv: string[]): Options {
    const options: Options = {
        build: true,
        cargoArgs: [],
        network: false,
        skipBenches: false,
        timeoutSeconds: 120,
        verbose: false,
        filters: [],
    };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        const value = (): string => {
            const next = argv[++i];
            if (next === undefined) throw new Error(`missing value for ${arg}`);
            return next;
        };
        switch (arg) {
            case "--no-build": options.build = false; break;
            case "--cargo-arg": options.cargoArgs.push(value()); break;
            case "--network": options.network = true; break;
            case "--skip-benches": options.skipBenches = true; break;
            case "--timeout": options.timeoutSeconds = Number(value()); break;
            case "--verbose": options.verbose = true; break;
            case "-h":
            case "--help":
                console.log("usage: node tools/run-scripts.ts [--no-build] [--cargo-arg ARG]... [--network] [--skip-benches] [--timeout S] [--verbose] [filter...]");
                process.exit(0);
            default:
                if (arg.startsWith("-")) throw new Error(`unknown option ${arg}`);
                options.filters.push(arg);
        }
    }
    return options;
}

function build(cargoArgs: string[]): void {
    const args = ["build", "--release", ...cargoArgs];
    console.log(`> cargo ${args.join(" ")}`);
    const result = spawnSync("cargo", args, { cwd: root, stdio: "inherit" });
    if (result.error) throw result.error;
    if (result.status !== 0) {
        console.error(`cargo build failed with exit code ${result.status}`);
        process.exit(result.status ?? 1);
    }
}

function runScript(name: string, spec: ScriptSpec, kind: Kind, options: Options): Promise<Result> {
    const file = join(scriptsDir, `${name}.napalm`);
    const timeoutMs = (spec.timeoutSeconds ?? options.timeoutSeconds) * 1000;
    const started = performance.now();
    return new Promise((done) => {
        const child = spawn(exe, [file], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
        let output = "";
        let settled = false;
        const finish = (status: Result["status"], reason: string): void => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            if (child.exitCode === null) child.kill();
            done({ name, kind, status, reason, millis: performance.now() - started, output });
        };
        const onData = (chunk: Buffer): void => {
            output += chunk.toString("utf8");
            if (spec.readyWhen?.test(output)) finish("pass", "expected output seen");
        };
        child.stdout.on("data", onData);
        child.stderr.on("data", onData);
        const timer = setTimeout(() => finish("fail", `timed out after ${timeoutMs / 1000}s`), timeoutMs);
        child.on("error", (error) => finish("fail", `failed to start: ${error.message}`));
        child.on("close", (code, signal) => {
            if (/panicked at/.test(output)) finish("fail", "Rust panic");
            else if (spec.readyWhen) finish("fail", `exited (${code ?? signal}) before the expected output`);
            else if (code === 0) finish("pass", "");
            else finish("fail", `exit code ${code ?? signal}`);
        });
    });
}

function indent(text: string): string {
    return text.trimEnd().split(/\r?\n/).map((line) => `    ${line}`).join("\n");
}

async function main(): Promise<void> {
    const options = parseArgs(process.argv.slice(2));
    if (options.build) build(options.cargoArgs);
    if (!existsSync(exe)) {
        console.error(`Napalm binary not found at ${exe}; run without --no-build.`);
        process.exit(1);
    }

    const names = readdirSync(scriptsDir)
        .filter((file) => file.endsWith(".napalm"))
        .map((file) => basename(file, ".napalm"))
        .filter((name) => options.filters.length === 0 || options.filters.some((f) => name.includes(f)))
        .sort();

    const results: Result[] = [];
    for (const name of names) {
        const spec = SPECS[name] ?? {};
        const kind = spec.kind ?? "test";
        const skip = (reason: string): void => {
            results.push({ name, kind, status: "skip", reason, millis: 0, output: "" });
        };
        if (spec.module) { skip("library module (run through its importers)"); continue; }
        if (spec.network && !options.network) { skip("needs network (use --network)"); continue; }
        if (kind === "bench" && options.skipBenches) { skip("benchmark (--skip-benches)"); continue; }

        const result = await runScript(name, spec, kind, options);
        results.push(result);
        const time = `${(result.millis / 1000).toFixed(2)}s`.padStart(8);
        const label = result.status === "pass" ? "PASS" : "FAIL";
        console.log(`${label} ${time}  ${kind.padEnd(6)} ${name}${result.reason ? `  (${result.reason})` : ""}`);
        if (result.status === "fail" || options.verbose) {
            if (result.output.trim()) console.log(indent(result.output));
        }
    }

    const count = (status: Result["status"]): number => results.filter((r) => r.status === status).length;
    for (const skipped of results.filter((r) => r.status === "skip")) {
        console.log(`SKIP           ${skipped.name}  (${skipped.reason})`);
    }
    console.log(`\n${count("pass")} passed, ${count("fail")} failed, ${count("skip")} skipped`);
    process.exit(count("fail") === 0 ? 0 : 1);
}

main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
});

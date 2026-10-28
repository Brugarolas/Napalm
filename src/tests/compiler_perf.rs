//! Manual compiler-only workload; no VM/JIT execution enters the timings.
//! Run with `cargo test --release compiler_workload -- --ignored --nocapture`.

use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "manual compiler timing workload"]
fn compiler_workload() {
    let mut source = String::new();
    for index in 0..256 {
        source.push_str(&format!(r#"
            function function_{index}(x: number) {{
                const inner = (y: number) => x + y;
                let sum = 0;
                for (let j = 0; j < 8; j++) {{ sum += inner(j); }}
                return sum;
            }}
        "#));
    }
    let mut timings = [Vec::new(), Vec::new(), Vec::new()];
    for iteration in 0..6 {
        let start = Instant::now();
        let parsed = crate::napalm_front_main::parse("compiler-workload.napalm", &source);
        let parse_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(!parsed.has_errors(), "{:?}", parsed.diagnostics);

        let mut hir = crate::napalm_ir::Program::new();
        crate::napalm_builtin::install_hir_builtins(&mut hir);
        let start = Instant::now();
        let lowered = crate::napalm_lowerer::lower_module(
            &mut hir, "compiler-workload.napalm", &source, "workload", &parsed.module,
        );
        let lower_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(!lowered.has_errors(), "{:?}", lowered.diagnostics);

        let start = Instant::now();
        let bytecode = crate::napalm_pipeline::compile_program(&mut hir).unwrap();
        let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
        black_box(&bytecode);
        if iteration != 0 {
            for (samples, value) in timings.iter_mut().zip([parse_ms, lower_ms, compile_ms]) {
                samples.push(value);
            }
        }
    }
    for (name, samples) in ["parse", "lower", "optimize+bytecode"].into_iter().zip(&mut timings) {
        samples.sort_by(f64::total_cmp);
        eprintln!("compiler_workload {name}: median {:.3} ms (5 samples)", samples[2]);
    }
}

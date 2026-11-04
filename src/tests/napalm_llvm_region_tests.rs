// LLVM trace-tree regions: a hot trace and the traces it reaches through
// stitched native links compiled as one function. Included in the
// interpreter's source-test module.

#[cfg(feature = "llvm-jit")]
const REGION_TEST_SOURCE: &str = r#"
    class Test {
        static run() {
            var sum = 0
            var escaped = 0
            var y = 0
            while (y < 140) {
                var x = 0
                while (x < 110) {
                    var i = 0
                    while (i < 9 && (x * 7 + y * 3 + i) % 11 != 4) {
                        i = i + 1
                    }
                    if (i < 9) {
                        escaped = escaped + 1
                    } else {
                        sum = sum - 2
                    }
                    sum = sum + i * (x + 1)
                    x = x + 1
                }
                y = y + 1
            }
            return sum * 100000 + escaped
        }
    }
    var result = Test.run()
"#;

#[cfg(feature = "llvm-jit")]
fn run_region_test(options: Option<TracingJitOptions>) -> (f64, Option<crate::napalm_tracing_jit::TracingJitStats>) {
    let (program, entry, manifest) = compile_napalm_module(REGION_TEST_SOURCE);
    let mut vm = source_vm(&program, &manifest);
    vm.set_jit_options(options);
    vm.run_module(entry).unwrap();
    (source_result(&vm), vm.jit_stats().cloned())
}

#[cfg(feature = "llvm-jit")]
#[test]
fn llvm_region_nested_loops_match_interpreter() {
    let (expected, _) = run_region_test(None);
    for regions in [false, true] {
        let (result, stats) = run_region_test(Some(TracingJitOptions {
            llvm_hot_threshold: 1,
            llvm_background_compile: false,
            llvm_trace_regions: regions,
            ..Default::default()
        }));
        let stats = stats.unwrap();
        assert_eq!(result, expected, "regions={regions} {stats:?}");
        assert!(stats.side_traces_compiled >= 1, "{stats:?}");
        if regions {
            assert!(stats.llvm_regions_installed >= 1, "{stats:?}");
            assert!(stats.llvm_region_members >= 2, "{stats:?}");
        } else {
            assert_eq!(stats.llvm_regions_installed, 0, "{stats:?}");
        }
    }
}

/// Internal region links spend the dispatcher's link budget: a tiny budget
/// makes the region yield to the manager (and the interpreter) constantly.
#[cfg(feature = "llvm-jit")]
#[test]
fn llvm_region_link_budget_yields_preserve_results() {
    let (expected, _) = run_region_test(None);
    for budget in [1, 2, 3, 7, 64] {
        let (result, stats) = run_region_test(Some(TracingJitOptions {
            llvm_hot_threshold: 1,
            llvm_background_compile: false,
            native_loop_iteration_budget: budget,
            ..Default::default()
        }));
        let stats = stats.unwrap();
        assert_eq!(result, expected, "budget={budget} {stats:?}");
        assert!(stats.native_budget_exits > 0, "budget={budget} {stats:?}");
    }
}

/// Background compilation: regions are installed (and regrown) while the
/// trace tree runs, through the tier-up yield of internal links.
#[cfg(feature = "llvm-jit")]
#[test]
fn llvm_region_background_install_preserves_results() {
    let (expected, _) = run_region_test(None);
    for threshold in [1, 5, 50] {
        let (result, stats) = run_region_test(Some(TracingJitOptions {
            llvm_hot_threshold: threshold,
            llvm_background_compile: true,
            ..Default::default()
        }));
        assert_eq!(result, expected, "threshold={threshold} {:?}", stats);
    }
}

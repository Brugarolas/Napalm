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

/// Values crossing internal links unboxed: NaN, -0, infinities, numbers
/// outside int32, fractions and negative integers, through the inner-loop
/// exit (root -> side trace) and back (side -> root).
#[cfg(feature = "llvm-jit")]
const REGION_SPECIAL_VALUES_SOURCE: &str = r#"
    class Test {
        static run() {
            var nans = 0
            var negzeros = 0
            var infs = 0
            var acc = 0
            var v = 0
            var y = 0
            while (y < 120) {
                var x = 0
                while (x < 90) {
                    var k = (x * 13 + y) % 7
                    if (k == 0) {
                        v = 0 / 0
                    } else {
                        if (k == 1) {
                            v = 0 * -1
                        } else {
                            if (k == 2) {
                                v = 1 / 0
                            } else {
                                if (k == 3) {
                                    v = 3000000000 + x
                                } else {
                                    if (k == 4) {
                                        v = x + 0.5
                                    } else {
                                        v = x - y
                                    }
                                }
                            }
                        }
                    }
                    var i = 0
                    var t = 0
                    while (i < 3) {
                        t = v * 1
                        i = i + 1
                    }
                    if (t != t) {
                        nans = nans + 1
                    } else {
                        if (t == 0 && 1 / t < 0) {
                            negzeros = negzeros + 1
                        } else {
                            if (t > 1e300) {
                                infs = infs + 1
                            } else {
                                acc = acc + t * i
                            }
                        }
                    }
                    x = x + 1
                }
                y = y + 1
            }
            return acc + nans * 7 + negzeros * 3 + infs * 5
        }
    }
    var result = Test.run()
"#;

/// Guards fail inside region members while deferred frame writes are pending:
/// `w` changes from small integers to fractions, to numbers outside int32 and
/// finally to NaN, and `step` from integer to fraction, so the region
/// exits (and is re-entered or abandoned) from many members. The interpreter
/// must resume with the exact frame.
#[cfg(feature = "llvm-jit")]
const REGION_DEFERRED_EXITS_SOURCE: &str = r#"
    class Test {
        static run() {
            var total = 0
            var w = 1
            var step = 1
            var text = 0
            var y = 0
            while (y < 160) {
                if (y == 60) { w = 0.25 }
                if (y == 90) { w = 5000000000 }
                if (y == 110) { step = 0.5 }
                if (y == 130) { w = 0 / 0 }
                var x = 0
                while (x < 70) {
                    var i = 0
                    while (i < 4) {
                        i = i + step
                    }
                    if (w != w) {
                        text = text + 1
                    } else {
                        total = total + w * i + x
                    }
                    x = x + 1
                }
                y = y + 1
            }
            return total * 10 + text
        }
    }
    var result = Test.run()
"#;

#[cfg(feature = "llvm-jit")]
fn run_region_test(options: Option<TracingJitOptions>) -> (f64, Option<crate::napalm_tracing_jit::TracingJitStats>) {
    run_region_source(REGION_TEST_SOURCE, options)
}

#[cfg(feature = "llvm-jit")]
fn run_region_source(
    source: &str,
    options: Option<TracingJitOptions>,
) -> (f64, Option<crate::napalm_tracing_jit::TracingJitStats>) {
    let (program, entry, manifest) = compile_napalm_module(source);
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

#[cfg(feature = "llvm-jit")]
#[test]
fn llvm_region_unboxed_links_preserve_special_numbers() {
    let (expected, _) = run_region_source(REGION_SPECIAL_VALUES_SOURCE, None);
    for regions in [false, true] {
        let (result, stats) = run_region_source(REGION_SPECIAL_VALUES_SOURCE, Some(TracingJitOptions {
            llvm_hot_threshold: 1,
            llvm_background_compile: false,
            llvm_trace_regions: regions,
            ..Default::default()
        }));
        let stats = stats.unwrap();
        assert_eq!(result.to_bits(), expected.to_bits(), "regions={regions} {result} != {expected} {stats:?}");
        if regions {
            assert!(stats.llvm_regions_installed >= 1, "{stats:?}");
        }
    }
}

#[cfg(feature = "llvm-jit")]
#[test]
fn llvm_region_deferred_frame_writes_survive_exits_from_any_member() {
    let (expected, _) = run_region_source(REGION_DEFERRED_EXITS_SOURCE, None);
    for budget in [3, 1 << 16] {
        for regions in [false, true] {
            let (result, stats) = run_region_source(REGION_DEFERRED_EXITS_SOURCE, Some(TracingJitOptions {
                llvm_hot_threshold: 1,
                llvm_background_compile: false,
                llvm_trace_regions: regions,
                native_loop_iteration_budget: budget,
                ..Default::default()
            }));
            let stats = stats.unwrap();
            assert_eq!(result.to_bits(), expected.to_bits(), "regions={regions} budget={budget} {result} != {expected} {stats:?}");
            if regions {
                assert!(stats.llvm_regions_installed >= 1, "{stats:?}");
            }
            assert!(stats.side_exits > 0, "{stats:?}");
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

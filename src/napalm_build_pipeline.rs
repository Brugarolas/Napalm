//! Build-oriented HIR -> bytecode pipeline, independent of source execution.
//!
//! Its initial passes match the interpreter pipeline, but complete HIR rounds
//! run to a fixed point before bytecode lowering. Build-specific stages and
//! policies belong here so normal source execution retains its cold-start cost.

use crate::napalm_bytecode::BytecodeProgram;
use crate::napalm_bytecodecompiler::{
    BytecodeCompiler, CompilationOutput, CompileError, CompileOptions,
};
use crate::napalm_ir::Program;
use crate::napalm_ir_optimizer::Optimize;
use crate::napalm_ir_optimizing::{
    InterpreterOptimizationBundle, MonotoneAutoMemoization, SharedConstantLiterals,
    StaticTypeSpecialization,
};
use crate::napalm_ir_optimizing::aot::{Ipcp, Specialization, Inlining, Sccp, Licm, Gvn, Pre, PartialEvaluation, TreeShaking};
use crate::napalm_ir_escape::ScalarReplacement;

/// Separate pass selection for ahead-of-time builds: the interpreter transforms
/// plus AOT-only HIR passes (see `napalm_ir_optimizing::aot`). The whole list is
/// one round of the fixed-point loop in `optimize_hir`, so every pass sees the
/// facts exposed by the others (inlined bodies, specialized clones, constants).
pub fn optimization_passes() -> Optimize {
    Optimize::new()
        // Function-local: interpreter bundle, then GVN, PRE, LICM and escape
        // analysis/scalar replacement over the result.
        .with_function_pass(InterpreterOptimizationBundle::default())
        .with_function_pass(Gvn)
        .with_function_pass(Pre)
        .with_function_pass(Licm)
        .with_function_pass(ScalarReplacement::default())
        // Whole-program: static types first (direct calls), then automatic
        // memoization, IPCP, SCCP, partial evaluation, interprocedural (object)
        // specialization and inlining; literal sharing follows memoization's
        // direct call graph, and tree shaking drops what became unreachable.
        // Memoization has priority: it classifies before any clone or inlined
        // copy can bypass a function, its classifications are never withdrawn in
        // later rounds, and no AOT pass clones or inlines a memoized function.
        .with_program_finalizer(StaticTypeSpecialization::default())
        .with_program_finalizer(MonotoneAutoMemoization)
        .with_program_finalizer(Ipcp)
        .with_program_finalizer(Sccp)
        .with_program_finalizer(PartialEvaluation)
        .with_program_finalizer(Specialization)
        .with_program_finalizer(Inlining)
        .with_program_finalizer(SharedConstantLiterals::default())
        .with_program_finalizer(TreeShaking)
}

pub struct BuildCompilationPipeline {
    optimizations: Optimize,
    bytecode: BytecodeCompiler,
}

impl Default for BuildCompilationPipeline {
    fn default() -> Self {
        Self::new(optimization_passes())
    }
}

impl BuildCompilationPipeline {
    pub fn new(optimizations: Optimize) -> Self {
        Self {
            optimizations,
            bytecode: BytecodeCompiler::default(),
        }
    }

    pub fn with_bytecode_options(mut self, options: CompileOptions) -> Self {
        self.bytecode = BytecodeCompiler::new(options);
        self
    }

    pub fn compile(&mut self, hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
        Ok(self.compile_with_stats(hir)?.program)
    }

    /// HIR-only phase. Compare the actual HIR before/after each complete round;
    /// pass change flags are not a convergence criterion. No bytecode is generated
    /// or inspected here. Returns the number of rounds, including the stable one.
    pub fn optimize_hir(&mut self, hir: &mut Program) -> Result<usize, CompileError> {
        #[cfg(debug_assertions)]
        let debug_compilation_id = {
            let id = crate::napalm_pipeline::debug_dump::next_compilation_id();
            crate::napalm_pipeline::debug_dump::dump_hir(id, "HIR BEFORE BUILD OPTIMIZATIONS", hir);
            id
        };

        let mut before = hir_snapshot(hir)?;
        let mut round = 0usize;
        loop {
            round += 1;
            self.optimizations.optimize_program(hir);
            let after = hir_snapshot(hir)?;
            let changed = before != after;
            #[cfg(debug_assertions)]
            crate::napalm_pipeline::debug_dump::dump_hir(
                debug_compilation_id,
                &format!("HIR AFTER BUILD OPTIMIZATION ROUND {round} (changed={changed})"),
                hir,
            );
            if !changed {
                return Ok(round);
            }
            before = after;
        }
    }

    /// The build orchestrator runs the standalone HIR phase first. Bytecode
    /// lowering is a subsequent stage, outside the optimization loop.
    pub fn compile_with_stats(
        &mut self,
        hir: &mut Program,
    ) -> Result<CompilationOutput, CompileError> {
        self.optimize_hir(hir)?;
        #[cfg(debug_assertions)]
        let debug_compilation_id = crate::napalm_pipeline::debug_dump::next_compilation_id();
        let output = self.bytecode.compile_with_stats(hir);
        #[cfg(debug_assertions)]
        match &output {
            Ok(output) => crate::napalm_pipeline::debug_dump::dump_bytecode(
                debug_compilation_id,
                &output.program,
            ),
            Err(error) => {
                crate::napalm_pipeline::debug_dump::dump_compile_error(debug_compilation_id, error)
            }
        }
        output
    }
}

fn hir_snapshot(hir: &Program) -> Result<Vec<u8>, CompileError> {
    serde_json::to_vec(hir).map_err(|error| CompileError::Internal {
        function: None,
        message: format!("failed to compare HIR optimization state: {error}"),
    })
}

pub fn compile_program(hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
    BuildCompilationPipeline::default().compile(hir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_interpreter::{ExecutionResult, Interpreter};
    use crate::napalm_ir::{Builder, Constant, Function, FunctionKind, Inst, Type};
    use crate::napalm_ir_optimizer::{
        FunctionOptimization, InstCursor, OptimizationPass, OptimizationWIR, ProgramOptimization,
        WirEditQueue,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    fn program() -> Program {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        let mut builder = Builder::new(
            &mut hir,
            module,
            "main",
            FunctionKind::ModuleBody,
            vec![],
            Type::Num,
        );
        let value = builder.const_num(20.0);
        builder.ret(value);
        builder.finish();
        hir
    }

    struct IncrementOncePerRound(Rc<Cell<usize>>);
    impl FunctionOptimization for IncrementOncePerRound {
        fn optimize_function(&mut self, function: &mut Function) -> bool {
            self.0.set(self.0.get() + 1);
            for node in &mut function.nodes {
                if let Inst::Const(Constant::Num(value)) = &mut node.kind {
                    if *value < 23.0 {
                        *value += 1.0;
                        return true;
                    }
                }
            }
            false
        }
    }

    #[test]
    fn build_repeats_hir_rounds_but_normal_compilation_runs_once() {
        for (build, expected_rounds, expected_value) in [(true, 4, 23.0), (false, 1, 21.0)] {
            let rounds = Rc::new(Cell::new(0));
            let optimizer =
                Optimize::new().with_function_pass(IncrementOncePerRound(rounds.clone()));
            let mut hir = program();
            let entry = hir.functions[0].id;
            let bytecode = if build {
                BuildCompilationPipeline::new(optimizer)
                    .compile(&mut hir)
                    .unwrap()
            } else {
                crate::napalm_pipeline::CompilationPipeline::new(optimizer)
                    .compile(&mut hir)
                    .unwrap()
            };
            assert_eq!(rounds.get(), expected_rounds);
            let mut vm = Interpreter::new(&bytecode).unwrap();
            vm.set_jit_options(None);
            let ExecutionResult::Returned(result) = vm.run_module(entry).unwrap() else {
                panic!("expected return")
            };
            assert_eq!(result.as_num(), Some(expected_value));
        }
    }

    #[test]
    fn whole_program_metadata_changes_keep_the_build_loop_running() {
        struct AddSlot(Rc<Cell<usize>>);
        impl ProgramOptimization for AddSlot {
            fn optimize_program(&mut self, program: &mut Program) -> bool {
                self.0.set(self.0.get() + 1);
                if program.modules[0].variable_count < 3 {
                    program.modules[0].add_hidden_slot();
                    true
                } else {
                    false
                }
            }
        }
        let rounds = Rc::new(Cell::new(0));
        let mut hir = program();
        let output = BuildCompilationPipeline::new(
            Optimize::new().with_program_finalizer(AddSlot(rounds.clone())),
        )
        .with_bytecode_options(CompileOptions {
            optimize_bytecode: false,
            ..CompileOptions::default()
        })
        .compile_with_stats(&mut hir)
        .unwrap();
        assert_eq!(rounds.get(), 4);
        assert_eq!(output.program.metadata.modules[0].hidden_variable_count, 3);
        assert_eq!(
            output.stats.total_bytecode_words,
            output.program.functions[0].chunk.code.len()
        );
    }

    #[test]
    fn hir_comparison_detects_both_direct_and_queued_node_changes() {
        struct ReplaceNumber(bool);
        impl OptimizationWIR for ReplaceNumber {
            fn optimize_node(
                &mut self,
                cur: &mut InstCursor<'_>,
                queue: &mut WirEditQueue,
            ) -> bool {
                if matches!(cur.cur_kind(), Inst::Const(Constant::Num(20.0))) {
                    let replacement = Inst::Const(Constant::Num(21.0));
                    if self.0 {
                        cur.with_cur(|node| node.kind = replacement);
                        return true;
                    }
                    queue.replace_kind(cur.cur_id(), replacement);
                    // Queued edits themselves must also propagate change information.
                }
                false
            }
        }
        for direct in [false, true] {
            let optimizer = Optimize::new()
                .with_pass(OptimizationPass::new(vec![Box::new(ReplaceNumber(direct))]));
            let mut hir = program();
            let mut pipeline = BuildCompilationPipeline::new(optimizer);
            assert_eq!(pipeline.optimize_hir(&mut hir).unwrap(), 2);
            assert_eq!(pipeline.optimize_hir(&mut hir).unwrap(), 1);
        }
    }

    #[test]
    fn default_build_passes_converge_on_modules_classes_closures_and_gc() {
        let mut hir = Program::new();
        crate::napalm_builtin::install_hir_builtins(&mut hir);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/burning/entry.napalm");
        crate::napalm_modules::compile_module_graph(&mut hir, &path).unwrap();
        let mut optimizer = optimization_passes();
        // A test watchdog, not a production iteration limit.
        let stable = (0..64).any(|_| {
            let before = hir_snapshot(&hir).unwrap();
            optimizer.optimize_program(&mut hir);
            before == hir_snapshot(&hir).unwrap()
        });
        assert!(stable, "default HIR passes failed to converge");
        assert_eq!(
            BuildCompilationPipeline::default()
                .optimize_hir(&mut hir)
                .unwrap(),
            1
        );
    }

    #[test]
    fn hir_phase_never_invokes_bytecode_lowering() {
        let mut hir = Program::new();
        let module = hir.add_module("main", 0);
        // Valid HIR with more arguments than the bytecode ABI can represent.
        let mut builder = Builder::new(
            &mut hir,
            module,
            "wide",
            FunctionKind::Closure,
            vec![Type::Num; 17],
            Type::Num,
        );
        let value = builder.const_num(20.0);
        builder.ret(value);
        builder.finish();
        assert!(BytecodeCompiler::default().compile(&hir).is_err());
        assert_eq!(
            BuildCompilationPipeline::new(Optimize::new())
                .optimize_hir(&mut hir)
                .unwrap(),
            1
        );
    }

    #[test]
    fn convergence_uses_actual_hir_even_when_pass_flags_are_wrong() {
        struct MisreportedChanges(Rc<Cell<usize>>);
        impl ProgramOptimization for MisreportedChanges {
            fn optimize_program(&mut self, program: &mut Program) -> bool {
                self.0.set(self.0.get() + 1);
                assert!(
                    self.0.get() <= 4,
                    "stable HIR must stop even if the pass returns true"
                );
                if program.modules[0].variable_count < 3 {
                    program.modules[0].add_hidden_slot();
                    false // Actual metadata change, falsely reported as unchanged.
                } else {
                    true // No change, falsely reported as changed.
                }
            }
        }
        let rounds = Rc::new(Cell::new(0));
        let mut hir = program();
        let mut pipeline = BuildCompilationPipeline::new(
            Optimize::new().with_program_finalizer(MisreportedChanges(rounds.clone())),
        );
        assert_eq!(pipeline.optimize_hir(&mut hir).unwrap(), 4);
        assert_eq!(rounds.get(), 4);
        assert_eq!(hir.modules[0].hidden_variable_count, 3);
    }

    #[test]
    fn hir_snapshot_preserves_float_bits_and_ignores_hash_iteration_order() {
        use crate::napalm_ir::BlockId;
        let mut hir = program();
        hir.functions[0]
            .profiles
            .block_counters
            .insert(BlockId(0), 10);
        hir.functions[0]
            .profiles
            .block_counters
            .insert(BlockId(1), 20);
        let before = hir_snapshot(&hir).unwrap();
        hir.functions[0].profiles.block_counters.clear();
        hir.functions[0]
            .profiles
            .block_counters
            .insert(BlockId(1), 20);
        hir.functions[0]
            .profiles
            .block_counters
            .insert(BlockId(0), 10);
        assert_eq!(before, hir_snapshot(&hir).unwrap());
        let constant = hir.functions[0]
            .nodes
            .iter()
            .position(|node| matches!(node.kind, Inst::Const(_)))
            .unwrap();
        let mut snapshots = Vec::new();
        for bits in [
            0,
            (-0.0f64).to_bits(),
            0x7ff8_0000_0000_0001,
            0x7ff8_0000_0000_0002,
        ] {
            hir.functions[0].nodes[constant].kind =
                Inst::Const(Constant::Num(f64::from_bits(bits)));
            let snapshot = hir_snapshot(&hir).unwrap();
            assert!(!snapshots.contains(&snapshot));
            assert_eq!(snapshot, hir_snapshot(&hir).unwrap());
            snapshots.push(snapshot);
        }
    }
}

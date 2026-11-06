//! Shared HIR -> optimizations -> bytecode compilation pipeline.
//!
//! Frontends lower into a `Program`, then enter this pipeline before execution.
//! Keep the default pass list in `napalm_ir_optimizing::optimization_passes`; custom
//! pipelines can use the same `Optimize` builder for function and node passes.

use crate::napalm_bytecode::BytecodeProgram;
use crate::napalm_bytecodecompiler::{
    BytecodeCompiler, CompilationOutput, CompileError, CompileOptions,
};
use crate::napalm_ir::Program;
use crate::napalm_ir_optimizer::Optimize;
use crate::napalm_ir_optimizing::optimization_passes;

#[cfg(debug_assertions)]
mod debug_dump {
    use super::*;
    use crate::napalm_bytecode::disassemble;
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    const LOG_PATH: &str = "run_log.txt";

    static LOG: OnceLock<Mutex<Option<BufWriter<File>>>> = OnceLock::new();
    static COMPILATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn log() -> &'static Mutex<Option<BufWriter<File>>> {
        LOG.get_or_init(|| {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(LOG_PATH);

            match file {
                Ok(file) => Mutex::new(Some(BufWriter::new(file))),
                Err(error) => {
                    eprintln!("failed to create {LOG_PATH}: {error}");
                    Mutex::new(None)
                }
            }
        })
    }

    fn with_log(write: impl FnOnce(&mut BufWriter<File>)) {
        let mut guard = log()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(writer) = guard.as_mut() else {
            return;
        };

        write(writer);
        // Runtime debugging often terminates the process immediately after an
        // error. Flush every stage so the latest compiler state survives.
        let _ = writer.flush();
    }

    pub fn next_compilation_id() -> u64 {
        COMPILATION_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn dump_hir(compilation_id: u64, stage: &str, hir: &Program) {
        with_log(|writer| {
            let _ = writeln!(
                writer,
                "\n================================================================================"
            );
            let _ = writeln!(writer, "COMPILATION #{compilation_id}: {stage}");
            let _ = writeln!(
                writer,
                "functions={} modules={} classes={}",
                hir.functions.len(),
                hir.modules.len(),
                hir.classes.len()
            );
            let _ = writeln!(
                writer,
                "================================================================================"
            );

            for (index, function) in hir.functions.iter().enumerate() {
                let _ = writeln!(
                    writer,
                    "\n--- HIR function index={index} id={:?} module={:?} kind={:?} async={} concurrent={} memoization={:?} field_owner={:?} ---",
                    function.id,
                    function.module,
                    function.kind,
                    function.is_async,
                    function.is_concurrent,
                    function.memoization,
                    function.field_owner
                );
                let _ = writeln!(writer, "{function}");
            }
        });
    }

    pub fn dump_bytecode(compilation_id: u64, program: &BytecodeProgram) {
        with_log(|writer| {
            let _ = writeln!(
                writer,
                "\n================================================================================"
            );
            let _ = writeln!(writer, "COMPILATION #{compilation_id}: BYTECODE");
            let _ = writeln!(
                writer,
                "functions={} methods={} strings={}",
                program.functions.len(),
                program.metadata.methods.len(),
                program.metadata.strings.len()
            );
            let _ = writeln!(
                writer,
                "================================================================================"
            );

            let _ = writeln!(writer, "\n--- Runtime method table ---");
            for (index, signature) in program.metadata.methods.iter().enumerate() {
                let name = program
                    .metadata
                    .string(signature.name)
                    .unwrap_or("<invalid-string>");
                let _ = writeln!(
                    writer,
                    "MethodSymbol#{index}: name={name:?} kind={:?} arity={}",
                    signature.kind,
                    signature.arity
                );
            }

            for (index, function) in program.functions.iter().enumerate() {
                let name = program
                    .metadata
                    .string(function.name)
                    .unwrap_or("<invalid-function-name>");

                let _ = writeln!(
                    writer,
                    "\n--------------------------------------------------------------------------------"
                );
                let _ = writeln!(
                    writer,
                    "BYTECODE FUNCTION index={index} id={:?} name={name:?}",
                    function.id
                );
                let _ = writeln!(
                    writer,
                    "module={:?} kind={:?} async={} concurrent={} memoization={:?} field_owner={:?}",
                    function.module,
                    function.kind,
                    function.is_async,
                    function.is_concurrent,
                    function.memoization,
                    function.field_owner
                );
                let _ = writeln!(
                    writer,
                    "arity={} captures={} gprs={} return_kind={:?}",
                    function.arity,
                    function.capture_count,
                    function.frame.general_register_count,
                    function.frame.return_kind
                );
                let _ = writeln!(writer, "frame.parameters = {:#?}", function.frame.parameters);
                let _ = writeln!(writer, "frame.captures = {:#?}", function.frame.captures);

                let _ = writeln!(writer, "constants = {:#?}", function.chunk.constants);

                let _ = writeln!(writer, "inline_caches:");
                for (cache_index, cache) in function.chunk.inline_caches.iter().enumerate() {
                    let signature = program.metadata.method(cache.method);
                    let method_name = signature
                        .and_then(|signature| program.metadata.string(signature.name))
                        .unwrap_or("<invalid-method>");
                    let _ = writeln!(
                        writer,
                        "  [{cache_index}] site={:?} method={:?} name={method_name:?} signature={signature:?} home_class={:?}",
                        cache.site,
                        cache.method,
                        cache.home_class
                    );
                }

                let _ = writeln!(writer, "type_guards = {:#?}", function.chunk.type_guards);
                let _ = writeln!(writer, "class_guards = {:#?}", function.chunk.class_guards);
                let _ = writeln!(writer, "switch_tables = {:#?}", function.chunk.switch_tables);
                let _ = writeln!(writer, "deopts = {:#?}", function.chunk.deopts);
                let _ = writeln!(writer, "safepoints = {:#?}", function.chunk.safepoints);
                let _ = writeln!(writer, "live_registers = {:#?}", function.chunk.live_registers);
                let _ = writeln!(writer, "resume_points = {:#?}", function.chunk.resume_points);
                let _ = writeln!(writer, "source_map = {:#?}", function.chunk.source_map);
                let _ = writeln!(
                    writer,
                    "profile_counters = {:#?}",
                    function.chunk.profile_counters
                );
                let _ = writeln!(
                    writer,
                    "exception_handlers = {:#?}",
                    function.chunk.exception_handlers
                );

                let _ = writeln!(writer, "\nDISASSEMBLY:");
                match disassemble(function) {
                    Ok(disassembly) => {
                        let _ = write!(writer, "{disassembly}");
                    }
                    Err(error) => {
                        let _ = writeln!(writer, "<disassembly failed: {error:?}>");
                    }
                }
            }
        });
    }

    pub fn dump_compile_error(compilation_id: u64, error: &CompileError) {
        with_log(|writer| {
            let _ = writeln!(
                writer,
                "\n================================================================================"
            );
            let _ = writeln!(writer, "COMPILATION #{compilation_id}: BYTECODE COMPILATION FAILED");
            let _ = writeln!(writer, "{error:?}");
            let _ = writeln!(
                writer,
                "================================================================================"
            );
        });
    }
}

pub struct CompilationPipeline {
    optimizations: Optimize,
    bytecode: BytecodeCompiler,
}

impl Default for CompilationPipeline {
    fn default() -> Self {
        Self::new(optimization_passes())
    }
}

impl CompilationPipeline {
    /// Select an ordered set of HIR passes. `Optimize::new()` disables this stage.
    pub fn new(optimizations: Optimize) -> Self {
        Self {
            optimizations,
            bytecode: BytecodeCompiler::default(),
        }
    }

    /// Configure bytecode lowering independently of the HIR optimization passes.
    pub fn with_bytecode_options(mut self, options: CompileOptions) -> Self {
        self.bytecode = BytecodeCompiler::new(options);
        self
    }

    /// Optimize the lowered HIR in place, then generate executable bytecode.
    pub fn compile(&mut self, hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
        Ok(self.compile_with_stats(hir)?.program)
    }

    /// Backend statistics describe the HIR after optimization.
    pub fn compile_with_stats(
        &mut self,
        hir: &mut Program,
    ) -> Result<CompilationOutput, CompileError> {
        #[cfg(debug_assertions)]
        let debug_compilation_id = {
            let id = debug_dump::next_compilation_id();
            debug_dump::dump_hir(id, "HIR BEFORE OPTIMIZATIONS", hir);
            id
        };

        self.optimizations.optimize_program(hir);

        #[cfg(debug_assertions)]
        debug_dump::dump_hir(debug_compilation_id, "HIR AFTER OPTIMIZATIONS", hir);

        let output = self.bytecode.compile_with_stats(hir);

        #[cfg(debug_assertions)]
        match &output {
            Ok(output) => debug_dump::dump_bytecode(debug_compilation_id, &output.program),
            Err(error) => debug_dump::dump_compile_error(debug_compilation_id, error),
        }

        output
    }
}

/// Compile lowered HIR using the language's default optimization passes.
pub fn compile_program(hir: &mut Program) -> Result<BytecodeProgram, CompileError> {
    CompilationPipeline::default().compile(hir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_bytecode::disassemble;
    use crate::napalm_interpreter::{ExecutionResult, Interpreter};
    use crate::napalm_ir::{Builder, Constant, FuncId, FunctionKind, Inst, IntBinOp, Type};
    use crate::napalm_ir_optimizer::{InstCursor, OptimizationPass, OptimizationWIR, WirEditQueue};
    use crate::napalm_ir_optimizing::ConstantFold;

    fn arithmetic_program() -> (Program, FuncId) {
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
        builder.const_i32(99); // Dead value exercises the separate backend DCE option.
        let left = builder.const_i32(20);
        let right = builder.const_i32(22);
        let sum = builder.emit1(
            Inst::IntBin {
                op: IntBinOp::Add,
                left,
                right,
            },
            Type::Int32,
        );
        let result = builder.emit1(Inst::Int32ToNum { value: sum }, Type::Num);
        builder.ret(result);
        let entry = builder.finish();
        (hir, entry)
    }

    fn execute(program: &BytecodeProgram, entry: FuncId) -> f64 {
        let mut vm = Interpreter::new(program).unwrap();
        vm.set_jit_options(None);
        let ExecutionResult::Returned(value) = vm.run_module(entry).unwrap() else {
            panic!("module should return");
        };
        value.as_num().unwrap()
    }

    #[test]
    fn default_pipeline_folds_hir_before_bytecode_generation() {
        let (mut hir, entry) = arithmetic_program();
        // Baseline is raw lowering: the post-bytecode optimizer would fold IAdd
        // too, hiding whether the HIR pipeline did it.
        let baseline = BytecodeCompiler::new(CompileOptions {
            optimize_bytecode: false,
            ..CompileOptions::default()
        })
        .compile(&hir)
        .unwrap();
        assert!(
            disassemble(&baseline.functions[0])
                .unwrap()
                .contains("IAdd")
        );

        let optimized = compile_program(&mut hir).unwrap();
        let function = hir.function(entry).unwrap();
        assert_eq!(function.blocks[0].insts.len(), 2, "constant and return");
        assert!(
            !disassemble(&optimized.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&baseline, entry), 42.0);
        assert_eq!(execute(&optimized, entry), 42.0);
    }

    #[test]
    fn empty_pipeline_preserves_hir_and_backend_configuration() {
        let (mut hir, entry) = arithmetic_program();
        let before = hir.function(entry).unwrap().to_string();
        let output = CompilationPipeline::new(Optimize::new())
            .with_bytecode_options(CompileOptions {
                eliminate_dead_code: false,
                // Inspect unoptimized lowering of the untouched HIR.
                optimize_bytecode: false,
                ..CompileOptions::default()
            })
            .compile_with_stats(&mut hir)
            .unwrap();
        assert_eq!(hir.function(entry).unwrap().to_string(), before);
        assert_eq!(output.stats.functions[0].emitted_hir_nodes, 6);
        assert!(
            disassemble(&output.program.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&output.program, entry), 42.0);
    }

    #[test]
    fn custom_node_and_function_passes_run_in_order_before_codegen() {
        struct ReplaceTwenty;
        impl OptimizationWIR for ReplaceTwenty {
            fn optimize_node(&mut self, cur: &mut InstCursor<'_>, q: &mut WirEditQueue) -> bool {
                if matches!(cur.cur_kind(), Inst::Const(Constant::Int32(20))) {
                    q.replace_kind(cur.cur_id(), Inst::Const(Constant::Int32(21)));
                    return true;
                }
                false
            }
        }

        let (mut hir, entry) = arithmetic_program();
        let passes = Optimize::new()
            .with_pass(OptimizationPass::new(vec![Box::new(ReplaceTwenty)]))
            .with_function_pass(ConstantFold);
        let bytecode = CompilationPipeline::new(passes).compile(&mut hir).unwrap();
        assert!(
            !disassemble(&bytecode.functions[0])
                .unwrap()
                .contains("IAdd")
        );
        assert_eq!(execute(&bytecode, entry), 43.0);
    }
}

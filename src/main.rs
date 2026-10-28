//! Napalm VM executable entry point.
//!
//! Expected layout:
//!
//! ```text
//! src/
//! ├── main.rs
//! ├── napalm_front.rs
//! ├── napalm_builtin.rs
//! ├── napalm_lowerer.rs
//! ├── napalm_ir.rs
//! ├── napalm_bytecode.rs
//! ├── napalm_bytecodecompiler.rs
//! ├── napalm_gc.rs
//! └── napalm_interpreter.rs
//! ```
//!
//! Usage:
//!
//! ```text
//! cargo run -- path/to/program.nap
//! ```
//!
//! The pipeline is:
//!
//! ```text
//! entry source
//!   -> static module graph resolution (@lib / @std / relative)
//!   -> tokenizer/parser for every module
//!   -> ASTs
//!   -> one multi-module dynamic HIR program
//!   -> HIR optimization passes
//!   -> register bytecode
//!   -> interpreter / tracing JIT
//! ```

#![allow(warnings)]

pub mod napalm_builtin;
pub mod napalm_collections;
pub mod napalm_bytecode;
pub mod napalm_bytecodecompiler;
pub mod napalm_front;
pub mod napalm_front_main;
pub mod napalm_gc;
pub mod napalm_ir;
pub mod napalm_lowerer;
pub mod napalm_modules;
pub mod napalm_trace_ir;
pub mod napalm_trace_ir_optimizer;
pub mod napalm_trace_ir_optimizing;
pub mod napalm_interpreter;
pub mod napalm_tracing_jit;
pub mod napalm_ir_optimizer;
pub mod napalm_ir_optimizing;
pub mod napalm_ir_escape;
pub mod napalm_pipeline;

#[cfg(test)]
#[path = "tests/compiler_perf.rs"]
mod compiler_perf;

use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mimalloc::MiMalloc;

use crate::napalm_interpreter::{ExecutionResult, Interpreter};
use crate::napalm_ir::Program;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const EXIT_SUCCESS: u8 = 0;
const EXIT_COMPILE_ERROR: u8 = 1;
const EXIT_USAGE_ERROR: u8 = 2;
const EXIT_IO_ERROR: u8 = 3;
const EXIT_RUNTIME_ERROR: u8 = 4;

fn main() -> ExitCode {
    ExitCode::from(run())
}

fn run() -> u8 {
    let source_path = match parse_command_line() {
        Ok(path) => path,
        Err(()) => return EXIT_USAGE_ERROR,
    };

    if cfg!(debug_assertions) {
        eprintln!("Debug build: use `cargo run --release -- <file.nap>` for performance measurements.");
    }

    // -------------------------------------------------------------------------
    // Module graph: resolve -> parse -> lower every statically named module.
    // -------------------------------------------------------------------------

    let mut hir = Program::new();
    let builtin_manifest = napalm_builtin::install_hir_builtins(&mut hir);
    let modules = match napalm_modules::compile_module_graph(&mut hir, &source_path) {
        Ok(modules) => modules,
        Err(error) => {
            if let Err(render_error) = error.emit() {
                eprintln!("error: failed to render module diagnostics: {render_error}");
                return EXIT_IO_ERROR;
            }
            return match error.kind() {
                napalm_modules::ModuleErrorKind::Io => EXIT_IO_ERROR,
                napalm_modules::ModuleErrorKind::Compile => EXIT_COMPILE_ERROR,
            };
        }
    };

    if let Err(error) = modules.emit_diagnostics() {
        eprintln!("error: failed to render module diagnostics: {error}");
        return EXIT_IO_ERROR;
    }

    let lowered_module = modules.entry;
    let module_name = modules.entry_name.clone();

    // -------------------------------------------------------------------------
    // Compilation pipeline: HIR -> optimizations -> register bytecode
    // -------------------------------------------------------------------------

    let bytecode = match napalm_pipeline::compile_program(&mut hir) {
        Ok(program) => program,
        Err(error) => {
            eprintln!("bytecode compilation error: {error}");
            return EXIT_COMPILE_ERROR;
        }
    };

    // -------------------------------------------------------------------------
    // Runtime: bytecode -> execution
    // -------------------------------------------------------------------------

    let host = Box::new(napalm_builtin::BuiltinHost::new(builtin_manifest.clone()));
    let mut interpreter = match Interpreter::with_host(&bytecode, host) {
        Ok(interpreter) => interpreter,
        Err(error) => {
            eprintln!("VM initialization error: {error}");
            return EXIT_RUNTIME_ERROR;
        }
    };
    napalm_builtin::install_vm_builtins(&mut interpreter, &builtin_manifest);

    // Diagnostics useful for comparing the threaded baseline against the tracing
    // tier without recompiling the VM. They are intentionally opt-in so normal
    // execution has no environment lookup in the hot path.
    if env::var_os("NAPALM_NO_JIT").is_some() || env::var_os("NAPALM_NO_JIT").is_some() {
        interpreter.set_jit_options(None);
    }

    let execution = interpreter.run_module(lowered_module.entry);

    if env::var_os("NAPALM_JIT_STATS").is_some() || env::var_os("NAPALM_JIT_STATS").is_some() {
        eprintln!("Napalm JIT stats: {:#?}", interpreter.jit_stats());
    }
    if env::var_os("NAPALM_GC_STATS").is_some() || env::var_os("NAPALM_GC_STATS").is_some() {
        eprintln!("Napalm GC stats: {:#?}", interpreter.gc_statistics());
    }

    match execution {
        Ok(ExecutionResult::Returned(_)) => EXIT_SUCCESS,

        Ok(ExecutionResult::Suspended(_)) => {
            eprintln!(
                "runtime error: top-level module '{}' suspended instead of returning",
                module_name
            );
            EXIT_RUNTIME_ERROR
        }

        Err(error) => {
            eprintln!("runtime error: {error}");
            EXIT_RUNTIME_ERROR
        }
    }
}

fn parse_command_line() -> Result<PathBuf, ()> {
    let mut args = env::args_os();
    let executable = args
        .next()
        .unwrap_or_else(|| OsStr::new("napalm").to_owned());

    let executable = Path::new(&executable)
        .file_name()
        .unwrap_or_else(|| OsStr::new("napalm"))
        .to_string_lossy();

    let Some(first) = args.next() else {
        print_usage(&executable);
        return Err(());
    };

    if first == OsStr::new("-h") || first == OsStr::new("--help") {
        print_usage(&executable);
        return Err(());
    }

    if let Some(extra) = args.next() {
        eprintln!(
            "error: unexpected extra argument '{}'",
            extra.to_string_lossy()
        );
        print_usage(&executable);
        return Err(());
    }

    Ok(PathBuf::from(first))
}

fn print_usage(executable: &str) {
    eprintln!("Usage: {executable} <file.nap>");
}

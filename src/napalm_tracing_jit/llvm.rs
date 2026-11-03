//! LLVM top tier for hot native traces.
//!
//! Tiering:
//!
//! ```text
//!   interpreter --(hot_loop/entry_threshold visits)--> Trace IR -> optimizer -> Cranelift
//!   Cranelift trace --(llvm_hot_threshold native visits)--> same optimized Trace IR -> LLVM IR -> O3 -> MCJIT
//! ```
//!
//! The LLVM function is an ABI-identical drop-in replacement for the Cranelift
//! function: `i32 (ptr %activation, ptr %links)`, same side-exit IDs, same
//! snapshot plans, same VM-binding table, same link-table protocol. Installing
//! it only swaps the trace's entry in the dense dispatcher table, so stitched
//! side traces, Rust-side exit handling and the native dispatcher keep working
//! untouched, and Cranelift code stays as the fallback if LLVM fails.
//!
//! The lowering emits textual LLVM IR. It mirrors `CraneliftLoweringState`
//! exactly (same guards, same epoch invalidation points), but annotates memory
//! with TBAA classes, side-exit branches with cold branch weights, pure helpers
//! with `memory(none)`, and leaves CSE/LICM of invariant VM loads to LLVM.
//!
//! The maximal optimization pipeline is deliberately staged: cheap canonicalization
//! first, LLVM O3 for whole-function/loop work, then a final SSA cleanup. Integer
//! guards and constant floating-point remainders are specialized in the lowering.
//! Floating-point reassociation is never enabled implicitly: the runtime requires
//! correct NaN, infinity and signed-zero behavior. FMA contraction is opt-in.

use super::*;
use std::fmt::Write as _;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::OnceLock;

use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple};
use inkwell::OptimizationLevel;

// =============================================================================
// Host / LLVM process-wide initialization
// =============================================================================

pub(super) struct HostInfo {
    triple: String,
    data_layout: String,
    cpu: String,
    features: String,
}

unsafe extern "C" fn napalm_llvm_memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
    dst
}
unsafe extern "C" fn napalm_llvm_memmove(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    unsafe { std::ptr::copy(src, dst, len) };
    dst
}
unsafe extern "C" fn napalm_llvm_memset(dst: *mut u8, value: i32, len: usize) -> *mut u8 {
    unsafe { std::ptr::write_bytes(dst, value as u8, len) };
    dst
}
extern "C" fn napalm_llvm_floor(value: f64) -> f64 { value.floor() }
extern "C" fn napalm_llvm_ceil(value: f64) -> f64 { value.ceil() }
extern "C" fn napalm_llvm_trunc(value: f64) -> f64 { value.trunc() }
extern "C" fn napalm_llvm_sqrt(value: f64) -> f64 { value.sqrt() }

#[cfg(all(windows, target_env = "msvc"))]
unsafe extern "C" {
    fn __chkstk();
}

fn llvm_symbol_table() -> Vec<(&'static str, usize)> {
    let mut symbols: Vec<(&'static str, usize)> = vec![
        ("napalm_jit_frem", native_frem as usize),
        ("napalm_jit_round", native_round as usize),
        ("napalm_jit_sin", native_sin as usize),
        ("napalm_jit_cos", native_cos as usize),
        ("napalm_jit_tan", native_tan as usize),
        ("napalm_jit_log", native_log as usize),
        ("napalm_jit_exp", native_exp as usize),
        ("napalm_jit_decode_code_point", native_decode_napalm_code_point as usize),
        // Libcalls the code generator may synthesize (loop idioms, legalized
        // rounding without SSE4.1, f64 remainder). MCJIT aborts the process on
        // an unresolved symbol, so every plausible one is mapped explicitly.
        ("memcpy", napalm_llvm_memcpy as usize),
        ("memmove", napalm_llvm_memmove as usize),
        ("memset", napalm_llvm_memset as usize),
        ("fmod", native_frem as usize),
        ("round", native_round as usize),
        ("floor", napalm_llvm_floor as usize),
        ("ceil", napalm_llvm_ceil as usize),
        ("trunc", napalm_llvm_trunc as usize),
        ("sqrt", napalm_llvm_sqrt as usize),
        ("sin", native_sin as usize),
        ("cos", native_cos as usize),
        ("tan", native_tan as usize),
        ("log", native_log as usize),
        ("exp", native_exp as usize),
    ];
    #[cfg(all(windows, target_env = "msvc"))]
    symbols.push(("__chkstk", __chkstk as usize));
    symbols
}

/// Windows links `LLVM-C.dll` with /DELAYLOAD. Load it explicitly (executable
/// directory/PATH first, then the build-time LLVM prefix) before any LLVM call;
/// once a module named LLVM-C.dll is loaded the delay-load helper reuses it.
#[cfg(windows)]
fn load_llvm_library() -> Result<(), String> {
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
    }
    let wide = |text: &std::ffi::OsStr| {
        use std::os::windows::ffi::OsStrExt;
        text.encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>()
    };
    let mut candidates: Vec<std::ffi::OsString> = Vec::new();
    if let Some(explicit) = std::env::var_os("NAPALM_LLVM_DLL") {
        candidates.push(explicit);
    }
    // The bundled copy shipped next to napalm.exe always wins over PATH, so a
    // different LLVM install on the user's machine can never be picked up.
    if let Some(bundled) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("LLVM-C.dll")))
        .filter(|path| path.is_file())
    {
        candidates.push(bundled.into_os_string());
    }
    candidates.push("LLVM-C.dll".into());
    if let Some(built) = option_env!("NAPALM_LLVM_DLL") {
        candidates.push(built.into());
    }
    for candidate in &candidates {
        if !unsafe { LoadLibraryW(wide(candidate).as_ptr()) }.is_null() {
            if jit_events_enabled() {
                eprintln!("[napalm-llvm] loaded {}", candidate.to_string_lossy());
            }
            return Ok(());
        }
    }
    Err(format!("LLVM-C.dll not found (tried {candidates:?}); the LLVM tier is disabled"))
}

#[cfg(not(windows))]
fn load_llvm_library() -> Result<(), String> {
    Ok(())
}

fn initialize_llvm() -> Result<(), String> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        LLVM_LOADED.store(true, Ordering::Release);
        load_llvm_library()?;
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|error| format!("cannot initialize LLVM native target: {error}"))?;
        for (name, address) in llvm_symbol_table() {
            let name = std::ffi::CString::new(name).expect("symbol name");
            unsafe {
                inkwell::llvm_sys::support::LLVMAddSymbol(name.as_ptr(), address as *mut c_void);
            }
        }
        // SLP is opt-in (`NAPALM_LLVM_SLP=1`). Its cost model ignores loop-carried
        // latency: on trace loops it packs a cheap induction (`i + 1`) with a long
        // accumulator chain into one vector op, serializing the loop behind the
        // divisions (measured 5x slower). PipelineTuningOptions alone did not
        // keep `default<O3>` from scheduling it, so use LLVM's own switch.
        let slp = std::env::var_os("NAPALM_LLVM_SLP").is_some();
        let args = [
            std::ffi::CString::new("napalm-llvm").unwrap(),
            std::ffi::CString::new(if slp { "-vectorize-slp=true" } else { "-vectorize-slp=false" }).unwrap(),
        ];
        let argv: Vec<*const std::ffi::c_char> = args.iter().map(|arg| arg.as_ptr()).collect();
        let overview = std::ffi::CString::new("Napalm LLVM trace tier").unwrap();
        unsafe {
            inkwell::llvm_sys::support::LLVMParseCommandLineOptions(
                argv.len() as i32,
                argv.as_ptr(),
                overview.as_ptr(),
            );
        }
        Ok(())
    })
    .clone()
}

fn jit_triple() -> String {
    let default = TargetMachine::get_default_triple();
    let triple = default.as_str().to_string_lossy().into_owned();
    // MCJIT's COFF loader is incomplete; ELF objects on a Windows triple keep the
    // Win64 calling convention while using the mature ELF RuntimeDyld path.
    if cfg!(windows) && !triple.ends_with("-elf") {
        format!("{triple}-elf")
    } else {
        triple
    }
}

fn create_host_target_machine(host: &HostInfo) -> Result<TargetMachine, String> {
    let triple = TargetTriple::create(&host.triple);
    let target = Target::from_triple(&triple).map_err(|error| format!("LLVM target lookup failed: {error}"))?;
    target
        .create_target_machine(
            &triple,
            &host.cpu,
            &host.features,
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| "LLVM could not create a host TargetMachine".to_owned())
}

pub(super) fn host_info() -> Result<&'static HostInfo, String> {
    static HOST: OnceLock<Result<HostInfo, String>> = OnceLock::new();
    HOST.get_or_init(|| {
        let started = std::time::Instant::now();
        initialize_llvm()?;
        if jit_events_enabled() {
            eprintln!("[napalm-llvm] LLVM loaded and initialized in {} us", started.elapsed().as_micros());
        }
        let mut host = HostInfo {
            triple: jit_triple(),
            data_layout: String::new(),
            cpu: TargetMachine::get_host_cpu_name().to_string(),
            features: TargetMachine::get_host_cpu_features().to_string(),
        };
        let machine = create_host_target_machine(&host)?;
        host.data_layout = machine
            .get_target_data()
            .get_data_layout()
            .as_str()
            .to_string_lossy()
            .into_owned();
        Ok(host)
    })
    .as_ref()
    .map_err(Clone::clone)
}

// =============================================================================
// Background compiler service
// =============================================================================

/// Everything a worker needs to lower and compile one trace. The manager only
/// clones data: the interpreter thread never loads LLVM, lowers or optimizes.
pub(super) struct LlvmJob {
    pub(super) trace_id: TraceId,
    pub(super) ticket: u64,
    pub(super) symbol: String,
    pub(super) trace: Trace,
    pub(super) exits: Vec<CompiledExitPlan>,
    pub(super) binding_slots: HashMap<VmBindingKey, u32>,
    pub(super) native_budget: u32,
    pub(super) pipeline: String,
    pub(super) slp_vectorize: bool,
    pub(super) dump: bool,
}

pub(super) struct LlvmDone {
    pub(super) trace: TraceId,
    pub(super) ticket: u64,
    /// LLVM itself could not be loaded or initialized; stop requesting the tier.
    pub(super) unavailable: bool,
    pub(super) result: Result<LlvmCode, String>,
}

/// Executables set this before tearing down the VM: compiler workers are then
/// detached instead of joined, and `finish_process` exits without running the
/// LLVM DLL's teardown (so a worker killed mid-compile cannot deadlock it).
static DETACH_WORKERS_ON_DROP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// LLVM has been (or is being) loaded into this process.
static LLVM_LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(super) fn set_detached_shutdown(detach: bool) {
    DETACH_WORKERS_ON_DROP.store(detach, Ordering::Release);
}

/// Terminate the process immediately once LLVM is loaded, skipping DLL/static
/// teardown (clang itself exits this way). Callers must already have dropped
/// the VM so every user-visible buffer is flushed.
pub(super) fn finish_process(code: u8) -> u8 {
    if !LLVM_LOADED.load(Ordering::Acquire) {
        return code;
    }
    use std::io::Write as _;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn TerminateProcess(process: *mut c_void, code: u32) -> i32;
        }
        TerminateProcess(GetCurrentProcess(), code as u32);
    }
    #[cfg(unix)]
    unsafe {
        unsafe extern "C" {
            fn _exit(code: i32) -> !;
        }
        _exit(code as i32);
    }
    #[allow(unreachable_code)]
    code
}

pub(super) struct LlvmCode {
    pub(super) address: usize,
    pub(super) micros: u64,
    pub(super) optimized_ir: Option<String>,
}

/// Pool of LLVM compiler workers owned by one TracingJit. Each worker owns its
/// own LLVM context and execution engines (LLVM contexts are single-threaded),
/// so generated code lives exactly as long as this service. Workers are spawned
/// lazily as the queue grows, up to `max_workers`.
pub(super) struct LlvmService {
    jobs: Option<Sender<LlvmJob>>,
    queue: Arc<Mutex<Receiver<LlvmJob>>>,
    done_tx: Sender<LlvmDone>,
    done: Receiver<LlvmDone>,
    /// Set by a worker after each result; read by the native dispatcher
    /// (`TraceActivation::tier_up_pending`), cleared by the manager.
    ready: Arc<AtomicU32>,
    /// Set on drop: queued jobs are skipped.
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Jobs submitted and not yet answered.
    pending: Arc<AtomicUsize>,
    workers: Vec<std::thread::JoinHandle<()>>,
    max_workers: usize,
}

impl LlvmService {
    pub(super) fn spawn() -> Result<Self, String> {
        let (jobs_tx, jobs_rx) = std::sync::mpsc::channel::<LlvmJob>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<LlvmDone>();
        let max_workers = std::env::var("NAPALM_LLVM_THREADS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, |cores| (cores.get() / 2).clamp(1, 4))
            })
            .max(1);
        Ok(Self {
            jobs: Some(jobs_tx),
            queue: Arc::new(Mutex::new(jobs_rx)),
            done_tx,
            done: done_rx,
            ready: Arc::new(AtomicU32::new(0)),
            cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            pending: Arc::new(AtomicUsize::new(0)),
            workers: Vec::new(),
            max_workers,
        })
    }

    fn spawn_worker(&mut self) -> Result<(), String> {
        let (queue, done, ready, cancel, pending) = (
            self.queue.clone(),
            self.done_tx.clone(),
            self.ready.clone(),
            self.cancel.clone(),
            self.pending.clone(),
        );
        let worker = std::thread::Builder::new()
            .name(format!("napalm-llvm-jit-{}", self.workers.len()))
            .stack_size(16 << 20)
            .spawn(move || llvm_worker_main(queue, done, ready, cancel, pending))
            .map_err(|error| format!("cannot spawn LLVM compiler thread: {error}"))?;
        self.workers.push(worker);
        Ok(())
    }

    pub(super) fn ready_flag(&self) -> *const u32 {
        self.ready.as_ptr() as *const u32
    }

    pub(super) fn clear_ready(&self) {
        self.ready.store(0, Ordering::Release);
    }

    pub(super) fn submit(&mut self, job: LlvmJob) -> Result<(), String> {
        let outstanding = self.pending.fetch_add(1, Ordering::AcqRel) + 1;
        let sent = self
            .jobs
            .as_ref()
            .ok_or_else(|| "LLVM compiler service stopped".to_owned())
            .and_then(|jobs| jobs.send(job).map_err(|_| "LLVM compiler threads terminated".to_owned()));
        if let Err(error) = sent {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            return Err(error);
        }
        // Grow the pool while every worker already has a job.
        if self.workers.is_empty() || (outstanding > self.workers.len() && self.workers.len() < self.max_workers) {
            if let Err(error) = self.spawn_worker() {
                if self.workers.is_empty() {
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    pub(super) fn try_recv(&self) -> Option<LlvmDone> {
        self.done.try_recv().ok()
    }

    pub(super) fn recv(&self) -> Option<LlvmDone> {
        self.done.recv().ok()
    }
}

impl Drop for LlvmService {
    fn drop(&mut self) {
        // Closing the job channel stops the workers, which then free their
        // engines. The owning TracingJit is being dropped, so no native code can
        // still reference LLVM-generated functions. Libraries/tests join, so a
        // worker is never killed inside LLVM; executables opt into detaching and
        // then leave through `finish_process`, which skips LLVM's teardown.
        self.cancel.store(true, Ordering::Release);
        self.jobs.take();
        let workers = std::mem::take(&mut self.workers);
        if !DETACH_WORKERS_ON_DROP.load(Ordering::Acquire) {
            for worker in workers {
                let _ = worker.join();
            }
        }
    }
}

fn llvm_worker_main(
    queue: Arc<Mutex<Receiver<LlvmJob>>>,
    done: Sender<LlvmDone>,
    ready: Arc<AtomicU32>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    pending: Arc<AtomicUsize>,
) {
    // LLVM is loaded here, never on the interpreter thread.
    let setup = host_info().and_then(|host| Ok((host, create_host_target_machine(host)?)));
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    {
        let mut engines: Vec<(Module<'static>, ExecutionEngine<'static>)> = Vec::new();
        loop {
            let job = queue.lock().recv();
            let Ok(job) = job else { break; };
            if cancel.load(Ordering::Acquire) {
                pending.fetch_sub(1, Ordering::AcqRel);
                break;
            }
            let (result, unavailable) = match &setup {
                Ok((host, machine)) => (compile_job(context, host, machine, &job, &mut engines), false),
                Err(error) => (Err(error.clone()), true),
            };
            pending.fetch_sub(1, Ordering::AcqRel);
            if done
                .send(LlvmDone { trace: job.trace_id, ticket: job.ticket, unavailable, result })
                .is_err()
            {
                break;
            }
            ready.store(1, Ordering::Release);
        }
        drop(engines);
    }
    // Every module/engine borrowing the context is gone.
    unsafe { drop(Box::from_raw(context as *const Context as *mut Context)) };
}

/// Final cleanup after O3 is useful for a trace: a large number of predicted-cold
/// side-exit blocks and scalar-replacement temporaries are exposed by inlining,
/// loop rotation, and unrolling in the default optimizer. Never run O3 twice:
/// the LLVM documentation warns that repeated O3 can grow code excessively.
///
/// A nondefault pipeline is owned by the caller and is left completely intact.
fn effective_optimization_pipeline(requested: &str) -> String {
    if requested != "default<O3>" {
        return requested.to_owned();
    }

    // These are standard new-PM passes (LLVM 19+). The first phase maximizes
    // SROA/CFG folding opportunities before vectorization and loop transforms.
    // After O3, finish with a second scalar cleanup to remove instructions
    // made redundant by loop and CFG transformations.
    "function(sroa,early-cse,instcombine,simplifycfg),default<O3>,function(aggressive-instcombine,gvn,dse,instcombine,simplifycfg)".to_owned()
}

fn compile_job(
    context: &'static Context,
    host: &HostInfo,
    machine: &TargetMachine,
    job: &LlvmJob,
    engines: &mut Vec<(Module<'static>, ExecutionEngine<'static>)>,
) -> Result<LlvmCode, String> {
    let started = std::time::Instant::now();
    let ir = lower_trace(&job.trace, &job.exits, job.binding_slots.clone(), job.native_budget, &job.symbol, host)?;
    if job.dump {
        eprintln!("[napalm-llvm] generated {}:\n{ir}", job.symbol);
    }
    compile_ir(context, machine, &job.symbol, &ir, &job.pipeline, job.slp_vectorize, job.dump, engines, started)
}

fn compile_ir(
    context: &'static Context,
    machine: &TargetMachine,
    symbol: &str,
    ir: &str,
    pipeline: &str,
    slp_vectorize: bool,
    dump: bool,
    engines: &mut Vec<(Module<'static>, ExecutionEngine<'static>)>,
    started: std::time::Instant,
) -> Result<LlvmCode, String> {
    let lowered = started.elapsed();
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), symbol);
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| format!("LLVM IR parse error: {error}"))?;
    module.verify().map_err(|error| format!("LLVM IR verification failed: {error}"))?;
    let parsed = started.elapsed();

    let pipeline = effective_optimization_pipeline(pipeline);
    let options = PassBuilderOptions::create();
    options.set_verify_each(cfg!(debug_assertions));
    options.set_loop_interleaving(true);
    options.set_loop_vectorization(true);
    // See initialize_llvm: SLP is additionally gated by `-vectorize-slp`.
    options.set_loop_slp_vectorization(slp_vectorize);
    options.set_loop_unrolling(true);
    options.set_merge_functions(true);
    module
        .run_passes(&pipeline, machine, options)
        .map_err(|error| format!("LLVM pass pipeline `{pipeline}` failed: {error}"))?;
    // Catch invalid IR after the custom passes, before MCJIT code generation.
    if cfg!(debug_assertions) {
        module.verify().map_err(|error| format!("post-optimization LLVM verification failed: {error}"))?;
    }
    let optimized = started.elapsed();

    let optimized_ir = dump.then(|| module.print_to_string().to_string());
    if let Some(text) = &optimized_ir {
        eprintln!("[napalm-llvm] optimized {symbol}:\n{text}");
    }

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|error| format!("cannot create LLVM MCJIT engine: {error}"))?;
    let engine_ready = started.elapsed();
    let address = engine
        .get_function_address(symbol)
        .map_err(|error| format!("LLVM did not produce {symbol}: {error}"))?;
    if address == 0 {
        return Err(format!("LLVM produced a null entry for {symbol}"));
    }
    if jit_events_enabled() {
        eprintln!(
            "[napalm-llvm] {symbol} lower {} us, parse+verify {} us, O3 {} us, engine {} us, codegen {} us (ir {} bytes)",
            lowered.as_micros(),
            (parsed - lowered).as_micros(),
            (optimized - parsed).as_micros(),
            (engine_ready - optimized).as_micros(),
            (started.elapsed() - engine_ready).as_micros(),
            ir.len()
        );
    }
    engines.push((module, engine));
    Ok(LlvmCode {
        address,
        micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
        optimized_ir,
    })
}

// =============================================================================
// Trace IR -> LLVM IR
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ty {
    I1,
    I8,
    I16,
    I32,
    I64,
    F64,
    Ptr,
}

impl Ty {
    #[inline]
    fn ir(self) -> &'static str {
        match self {
            Ty::I1 => "i1",
            Ty::I8 => "i8",
            Ty::I16 => "i16",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F64 => "double",
            Ty::Ptr => "ptr",
        }
    }

    #[inline]
    fn align(self) -> u32 {
        match self {
            Ty::I1 | Ty::I8 => 1,
            Ty::I16 => 2,
            Ty::I32 => 4,
            Ty::I64 | Ty::F64 | Ty::Ptr => 8,
        }
    }

    fn of_rep(rep: TraceRep) -> Ty {
        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => Ty::I64,
            TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => Ty::I32,
            TraceRep::F64 => Ty::F64,
        }
    }
}

/// An LLVM operand: an SSA name (`%t3`) or an inline constant (`42`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct V {
    op: String,
    ty: Ty,
}

impl V {
    fn i64(value: i64) -> V { V { op: value.to_string(), ty: Ty::I64 } }
    fn i32(value: i32) -> V { V { op: value.to_string(), ty: Ty::I32 } }
    fn f64_bits(bits: u64) -> V { V { op: format!("0x{bits:016X}"), ty: Ty::F64 } }
    fn f64(value: f64) -> V { V::f64_bits(value.to_bits()) }
}

/// TBAA classes. Two accesses with different classes never touch the same bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mem {
    /// `TraceActivation` fields.
    Act,
    /// Architectural register banks.
    Reg,
    /// `TraceHeapEntry` sidecars and the handle chunk table.
    Sidecar,
    /// GC payload words, cells, VM Value slots and profile counters.
    Heap,
    /// Immutable string bytes.
    Str,
    /// Per-trace VM relocation table (invariant for one native entry).
    Binding,
    /// Per-trace child-link table.
    Links,
}

impl Mem {
    fn tag(self) -> u32 {
        match self {
            Mem::Act => 11,
            Mem::Reg => 12,
            Mem::Sidecar => 13,
            Mem::Heap => 14,
            Mem::Str => 15,
            Mem::Binding => 16,
            Mem::Links => 17,
        }
    }
}

const MD_COLD_TRUE: u32 = 20;
const MD_HOT_TRUE: u32 = 21;
const MD_EMPTY: u32 = 22;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Fwd {
    Field { entry: V, slot: u32 },
    Cell { entry: V },
    Indexed { entry: V, index: V },
    ShapedMap { entry: V, slot: u32 },
}

struct LlvmLowering<'p> {
    trace: &'p Trace,
    exit_plans: &'p [CompiledExitPlan],
    entry: String,
    body: String,
    tmp: u32,
    labels: u32,
    current: String,
    values: Vec<Option<V>>,
    heap_addresses: Vec<Option<V>>,
    heap_entries: HashMap<(TraceValueId, u32), (V, V)>,
    heap_live_entries: HashMap<TraceValueId, (V, V)>,
    heap_data: HashMap<V, V>,
    heap_len: HashMap<V, V>,
    heap_metadata: HashMap<V, V>,
    heap_receiver_class: HashMap<V, V>,
    list_indices: HashMap<(TraceValueId, TraceValueId), V>,
    list_lengths: HashMap<TraceValueId, TraceValueId>,
    bounds_proofs: HashSet<(TraceValueId, TraceValueId)>,
    constant_index_proofs: HashSet<(V, u32)>,
    map_shape_proofs: HashSet<(TraceValueId, u32)>,
    string_key_proofs: HashSet<(TraceValueId, Vec<u8>)>,
    shared_constant_exit_blocks: HashMap<(u32, Option<V>), String>,
    forwarded_heap_loads: HashMap<Fwd, V>,
    forwarded_absolute_loads: HashMap<u64, V>,
    vm_binding_slots: HashMap<VmBindingKey, u32>,
    vm_binding_values: HashMap<u32, V>,
    vm_bindings_base: Option<V>,
    runtime_context: Option<V>,
    execute_callee: Option<V>,
    heap_barrier_callee: Option<V>,
    method_epoch_value: Option<V>,
    native_budget_remaining: Option<V>,
    native_budget_initial: u32,
    /// Preserve IEEE arithmetic unless the application explicitly accepts
    /// the different rounding that contracting multiply-add can introduce.
    allow_fp_contract: bool,
}

/// Lower one already-optimized trace. `vm_binding_slots` must describe the
/// relocation table the installed trace already uses.
pub(super) fn lower_trace(
    trace: &Trace,
    exit_plans: &[CompiledExitPlan],
    vm_binding_slots: HashMap<VmBindingKey, u32>,
    native_budget: u32,
    symbol: &str,
    host: &HostInfo,
) -> Result<String, String> {
    let mut state = LlvmLowering {
        trace,
        exit_plans,
        entry: String::with_capacity(1024),
        body: String::with_capacity(trace.nodes.len() * 160 + 1024),
        tmp: 0,
        labels: 0,
        current: "start".to_owned(),
        values: vec![None; trace.values.len()],
        heap_addresses: vec![None; trace.values.len()],
        heap_entries: HashMap::default(),
        heap_live_entries: HashMap::default(),
        heap_data: HashMap::default(),
        heap_len: HashMap::default(),
        heap_metadata: HashMap::default(),
        heap_receiver_class: HashMap::default(),
        list_indices: HashMap::default(),
        list_lengths: HashMap::default(),
        bounds_proofs: HashSet::default(),
        constant_index_proofs: HashSet::default(),
        map_shape_proofs: HashSet::default(),
        string_key_proofs: HashSet::default(),
        shared_constant_exit_blocks: HashMap::default(),
        forwarded_heap_loads: HashMap::default(),
        forwarded_absolute_loads: HashMap::default(),
        vm_binding_slots,
        vm_binding_values: HashMap::default(),
        vm_bindings_base: None,
        runtime_context: None,
        execute_callee: None,
        heap_barrier_callee: None,
        method_epoch_value: None,
        native_budget_remaining: None,
        native_budget_initial: native_budget,
        allow_fp_contract: std::env::var("NAPALM_LLVM_ALLOW_FMA").ok().as_deref() == Some("1"),
    };
    state.lower_function(native_budget)?;
    Ok(state.finish_module(symbol, host))
}

impl LlvmLowering<'_> {
    // ---------------------------------------------------------------- emission

    fn fresh(&mut self) -> String {
        let name = format!("%t{}", self.tmp);
        self.tmp += 1;
        name
    }

    fn new_block(&mut self) -> String {
        let label = format!("b{}", self.labels);
        self.labels += 1;
        label
    }

    fn begin(&mut self, label: &str) {
        let _ = writeln!(self.body, "{label}:");
        self.current = label.to_owned();
    }

    fn line(&mut self, text: &str) {
        self.body.push_str("  ");
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn def(&mut self, ty: Ty, rhs: &str) -> V {
        let name = self.fresh();
        let _ = writeln!(self.body, "  {name} = {rhs}");
        V { op: name, ty }
    }

    fn def_entry(&mut self, ty: Ty, rhs: &str) -> V {
        let name = self.fresh();
        let _ = writeln!(self.entry, "  {name} = {rhs}");
        V { op: name, ty }
    }

    fn br(&mut self, target: &str) {
        self.line(&format!("br label %{target}"));
    }

    /// Branch with `cold` placed on the `if_true` successor when `true_is_cold`.
    fn cond_br(&mut self, condition: &V, if_true: &str, if_false: &str, weights: Option<u32>) {
        let condition = self.cond(condition);
        let prof = weights.map(|md| format!(", !prof !{md}")).unwrap_or_default();
        self.line(&format!("br i1 {}, label %{if_true}, label %{if_false}{prof}", condition.op));
    }

    fn ret(&mut self, code: u32) {
        self.line(&format!("ret i32 {code}"));
    }

    fn error(&self, message: impl Into<String>) -> String {
        format!("LLVM lowering of trace {:?}: {}", self.trace.id, message.into())
    }

    // --------------------------------------------------------------- operations

    fn bin(&mut self, op: &str, lhs: &V, rhs: &V) -> V {
        let ty = lhs.ty;
        // `contract` is substantially safer than `fast`: it does not assert
        // no-NaNs, no-infinities, no-signed-zeros or allow reassociation.
        // Contraction may nevertheless change the final rounding, so only use
        // it when the application explicitly permits that semantic difference.
        let flags = if self.allow_fp_contract
            && ty == Ty::F64
            && matches!(op, "fadd" | "fsub" | "fmul")
        {
            " contract"
        } else {
            ""
        };
        self.def(ty, &format!("{op}{flags} {} {}, {}", ty.ir(), lhs.op, rhs.op))
    }

    fn icmp(&mut self, pred: &str, lhs: &V, rhs: &V) -> V {
        let (lhs, rhs) = self.same_int(lhs, rhs);
        self.def(Ty::I1, &format!("icmp {pred} {} {}, {}", lhs.ty.ir(), lhs.op, rhs.op))
    }

    fn fcmp(&mut self, pred: &str, lhs: &V, rhs: &V) -> V {
        self.def(Ty::I1, &format!("fcmp {pred} double {}, {}", lhs.op, rhs.op))
    }

    /// Make two integer-ish operands comparable (pointer -> i64 when mixed).
    fn same_int(&mut self, lhs: &V, rhs: &V) -> (V, V) {
        if lhs.ty == rhs.ty {
            return (lhs.clone(), rhs.clone());
        }
        (self.word(lhs), self.word(rhs))
    }

    fn cast(&mut self, op: &str, value: &V, to: Ty) -> V {
        if value.ty == to {
            return value.clone();
        }
        self.def(to, &format!("{op} {} {} to {}", value.ty.ir(), value.op, to.ir()))
    }

    fn zext(&mut self, value: &V, to: Ty) -> V { self.cast("zext", value, to) }
    fn sext(&mut self, value: &V, to: Ty) -> V { self.cast("sext", value, to) }
    fn trunc(&mut self, value: &V, to: Ty) -> V { self.cast("trunc", value, to) }

    fn bitcast(&mut self, value: &V, to: Ty) -> V {
        if value.ty == to {
            return value.clone();
        }
        if value.ty == Ty::Ptr {
            let word = self.word(value);
            return self.cast("bitcast", &word, to);
        }
        self.cast("bitcast", value, to)
    }

    fn select(&mut self, condition: &V, if_true: &V, if_false: &V) -> V {
        let condition = self.cond(condition);
        let (if_true, if_false) = if if_true.ty == if_false.ty {
            (if_true.clone(), if_false.clone())
        } else {
            (self.word(if_true), self.word(if_false))
        };
        self.def(
            if_true.ty,
            &format!("select i1 {}, {} {}, {} {}", condition.op, if_true.ty.ir(), if_true.op, if_true.ty.ir(), if_false.op),
        )
    }

    /// Pointer -> i64 (word view).
    fn word(&mut self, value: &V) -> V {
        if value.ty == Ty::Ptr {
            self.def(Ty::I64, &format!("ptrtoint ptr {} to i64", value.op))
        } else {
            value.clone()
        }
    }

    fn as_ptr(&mut self, value: &V) -> V {
        match value.ty {
            Ty::Ptr => value.clone(),
            Ty::I64 => self.def(Ty::Ptr, &format!("inttoptr i64 {} to ptr", value.op)),
            other => {
                let wide = self.zext(value, Ty::I64);
                debug_assert!(other != Ty::F64);
                self.def(Ty::Ptr, &format!("inttoptr i64 {} to ptr", wide.op))
            }
        }
    }

    /// Any boolean-ish value -> i1.
    fn cond(&mut self, value: &V) -> V {
        match value.ty {
            Ty::I1 => value.clone(),
            _ => {
                let zero = V { op: "0".to_owned(), ty: value.ty };
                self.icmp("ne", value, &zero)
            }
        }
    }

    fn not1(&mut self, value: &V) -> V {
        let value = self.cond(value);
        self.def(Ty::I1, &format!("xor i1 {}, true", value.op))
    }

    fn and1(&mut self, lhs: &V, rhs: &V) -> V {
        let lhs = self.cond(lhs);
        let rhs = self.cond(rhs);
        self.bin("and", &lhs, &rhs)
    }

    fn or1(&mut self, lhs: &V, rhs: &V) -> V {
        let lhs = self.cond(lhs);
        let rhs = self.cond(rhs);
        self.bin("or", &lhs, &rhs)
    }

    fn bool32(&mut self, value: &V) -> V {
        let value = self.cond(value);
        self.zext(&value, Ty::I32)
    }

    fn gep8(&mut self, base: &V, offset: i64) -> V {
        let base = self.as_ptr(base);
        if offset == 0 {
            return base;
        }
        self.def(Ty::Ptr, &format!("getelementptr inbounds i8, ptr {}, i64 {offset}", base.op))
    }

    fn gep8_dyn(&mut self, base: &V, offset: &V) -> V {
        let base = self.as_ptr(base);
        let offset = self.word(offset);
        self.def(Ty::Ptr, &format!("getelementptr i8, ptr {}, i64 {}", base.op, offset.op))
    }

    fn gep_words(&mut self, base: &V, index: &V) -> V {
        let base = self.as_ptr(base);
        let index = self.word(index);
        self.def(Ty::Ptr, &format!("getelementptr i64, ptr {}, i64 {}", base.op, index.op))
    }

    fn load_aligned(&mut self, ty: Ty, ptr: &V, mem: Mem, align: u32) -> V {
        let ptr = self.as_ptr(ptr);
        let invariant = if mem == Mem::Binding { format!(", !invariant.load !{MD_EMPTY}") } else { String::new() };
        self.def(
            ty,
            &format!("load {}, ptr {}, align {align}, !tbaa !{}{invariant}", ty.ir(), ptr.op, mem.tag()),
        )
    }

    fn load(&mut self, ty: Ty, ptr: &V, mem: Mem) -> V {
        self.load_aligned(ty, ptr, mem, ty.align())
    }

    fn load_at(&mut self, ty: Ty, base: &V, offset: i64, mem: Mem) -> V {
        let ptr = self.gep8(base, offset);
        self.load(ty, &ptr, mem)
    }

    fn store(&mut self, value: &V, ptr: &V, mem: Mem) {
        let ptr = self.as_ptr(ptr);
        self.line(&format!(
            "store {} {}, ptr {}, align {}, !tbaa !{}",
            value.ty.ir(),
            value.op,
            ptr.op,
            value.ty.align(),
            mem.tag()
        ));
    }

    fn store_at(&mut self, value: &V, base: &V, offset: i64, mem: Mem) {
        let ptr = self.gep8(base, offset);
        self.store(value, &ptr, mem);
    }

    fn activation() -> V {
        V { op: "%act".to_owned(), ty: Ty::Ptr }
    }

    fn links() -> V {
        V { op: "%links".to_owned(), ty: Ty::Ptr }
    }

    /// Load an activation field that is invariant for one native invocation in
    /// the function entry block so it dominates every use.
    fn entry_activation_load(&mut self, ty: Ty, offset: usize) -> V {
        let ptr = self.def_entry(Ty::Ptr, &format!("getelementptr inbounds i8, ptr %act, i64 {offset}"));
        self.def_entry(ty, &format!("load {}, ptr {}, align {}, !tbaa !{}", ty.ir(), ptr.op, ty.align(), Mem::Act.tag()))
    }

    // ------------------------------------------------------------- VM bindings

    fn vm_bindings_base(&mut self) -> V {
        if let Some(value) = &self.vm_bindings_base {
            return value.clone();
        }
        let value = self.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, vm_bindings));
        self.vm_bindings_base = Some(value.clone());
        value
    }

    fn vm_binding_word(&mut self, kind: VmBindingKind, raw: u64) -> Result<V, String> {
        let key = VmBindingKey::new(kind, raw);
        let slot = *self
            .vm_binding_slots
            .get(&key)
            .ok_or_else(|| self.error(format!("missing VM binding {kind:?} for {raw:#x}")))?;
        if let Some(value) = self.vm_binding_values.get(&slot) {
            return Ok(value.clone());
        }
        let base = self.vm_bindings_base();
        let ptr = self.def_entry(Ty::Ptr, &format!("getelementptr inbounds i64, ptr {}, i64 {slot}", base.op));
        let value = self.def_entry(
            Ty::I64,
            &format!(
                "load i64, ptr {}, align 8, !tbaa !{}, !invariant.load !{MD_EMPTY}",
                ptr.op,
                Mem::Binding.tag()
            ),
        );
        self.vm_binding_values.insert(slot, value.clone());
        Ok(value)
    }

    fn vm_binding_i32(&mut self, kind: VmBindingKind, raw: u32) -> Result<V, String> {
        let word = self.vm_binding_word(kind, raw as u64)?;
        Ok(self.trunc(&word, Ty::I32))
    }

    fn runtime_context(&mut self) -> V {
        if let Some(value) = &self.runtime_context {
            return value.clone();
        }
        let value = self.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, runtime));
        self.runtime_context = Some(value.clone());
        value
    }

    fn execute_callee(&mut self) -> V {
        if let Some(value) = &self.execute_callee {
            return value.clone();
        }
        let value = self.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, execute));
        self.execute_callee = Some(value.clone());
        value
    }

    fn heap_barrier_callee(&mut self) -> V {
        if let Some(value) = &self.heap_barrier_callee {
            return value.clone();
        }
        let value = self.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, heap_write_barrier));
        self.heap_barrier_callee = Some(value.clone());
        value
    }

    fn method_epoch_value(&mut self) -> V {
        if let Some(value) = &self.method_epoch_value {
            return value.clone();
        }
        let value = self.entry_activation_load(Ty::I64, offset_of!(TraceActivation, method_epoch));
        self.method_epoch_value = Some(value.clone());
        value
    }

    /// Epoch-local: runtime helpers may refresh the chunk array. Reloaded at
    /// every use; LLVM GVN merges loads between runtime boundaries.
    fn heap_chunks_base(&mut self) -> V {
        let activation = Self::activation();
        self.load_at(Ty::Ptr, &activation, offset_of!(TraceActivation, heap_chunks) as i64, Mem::Act)
    }

    // ------------------------------------------------------------- trace values

    fn cached_value(&self, id: TraceValueId) -> Option<V> {
        self.values.get(id.0 as usize).and_then(Clone::clone)
    }

    fn cache_value(&mut self, id: TraceValueId, value: V) {
        if let Some(slot) = self.values.get_mut(id.0 as usize) {
            *slot = Some(value);
        }
    }

    fn value_rep(&self, value: TraceValueId) -> Result<TraceRep, String> {
        self.trace
            .value(value)
            .map(|value| value.rep)
            .ok_or_else(|| self.error(format!("unknown trace value {value:?}")))
    }

    fn lower_value(&mut self, value: TraceValueId) -> Result<V, String> {
        let value = self.trace.canonical_value(value);
        if let Some(existing) = self.cached_value(value) {
            return Ok(existing);
        }
        let (origin, rep) = {
            let info = self.trace.value(value).ok_or_else(|| self.error(format!("unknown trace value {value:?}")))?;
            (info.origin.clone(), info.rep)
        };
        let lowered = match origin {
            TraceValueOrigin::EntrySlot { frame, slot } => {
                if frame.0 != 0 {
                    return Err(self.error("LLVM tier cannot load recreated inlined frames"));
                }
                self.load_frame_slot(slot, rep)
            }
            TraceValueOrigin::GuardedEntry { value: source, exit } => {
                let tagged = self.lower_value(source)?;
                match rep {
                    TraceRep::F64 => self.guard_numeric(&tagged, exit)?,
                    TraceRep::Int32 => self.guard_int32_numeric(&tagged, exit)?,
                    TraceRep::UInt32 => self.guard_uint32_numeric(&tagged, exit)?,
                    _ => return Err(self.error(format!("unsupported guarded-entry representation {rep:?}"))),
                }
            }
            TraceValueOrigin::Constant(constant) => self.lower_constant(constant)?,
            TraceValueOrigin::InstResult { .. } => {
                return Err(self.error(format!("instruction result {value:?} requested before its producer")));
            }
            TraceValueOrigin::LoopPhi { .. } | TraceValueOrigin::LoopTemporary => {
                return Err(self.error(format!("loop value {value:?} has no LLVM definition")));
            }
            TraceValueOrigin::Alias(alias) => return self.lower_value(alias),
        };
        self.cache_value(value, lowered.clone());
        Ok(lowered)
    }

    fn lower_constant(&mut self, constant: TraceConstant) -> Result<V, String> {
        Ok(match constant {
            TraceConstant::Null => V::i64(TAG_NULL as i64),
            TraceConstant::Bool(value) => V::i64(if value { TAG_TRUE } else { TAG_FALSE } as i64),
            TraceConstant::NumBits(bits) => V::i64(bits as i64),
            TraceConstant::Int32(value) => V::i32(value),
            TraceConstant::F64Bits(bits) => V::f64_bits(bits),
            TraceConstant::StableTagged(bits) => self.vm_binding_word(VmBindingKind::StableTagged, bits)?,
            TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                return Err(self.error("runtime-owned constant reached the LLVM tier"));
            }
        })
    }

    fn bank_slot(&mut self, slot: FrameSlot) -> V {
        let (bank, index) = match slot {
            FrameSlot::General(reg) => (offset_of!(TraceActivation, general), reg.0 as i64),
            FrameSlot::Argument(reg) => (offset_of!(TraceActivation, arguments), reg.0 as i64),
            FrameSlot::Return(reg) => (offset_of!(TraceActivation, returns), reg.0 as i64),
            FrameSlot::Special(reg) => (offset_of!(TraceActivation, special), reg as u8 as i64),
        };
        // Runtime helpers may replace the active frame; the bank pointer is
        // reloaded for every access exactly like the Cranelift tier.
        let activation = Self::activation();
        let base = self.load_at(Ty::Ptr, &activation, bank as i64, Mem::Act);
        self.gep8(&base, index * std::mem::size_of::<RegisterWord>() as i64)
    }

    fn load_frame_slot(&mut self, slot: FrameSlot, rep: TraceRep) -> V {
        let ptr = self.bank_slot(slot);
        let raw = self.load(Ty::I64, &ptr, Mem::Reg);
        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => raw,
            TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => self.trunc(&raw, Ty::I32),
            TraceRep::F64 => self.bitcast(&raw, Ty::F64),
        }
    }

    fn store_frame_word(&mut self, slot: FrameSlot, word: &V) {
        let ptr = self.bank_slot(slot);
        let word = self.word(word);
        self.store(&word, &ptr, Mem::Reg);
    }

    fn normalize_to_word(&mut self, value: &V, rep: TraceRep) -> V {
        match rep {
            TraceRep::Tagged | TraceRep::HeapHandle | TraceRep::RawPtr => self.word(value),
            TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => {
                let narrow = if value.ty == Ty::I1 { self.zext(value, Ty::I32) } else { value.clone() };
                self.zext(&narrow, Ty::I64)
            }
            TraceRep::F64 => self.bitcast(value, Ty::I64),
        }
    }

    fn materialize_exit_word(&mut self, value: &V, source: TraceRep, target: TraceRep) -> Result<V, String> {
        if source == target {
            return Ok(self.normalize_to_word(value, source));
        }
        Ok(match (source, target) {
            (TraceRep::F64, TraceRep::Tagged) => self.box_f64(value),
            (TraceRep::Int32, TraceRep::Tagged) | (TraceRep::Int32, TraceRep::F64) => {
                let number = self.cast("sitofp", value, Ty::F64);
                self.bitcast(&number, Ty::I64)
            }
            (TraceRep::UInt32, TraceRep::Tagged) | (TraceRep::UInt32, TraceRep::F64) => {
                let number = self.cast("uitofp", value, Ty::F64);
                self.bitcast(&number, Ty::I64)
            }
            (TraceRep::Bool, TraceRep::Tagged) => self.box_bool(value),
            (TraceRep::HeapHandle, TraceRep::Tagged) => {
                let word = self.word(value);
                self.bin("or", &word, &V::i64(TAG_HEAP as i64))
            }
            (TraceRep::Bool, TraceRep::Int32) => self.zext(value, Ty::I64),
            (TraceRep::Tagged, TraceRep::HeapHandle) => {
                let word = self.word(value);
                self.bin("and", &word, &V::i64(PAYLOAD_MASK as i64))
            }
            _ => return Err(self.error(format!("cannot materialize side-exit value from {source:?} to {target:?}"))),
        })
    }

    // ------------------------------------------------------- tagged arithmetic

    fn tagged_is_num(&mut self, bits: &V) -> V {
        let bits = self.word(bits);
        let masked = self.bin("and", &bits, &V::i64(TAG_MASK as i64));
        let not_tag = self.icmp("ne", &masked, &V::i64(TAG_BASE as i64));
        let heap_masked = self.bin("and", &bits, &V::i64(TAG_HEAP_MASK as i64));
        let not_heap = self.icmp("ne", &heap_masked, &V::i64(TAG_HEAP as i64));
        self.and1(&not_tag, &not_heap)
    }

    fn tagged_is_heap(&mut self, bits: &V) -> V {
        let bits = self.word(bits);
        let masked = self.bin("and", &bits, &V::i64(TAG_HEAP_MASK as i64));
        self.icmp("eq", &masked, &V::i64(TAG_HEAP as i64))
    }

    fn tagged_falsey(&mut self, bits: &V) -> V {
        let bits = self.word(bits);
        let shifted = self.bin("add", &bits, &V::i64((TAG_NULL as i64).wrapping_neg()));
        self.icmp("ule", &shifted, &V::i64(1))
    }

    fn box_bool(&mut self, condition: &V) -> V {
        let condition = self.cond(condition);
        let bit = self.zext(&condition, Ty::I64);
        self.bin("add", &V::i64(TAG_FALSE as i64), &bit)
    }

    fn box_f64(&mut self, value: &V) -> V {
        let is_nan = self.fcmp("uno", value, value);
        let bits = self.bitcast(value, Ty::I64);
        self.select(&is_nan, &V::i64(CANONICAL_NAN as i64), &bits)
    }

    fn call_pure_f64(&mut self, helper: &str, args: &[&V]) -> V {
        let args = args.iter().map(|arg| format!("double {}", arg.op)).collect::<Vec<_>>().join(", ");
        self.def(Ty::F64, &format!("call double @{helper}({args}) #1"))
    }

    fn intrinsic_f64(&mut self, name: &str, value: &V) -> V {
        self.def(Ty::F64, &format!("call double @llvm.{name}.f64(double {})", value.op))
    }

    /// For an exact binary power of two, IEEE-754 division can be reduced to
    /// multiplication by its exactly representable reciprocal. The finite,
    /// normal divisor restriction also excludes division by zero/infinity,
    /// NaN, and the case where the reciprocal would overflow.
    fn exact_power_of_two_reciprocal(divisor: f64) -> Option<f64> {
        const FRACTION_MASK: u64 = (1u64 << 52) - 1;
        let abs_bits = divisor.to_bits() & !(1u64 << 63);
        let exponent = (abs_bits >> 52) & 0x7ff;
        if exponent == 0 || exponent == 0x7ff || (abs_bits & FRACTION_MASK) != 0 {
            return None;
        }
        let reciprocal = divisor.recip();
        (reciprocal.is_finite() && reciprocal != 0.0).then_some(reciprocal)
    }

    fn f64_remainder(&mut self, lhs: &V, rhs: &V, constant: Option<f64>) -> V {
        let Some(divisor) = constant
            .map(f64::abs)
            .filter(|n| n.fract() == 0.0 && (1.0..=i32::MAX as f64).contains(n))
        else {
            return self.call_pure_f64("napalm_jit_frem", &[lhs, rhs]);
        };
        // x % (+/-1) is the signed fractional part of x for every finite
        // binary64 value. For infinities/NaNs the subtraction produces NaN,
        // matching fmod. This removes the generic libcall and range branch.
        if divisor == 1.0 {
            let magnitude = self.intrinsic_f64("fabs", lhs);
            let integral = self.intrinsic_f64("trunc", &magnitude);
            let fractional = self.bin("fsub", &magnitude, &integral);
            return self.def(
                Ty::F64,
                &format!(
                    "call double @llvm.copysign.f64(double {}, double {})",
                    fractional.op, lhs.op
                ),
            );
        }

        let fast = self.new_block();
        let slow = self.new_block();
        let done = self.new_block();
        let magnitude = self.intrinsic_f64("fabs", lhs);
        let in_range = self.fcmp("ole", &magnitude, &V::f64(i32::MAX as f64));
        self.cond_br(&in_range, &fast, &slow, Some(MD_HOT_TRUE));

        self.begin(&fast);
        // For integral d, floor(x / d) == floor(x) / d; every product fits i32.
        // The range guard strictly dominates this conversion. A saturating
        // intrinsic would add unnecessary min/max handling on many targets.
        let integer = self.cast("fptosi", &magnitude, Ty::I32);
        let quotient = self.bin("udiv", &integer, &V::i32(divisor as i32));
        let multiple = self.bin("mul", &quotient, &V::i32(divisor as i32));
        let multiple = self.cast("sitofp", &multiple, Ty::F64);
        let remainder = self.bin("fsub", &magnitude, &multiple);
        let fast_result = self.def(
            Ty::F64,
            &format!("call double @llvm.copysign.f64(double {}, double {})", remainder.op, lhs.op),
        );
        let fast_end = self.current.clone();
        self.br(&done);

        self.begin(&slow);
        let slow_result = self.call_pure_f64("napalm_jit_frem", &[lhs, rhs]);
        let slow_end = self.current.clone();
        self.br(&done);

        self.begin(&done);
        self.def(
            Ty::F64,
            &format!("phi double [ {}, %{fast_end} ], [ {}, %{slow_end} ]", fast_result.op, slow_result.op),
        )
    }

    fn guard_numeric(&mut self, tagged: &V, exit: SideExitId) -> Result<V, String> {
        let ok = self.tagged_is_num(tagged);
        let failure = self.not1(&ok);
        self.emit_side_exit(exit, &failure)?;
        Ok(self.bitcast(tagged, Ty::F64))
    }

    fn guard_int32_numeric(&mut self, tagged: &V, exit: SideExitId) -> Result<V, String> {
        let number = self.guard_numeric(tagged, exit)?;
        let tagged = self.word(tagged);
        let negative_zero = self.icmp("eq", &tagged, &V::i64((-0.0f64).to_bits() as i64));
        let ge_min = self.fcmp("oge", &number, &V::f64(i32::MIN as f64));
        let le_max = self.fcmp("ole", &number, &V::f64(i32::MAX as f64));
        let in_range = self.and1(&ge_min, &le_max);
        let out_of_range = self.not1(&in_range);
        let failure = self.or1(&negative_zero, &out_of_range);
        self.emit_side_exit(exit, &failure)?;
        // The range / NaN side exit has already executed. Convert directly
        // to the final width so LLVM can use the cheaper narrow conversion.
        let narrow = self.cast("fptosi", &number, Ty::I32);
        let back = self.cast("sitofp", &narrow, Ty::F64);
        let exact = self.fcmp("oeq", &number, &back);
        let not_exact = self.not1(&exact);
        self.emit_side_exit(exit, &not_exact)?;
        Ok(narrow)
    }

    fn guard_uint32_numeric(&mut self, tagged: &V, exit: SideExitId) -> Result<V, String> {
        let number = self.guard_numeric(tagged, exit)?;
        let ge_zero = self.fcmp("oge", &number, &V::f64(0.0));
        let le_max = self.fcmp("ole", &number, &V::f64(u32::MAX as f64));
        let in_range = self.and1(&ge_zero, &le_max);
        let failure = self.not1(&in_range);
        self.emit_side_exit(exit, &failure)?;
        let narrow = self.cast("fptoui", &number, Ty::I32);
        let back = self.cast("uitofp", &narrow, Ty::F64);
        let exact = self.fcmp("oeq", &number, &back);
        let not_exact = self.not1(&exact);
        self.emit_side_exit(exit, &not_exact)?;
        Ok(narrow)
    }

    // --------------------------------------------------------------- side exits

    fn emit_side_exit(&mut self, exit: SideExitId, failure: &V) -> Result<(), String> {
        self.emit_conditional_side_exit(exit, failure, true)
    }

    fn emit_conditional_side_exit(&mut self, exit: SideExitId, condition: &V, exit_when_true: bool) -> Result<(), String> {
        let plans = self.exit_plans;
        let plan = plans.get(exit.0 as usize).ok_or_else(|| self.error(format!("unknown side exit {exit:?}")))?;
        let resume = self.new_block();

        let branch = |state: &mut Self, stop: &str, resume: &str| {
            if exit_when_true {
                state.cond_br(condition, stop, resume, Some(MD_COLD_TRUE));
            } else {
                state.cond_br(condition, resume, stop, Some(MD_HOT_TRUE));
            }
        };

        if plan.dynamic_values.is_empty() {
            let key = (exit.0, self.native_budget_remaining.clone());
            if let Some(stop) = self.shared_constant_exit_blocks.get(&key).cloned() {
                branch(self, &stop, &resume);
                self.begin(&resume);
                return Ok(());
            }
            let stop = self.new_block();
            self.shared_constant_exit_blocks.insert(key, stop.clone());
            branch(self, &stop, &resume);
            self.begin(&stop);
            self.materialize_exit_or_native_link(plan)?;
            self.begin(&resume);
            return Ok(());
        }

        let stop = self.new_block();
        branch(self, &stop, &resume);
        self.begin(&stop);
        self.materialize_exit_or_native_link(plan)?;
        self.begin(&resume);
        Ok(())
    }

    fn emit_unconditional_side_exit(&mut self, exit: SideExitId) -> Result<(), String> {
        let plans = self.exit_plans;
        let plan = plans.get(exit.0 as usize).ok_or_else(|| self.error(format!("unknown side exit {exit:?}")))?;
        self.materialize_exit_or_native_link(plan)
    }

    fn materialize_exit_or_native_link(&mut self, plan: &CompiledExitPlan) -> Result<(), String> {
        if !plan.native_linkable {
            return self.materialize_side_exit(plan);
        }
        let links = Self::links();
        let child = self.load_at(Ty::I32, &links, plan.id.0 as i64 * 4, Mem::Links);
        let linked = self.icmp("ne", &child, &V::i32(-1));
        let linked_block = self.new_block();
        let slow_exit = self.new_block();
        self.cond_br(&linked, &linked_block, &slow_exit, None);
        self.begin(&linked_block);
        self.materialize_native_link(plan, &child)?;
        self.begin(&slow_exit);
        self.materialize_side_exit(plan)
    }

    fn store_completed_iterations(&mut self) {
        let completed = if let Some(remaining) = self.native_budget_remaining.clone() {
            let completed = self.bin("sub", &V::i64(self.native_budget_initial as i64), &remaining);
            self.trunc(&completed, Ty::I32)
        } else {
            V::i32(0)
        };
        let activation = Self::activation();
        self.store_at(&completed, &activation, offset_of!(TraceActivation, native_iterations) as i64, Mem::Act);
    }

    fn exit_constant_rep(&self, constant: TraceConstant) -> Result<TraceRep, String> {
        Ok(match constant {
            TraceConstant::Int32(_) => TraceRep::Int32,
            TraceConstant::F64Bits(_) => TraceRep::F64,
            TraceConstant::Null | TraceConstant::Bool(_) | TraceConstant::NumBits(_) | TraceConstant::StableTagged(_) => {
                TraceRep::Tagged
            }
            TraceConstant::String(_) | TraceConstant::BuiltinClass(_) => {
                return Err(self.error("runtime-owned constant reached a native trace link"));
            }
        })
    }

    fn materialize_native_link(&mut self, plan: &CompiledExitPlan, child: &V) -> Result<(), String> {
        self.store_completed_iterations();
        for write in &plan.writes {
            let word = match write.source {
                ExitWriteSource::Dynamic(index) => {
                    let dynamic = plan
                        .dynamic_values
                        .get(index as usize)
                        .copied()
                        .ok_or_else(|| self.error("side-exit dynamic value index out of bounds"))?;
                    let source = self.value_rep(dynamic.value)?;
                    let value = self.lower_value(dynamic.value)?;
                    self.materialize_exit_word(&value, source, dynamic.target_rep)?
                }
                ExitWriteSource::Constant(constant) => {
                    let source = self.exit_constant_rep(constant)?;
                    let value = self.lower_constant(constant)?;
                    self.materialize_exit_word(&value, source, write.target_rep)?
                }
            };
            self.store_frame_word(write.slot, &word);
        }
        let activation = Self::activation();
        self.store_at(child, &activation, offset_of!(TraceActivation, next_trace) as i64, Mem::Act);
        self.ret(TRACE_RESULT_LINK);
        Ok(())
    }

    fn materialize_side_exit(&mut self, plan: &CompiledExitPlan) -> Result<(), String> {
        self.store_completed_iterations();
        for write in &plan.writes {
            let ExitWriteSource::Dynamic(index) = write.source else { continue; };
            let dynamic = plan
                .dynamic_values
                .get(index as usize)
                .copied()
                .ok_or_else(|| self.error("side-exit dynamic value index out of bounds"))?;
            let source = self.value_rep(dynamic.value)?;
            let value = self.lower_value(dynamic.value)?;
            let word = self.materialize_exit_word(&value, source, dynamic.target_rep)?;
            self.store_frame_word(write.slot, &word);
        }
        let activation = Self::activation();
        self.store_at(&V::i32(plan.id.0 as i32), &activation, offset_of!(TraceActivation, exit_id) as i64, Mem::Act);
        self.ret(TRACE_RESULT_SIDE_EXIT);
        Ok(())
    }

    // ------------------------------------------------------------ heap helpers

    fn trace_value_may_be_heap(&self, value: TraceValueId) -> Result<bool, String> {
        let value = self.trace.value(value).ok_or_else(|| self.error("unknown heap-store value"))?;
        Ok(matches!(value.ty, TraceType::Any))
    }

    fn heap_entry_metadata(&mut self, entry: &V, cacheable: bool) -> V {
        if cacheable {
            if let Some(value) = self.heap_metadata.get(entry) {
                return value.clone();
            }
        }
        let value = self.load_at(Ty::I32, entry, offset_of!(TraceHeapEntry, kind) as i64, Mem::Sidecar);
        if cacheable {
            self.heap_metadata.insert(entry.clone(), value.clone());
        }
        value
    }

    fn heap_entry_data(&mut self, entry: &V) -> V {
        if let Some(value) = self.heap_data.get(entry) {
            return value.clone();
        }
        let value = self.load_at(Ty::Ptr, entry, offset_of!(TraceHeapEntry, data) as i64, Mem::Sidecar);
        self.heap_data.insert(entry.clone(), value.clone());
        value
    }

    fn heap_entry_map_data(&mut self, entry: &V, owner: &V) -> V {
        let encoded = self.heap_entry_data(entry);
        let encoded = self.word(&encoded);
        let offset = self.bin("and", &encoded, &V::i64(!(TRACE_HEAP_DATA_RELATIVE_MASK as u64) as i64));
        self.gep8_dyn(owner, &offset)
    }

    fn heap_entry_len(&mut self, entry: &V) -> V {
        if let Some(value) = self.heap_len.get(entry) {
            return value.clone();
        }
        let value = self.load_at(Ty::I64, entry, offset_of!(TraceHeapEntry, len) as i64, Mem::Sidecar);
        self.heap_len.insert(entry.clone(), value.clone());
        value
    }

    fn heap_entry_receiver_class(&mut self, entry: &V) -> V {
        if let Some(value) = self.heap_receiver_class.get(entry) {
            return value.clone();
        }
        let value = self.load_at(Ty::I64, entry, offset_of!(TraceHeapEntry, receiver_class) as i64, Mem::Sidecar);
        self.heap_receiver_class.insert(entry.clone(), value.clone());
        value
    }

    fn heap_metadata_payload(&mut self, metadata: &V) -> V {
        self.bin("lshr", metadata, &V::i32(TRACE_HEAP_METADATA_SHIFT as i32))
    }

    fn guard_heap_entry_kind(&mut self, entry: &V, expected: TraceHeapKind, exit: SideExitId) -> Result<V, String> {
        let metadata = self.heap_entry_metadata(entry, expected != TraceHeapKind::Promise);
        let kind = self.bin("and", &metadata, &V::i32(TRACE_HEAP_KIND_MASK as i32));
        let wrong = self.icmp("ne", &kind, &V::i32(expected as i32));
        self.emit_side_exit(exit, &wrong)?;
        Ok(metadata)
    }

    fn live_heap_entry_from_heap_tagged(&mut self, tagged: &V, exit: SideExitId) -> Result<(V, V), String> {
        let tagged = self.word(tagged);
        let handle = self.bin("and", &tagged, &V::i64(PAYLOAD_MASK as i64));
        let chunk_index = self.bin("lshr", &handle, &V::i64(TRACE_HEAP_CHUNK_BITS as i64));
        let entry_index = self.bin("and", &handle, &V::i64(TRACE_HEAP_CHUNK_MASK as i64));
        let chunks = self.heap_chunks_base();
        let chunk_slot = self.def(Ty::Ptr, &format!("getelementptr ptr, ptr {}, i64 {}", chunks.op, chunk_index.op));
        let chunk = self.load(Ty::Ptr, &chunk_slot, Mem::Sidecar);
        let entry_offset = self.bin("mul", &entry_index, &V::i64(std::mem::size_of::<TraceHeapEntry>() as i64));
        let entry = self.gep8_dyn(&chunk, &entry_offset);
        let address = self.load_at(Ty::Ptr, &entry, offset_of!(TraceHeapEntry, address) as i64, Mem::Sidecar);
        let dead = self.def(Ty::I1, &format!("icmp eq ptr {}, null", address.op));
        self.emit_side_exit(exit, &dead)?;
        Ok((entry, address))
    }

    fn guard_live_heap_entry(&mut self, tagged: &V, exit: SideExitId) -> Result<(V, V), String> {
        let is_heap = self.tagged_is_heap(tagged);
        let not_heap = self.not1(&is_heap);
        self.emit_side_exit(exit, &not_heap)?;
        self.live_heap_entry_from_heap_tagged(tagged, exit)
    }

    fn proven_heap_entry_kind(&self, value: TraceValueId) -> Option<TraceHeapObjectKind> {
        let value = self.trace.canonical_value(value);
        let TraceValueOrigin::InstResult { node } = &self.trace.value(value)?.origin else { return None; };
        match &self.trace.nodes.get(*node)?.kind {
            TraceInst::GuardHeapObject { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    fn live_heap_entry_for_value(&mut self, value: TraceValueId, exit: SideExitId) -> Result<(V, V), String> {
        let canonical = self.trace.canonical_value(value);
        if let Some(entry) = self.heap_live_entries.get(&canonical) {
            return Ok(entry.clone());
        }
        if self.proven_heap_entry_kind(canonical).is_some() {
            let entry = self.lower_value(canonical)?;
            let entry = self.as_ptr(&entry);
            let address = if let Some(address) = self.heap_addresses.get(canonical.0 as usize).and_then(Clone::clone) {
                address
            } else {
                let address = self.load_at(Ty::Ptr, &entry, offset_of!(TraceHeapEntry, address) as i64, Mem::Sidecar);
                let dead = self.def(Ty::I1, &format!("icmp eq ptr {}, null", address.op));
                self.emit_side_exit(exit, &dead)?;
                address
            };
            let result = (entry, address);
            self.heap_live_entries.insert(canonical, result.clone());
            return Ok(result);
        }
        let tagged = self.lower_value(canonical)?;
        let result = self.guard_live_heap_entry(&tagged, exit)?;
        self.heap_live_entries.insert(canonical, result.clone());
        Ok(result)
    }

    fn heap_entry_for_value(&mut self, value: TraceValueId, expected: TraceHeapKind, exit: SideExitId) -> Result<(V, V), String> {
        let canonical = self.trace.canonical_value(value);
        if let Some(entry) = self.heap_entries.get(&(canonical, expected as u32)) {
            return Ok(entry.clone());
        }
        let proven = self.proven_heap_entry_kind(canonical).map(CraneliftLoweringState::runtime_heap_kind) == Some(expected);
        let (entry, address) = self.live_heap_entry_for_value(canonical, exit)?;
        if !proven {
            self.guard_heap_entry_kind(&entry, expected, exit)?;
        }
        let result = (entry, address);
        self.heap_entries.insert((canonical, expected as u32), result.clone());
        Ok(result)
    }

    fn heap_entry_with_metadata_for_value(
        &mut self,
        value: TraceValueId,
        expected: TraceHeapKind,
        exit: SideExitId,
    ) -> Result<(V, V, V), String> {
        let (entry, address) = self.heap_entry_for_value(value, expected, exit)?;
        let metadata = self.heap_entry_metadata(&entry, expected != TraceHeapKind::Promise);
        Ok((entry, address, metadata))
    }

    fn guard_uncontracted_store(&mut self, entry: &V, exit: SideExitId) -> Result<(), String> {
        let contract = self.load_at(Ty::I32, entry, offset_of!(TraceHeapEntry, has_contract) as i64, Mem::Sidecar);
        let invalid = self.icmp("ne", &contract, &V::i32(0));
        self.emit_side_exit(exit, &invalid)
    }

    fn guard_map_shape_once(&mut self, map: TraceValueId, metadata: &V, expected: u32, exit: SideExitId) -> Result<(), String> {
        let key = (self.trace.canonical_value(map), expected);
        if self.map_shape_proofs.contains(&key) {
            return Ok(());
        }
        let shape = self.heap_metadata_payload(metadata);
        let expected_shape = self.vm_binding_i32(VmBindingKind::ShapeId, expected)?;
        let fail = self.icmp("ne", &shape, &expected_shape);
        self.emit_side_exit(exit, &fail)?;
        self.map_shape_proofs.insert(key);
        Ok(())
    }

    fn guard_string_bytes_once(&mut self, key: TraceValueId, tagged: &V, expected: &[u8], exit: SideExitId) -> Result<(), String> {
        let key = self.trace.canonical_value(key);
        if self
            .trace
            .value(key)
            .is_some_and(|value| matches!(&value.origin, TraceValueOrigin::Constant(TraceConstant::StableTagged(_))))
        {
            return Ok(());
        }
        let proof = (key, expected.to_vec());
        if self.string_key_proofs.contains(&proof) {
            return Ok(());
        }
        self.guard_string_bytes(tagged, expected, exit)?;
        self.string_key_proofs.insert(proof);
        Ok(())
    }

    fn guard_string_bytes(&mut self, tagged: &V, expected: &[u8], exit: SideExitId) -> Result<(), String> {
        let (entry, _) = self.guard_live_heap_entry(tagged, exit)?;
        let metadata = self.guard_heap_entry_kind(&entry, TraceHeapKind::String, exit)?;
        let byte_len = self.heap_metadata_payload(&metadata);
        let wrong_len = self.icmp("ne", &byte_len, &V::i32(expected.len() as i32));
        self.emit_side_exit(exit, &wrong_len)?;

        let data = self.heap_entry_data(&entry);
        let mut mismatch: Option<V> = None;
        let mut offset = 0usize;
        let chunks: [(usize, Ty); 4] = [(8, Ty::I64), (4, Ty::I32), (2, Ty::I16), (1, Ty::I8)];
        for (width, ty) in chunks {
            while offset + width <= expected.len() {
                let mut bytes = [0u8; 8];
                bytes[..width].copy_from_slice(&expected[offset..offset + width]);
                let wanted = u64::from_le_bytes(bytes);
                let ptr = self.gep8(&data, offset as i64);
                let actual = self.load_aligned(ty, &ptr, Mem::Str, 1);
                let actual = self.zext(&actual, Ty::I64);
                let diff = self.bin("xor", &actual, &V::i64(wanted as i64));
                mismatch = Some(match mismatch {
                    Some(previous) => self.bin("or", &previous, &diff),
                    None => diff,
                });
                offset += width;
                if width == 8 {
                    continue;
                }
                break;
            }
        }
        if let Some(mismatch) = mismatch {
            let failure = self.icmp("ne", &mismatch, &V::i64(0));
            self.emit_side_exit(exit, &failure)?;
        }
        Ok(())
    }

    fn guard_constant_index(&mut self, len: &V, index: u32, exit: SideExitId) -> Result<(), String> {
        if !self.constant_index_proofs.insert((len.clone(), index)) {
            return Ok(());
        }
        let failure = self.icmp("ule", len, &V::i64(index as i64));
        self.emit_side_exit(exit, &failure)
    }

    fn normalize_list_index(&mut self, index: &V, len: &V, exit: SideExitId) -> Result<V, String> {
        let index = self.sext(index, Ty::I64);
        let negative = self.icmp("slt", &index, &V::i64(0));
        let from_end = self.bin("add", len, &index);
        let normalized = self.select(&negative, &from_end, &index);
        let fail = self.icmp("uge", &normalized, len);
        self.emit_side_exit(exit, &fail)?;
        Ok(normalized)
    }

    fn validated_list_index(&mut self, list: TraceValueId, index: TraceValueId, entry: &V, exit: SideExitId) -> Result<V, String> {
        let key = (self.trace.canonical_value(list), self.trace.canonical_value(index));
        if let Some(normalized) = self.list_indices.get(&key) {
            return Ok(normalized.clone());
        }
        let value = self.lower_value(index)?;
        let len = self.heap_entry_len(entry);
        let normalized = self.normalize_list_index(&value, &len, exit)?;
        self.list_indices.insert(key, normalized.clone());
        Ok(normalized)
    }

    fn invalidate_heap_proofs(&mut self) {
        self.heap_addresses.fill(None);
        self.heap_entries.clear();
        self.heap_live_entries.clear();
        self.heap_data.clear();
        self.heap_len.clear();
        self.heap_metadata.clear();
        self.heap_receiver_class.clear();
        self.list_indices.clear();
        self.list_lengths.clear();
        self.map_shape_proofs.clear();
        self.string_key_proofs.clear();
        self.forwarded_heap_loads.clear();
        self.forwarded_absolute_loads.clear();
    }

    fn publish_heap_store(&mut self, location: Fwd, value: V) {
        self.forwarded_heap_loads.clear();
        self.forwarded_heap_loads.insert(location, value);
    }

    fn call_heap_barrier(&mut self, owner: &V) {
        let callee = self.heap_barrier_callee();
        let runtime = self.runtime_context();
        let owner = self.as_ptr(owner);
        self.line(&format!("call void {}(ptr {}, ptr {}) #3", callee.op, runtime.op, owner.op));
    }

    fn emit_heap_write_barrier(&mut self, owner: &V, value_id: TraceValueId, tagged: &V) -> Result<(), String> {
        if !self.trace_value_may_be_heap(value_id)? {
            return Ok(());
        }
        let is_heap = self.tagged_is_heap(tagged);
        let barrier = self.new_block();
        let done = self.new_block();
        self.cond_br(&is_heap, &barrier, &done, None);
        self.begin(&barrier);
        self.call_heap_barrier(owner);
        self.br(&done);
        self.begin(&done);
        Ok(())
    }

    fn class_matches(&mut self, actual: &V, classes: &[u64]) -> Result<V, String> {
        let mut matches: Option<V> = None;
        for &class in classes {
            let expected = self.vm_binding_word(VmBindingKind::ClassKey, class)?;
            let equal = self.icmp("eq", actual, &expected);
            matches = Some(match matches {
                Some(previous) => self.or1(&previous, &equal),
                None => equal,
            });
        }
        Ok(matches.unwrap_or(V { op: "false".to_owned(), ty: Ty::I1 }))
    }

    fn store_activation_i32(&mut self, offset: usize, value: &V) {
        let activation = Self::activation();
        self.store_at(value, &activation, offset as i64, Mem::Act);
    }

    fn runtime_call(&mut self, function: u32, pc: u32, next: u32) -> V {
        let callee = self.execute_callee();
        self.def(
            Ty::I32,
            &format!(
                "call i32 {}(ptr %act, i32 {}, i32 {}, i32 {}) #3",
                callee.op, function as i32, pc as i32, next as i32
            ),
        )
    }

    fn after_runtime_status(&mut self, status: &V, terminal: bool) {
        let stop = self.new_block();
        let resume = self.new_block();
        if terminal {
            self.br(&stop);
        } else {
            self.cond_br(status, &stop, &resume, Some(MD_COLD_TRUE));
        }
        self.begin(&stop);
        self.ret(TRACE_RESULT_RUNTIME);
        self.begin(&resume);
    }

    // ------------------------------------------------------------------- nodes

    fn one_result(&self, node: &crate::napalm_trace_ir::TraceNode) -> Result<TraceValueId, String> {
        if node.results.len() == 1 {
            Ok(node.results[0])
        } else {
            Err(self.error(format!("trace node {} expected exactly one result", node.id)))
        }
    }

    fn lower_node(&mut self, node: &crate::napalm_trace_ir::TraceNode) -> Result<(), String> {
        use TraceInst::*;
        match &node.kind {
            Nop => {}

            Runtime { location, next, terminal, direct_call } => {
                self.invalidate_heap_proofs();
                let encoded_next = if let Some(direct) = direct_call {
                    if *terminal || next.0 & TRACE_RUNTIME_MODE_MASK != 0 || direct.args.len() > MAX_NAPALM_ARITY {
                        return Err(self.error("invalid direct bytecode call runtime payload"));
                    }
                    let stage_captures = direct.closure_guard_exit.is_some()
                        && direct.closure_capture_count != 0
                        && direct.args.len() < MAX_NAPALM_ARITY;
                    if direct.closure_guard_exit.is_none() && direct.closure_capture_count != 0 {
                        return Err(self.error("non-closure direct call carries capture metadata"));
                    }
                    let mut staged_capture_ptr = None;
                    if let Some(exit) = direct.closure_guard_exit {
                        let (entry, _, metadata) =
                            self.heap_entry_with_metadata_for_value(direct.receiver, TraceHeapKind::Closure, exit)?;
                        let actual = self.heap_entry_len(&entry);
                        let expected =
                            self.vm_binding_word(VmBindingKind::PreparedFunctionIndex, direct.function_index as u64)?;
                        let mismatch = self.icmp("ne", &actual, &expected);
                        self.emit_side_exit(exit, &mismatch)?;
                        if stage_captures {
                            let count = self.heap_metadata_payload(&metadata);
                            let wrong = self.icmp("ne", &count, &V::i32(direct.closure_capture_count as i32));
                            self.emit_side_exit(exit, &wrong)?;
                            staged_capture_ptr = Some(self.heap_entry_data(&entry));
                        }
                    }
                    let function = self.vm_binding_i32(VmBindingKind::PreparedFunctionIndex, direct.function_index)?;
                    self.store_activation_i32(offset_of!(TraceActivation, direct_call_function), &function);
                    self.store_activation_i32(
                        offset_of!(TraceActivation, direct_call_destination),
                        &V::i32(direct.destination.0 as i32),
                    );
                    self.store_activation_i32(offset_of!(TraceActivation, direct_call_arity), &V::i32(direct.args.len() as i32));
                    let flags = if direct.closure_guard_exit.is_some() {
                        TRACE_DIRECT_CALL_FLAG_CLOSURE | if stage_captures { TRACE_DIRECT_CALL_FLAG_STAGED_CAPTURES } else { 0 }
                    } else {
                        0
                    };
                    self.store_activation_i32(offset_of!(TraceActivation, direct_call_reserved), &V::i32(flags as i32));
                    let receiver = self.lower_value(direct.receiver)?;
                    let receiver = self.word(&receiver);
                    let activation = Self::activation();
                    self.store_at(&receiver, &activation, offset_of!(TraceActivation, direct_call_receiver) as i64, Mem::Act);
                    let args_base = offset_of!(TraceActivation, direct_call_args) as i64;
                    for (index, argument) in direct.args.iter().copied().enumerate() {
                        let argument = self.lower_value(argument)?;
                        let argument = self.word(&argument);
                        self.store_at(&argument, &activation, args_base + (index * 8) as i64, Mem::Act);
                    }
                    if let Some(captures) = staged_capture_ptr {
                        let captures = self.word(&captures);
                        self.store_at(&captures, &activation, args_base + (direct.args.len() * 8) as i64, Mem::Act);
                    }
                    next.0 | TRACE_RUNTIME_MODE_DIRECT_CALL
                } else {
                    next.0
                };
                let status = self.runtime_call(location.function.0, location.pc.0, encoded_next);
                self.invalidate_heap_proofs();
                self.after_runtime_status(&status, *terminal);
            }

            RuntimeCall { location, next } => {
                self.invalidate_heap_proofs();
                if next.0 & TRACE_RUNTIME_MODE_MASK != 0 {
                    return Err(self.error("RuntimeCall continuation PC uses reserved runtime-mode bits"));
                }
                let status = self.runtime_call(location.function.0, location.pc.0, next.0 | TRACE_RUNTIME_MODE_CALL);
                self.invalidate_heap_proofs();
                self.after_runtime_status(&status, false);
            }

            LoadAbsoluteValue { address } => {
                let value = if let Some(value) = self.forwarded_absolute_loads.get(address) {
                    value.clone()
                } else {
                    let pointer = self.vm_binding_word(VmBindingKind::ValueAddress, *address)?;
                    let value = self.load(Ty::I64, &pointer, Mem::Heap);
                    self.forwarded_absolute_loads.insert(*address, value.clone());
                    value
                };
                self.cache_value(self.one_result(node)?, value);
            }

            StoreAbsoluteValue { address, value } => {
                let value = self.lower_value(*value)?;
                let value = self.word(&value);
                let pointer = self.vm_binding_word(VmBindingKind::ValueAddress, *address)?;
                self.store(&value, &pointer, Mem::Heap);
                self.forwarded_absolute_loads.insert(*address, value);
            }

            IncrementProfileCounter { address } => {
                let pointer = self.vm_binding_word(VmBindingKind::ProfileCounterAddress, *address)?;
                let current = self.load(Ty::I64, &pointer, Mem::Heap);
                let saturated = self.icmp("eq", &current, &V::i64(-1));
                let incremented = self.bin("add", &current, &V::i64(1));
                let next = self.select(&saturated, &current, &incremented);
                self.store(&next, &pointer, Mem::Heap);
            }

            IsInstanceObserved { value, class, expected_class, instance_classes, known_classes, exit } => {
                let (class_entry, _) = self.live_heap_entry_for_value(*class, *exit)?;
                let class_key = self.heap_entry_receiver_class(&class_entry);
                let expected = self.vm_binding_word(VmBindingKind::ClassKey, *expected_class)?;
                let wrong_class = self.icmp("ne", &class_key, &expected);
                self.emit_side_exit(*exit, &wrong_class)?;

                let tagged = self.lower_value(*value)?;
                let is_heap = self.tagged_is_heap(&tagged);
                let heap_block = self.new_block();
                let non_heap_block = self.new_block();
                let done = self.new_block();
                self.cond_br(&is_heap, &heap_block, &non_heap_block, None);

                self.begin(&non_heap_block);
                let non_heap_end = self.current.clone();
                self.br(&done);

                self.begin(&heap_block);
                // Branch-local proof: never published into the path-insensitive caches.
                let (entry, _) = self.live_heap_entry_from_heap_tagged(&tagged, *exit)?;
                let actual = self.load_at(Ty::I64, &entry, offset_of!(TraceHeapEntry, receiver_class) as i64, Mem::Sidecar);
                let matches = self.class_matches(&actual, instance_classes)?;
                let known = self.class_matches(&actual, known_classes)?;
                let kind = self.bin("lshr", &actual, &V::i64(56));
                let instance_kind = self.icmp("eq", &kind, &V::i64(1));
                let foreign_kind = self.icmp("eq", &kind, &V::i64(4));
                let dynamic_instance = self.or1(&instance_kind, &foreign_kind);
                let not_known = self.not1(&known);
                let unknown = self.and1(&dynamic_instance, &not_known);
                self.emit_side_exit(*exit, &unknown)?;
                let boxed = self.box_bool(&matches);
                let heap_end = self.current.clone();
                self.br(&done);

                self.begin(&done);
                let result = self.def(
                    Ty::I64,
                    &format!("phi i64 [ {}, %{non_heap_end} ], [ {}, %{heap_end} ]", TAG_FALSE as i64, boxed.op),
                );
                self.cache_value(self.one_result(node)?, result);
            }

            Copy { value } => {
                let result = self.one_result(node)?;
                let value = self.lower_value(*value)?;
                self.cache_value(result, value);
            }

            Reinterpret { value } => {
                let result = self.one_result(node)?;
                let source_rep = self.value_rep(*value)?;
                let target_rep = self.value_rep(result)?;
                let value = self.lower_value(*value)?;
                let word = self.normalize_to_word(&value, source_rep);
                let value = match target_rep {
                    TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool => self.trunc(&word, Ty::I32),
                    TraceRep::F64 => self.bitcast(&word, Ty::F64),
                    _ => word,
                };
                self.cache_value(result, value);
            }

            LoadFrameSlot { slot } => {
                let result = self.one_result(node)?;
                let rep = self.value_rep(result)?;
                let value = self.load_frame_slot(*slot, rep);
                self.cache_value(result, value);
            }

            CommitFrameSlot { slot, value } => {
                let rep = self.value_rep(*value)?;
                let value = self.lower_value(*value)?;
                let word = self.normalize_to_word(&value, rep);
                self.store_frame_word(*slot, &word);
            }

            GuardTruthy { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let failure = self.tagged_falsey(&tagged);
                self.emit_side_exit(*exit, &failure)?;
            }

            GuardFalsey { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let falsey = self.tagged_falsey(&tagged);
                let failure = self.not1(&falsey);
                self.emit_side_exit(*exit, &failure)?;
            }

            GuardNotNull { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let word = self.word(&tagged);
                let failure = self.icmp("eq", &word, &V::i64(TAG_NULL as i64));
                self.emit_side_exit(*exit, &failure)?;
                if let Some(result) = node.results.first().copied() {
                    self.cache_value(result, tagged);
                }
            }

            GuardF64 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_numeric(&tagged, *exit)?;
                self.cache_value(self.one_result(node)?, raw);
            }

            GuardInt32 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_int32_numeric(&tagged, *exit)?;
                self.cache_value(self.one_result(node)?, raw);
            }

            GuardUInt32 { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let raw = self.guard_uint32_numeric(&tagged, *exit)?;
                self.cache_value(self.one_result(node)?, raw);
            }

            GuardBounds { index, length, exit } => {
                let index_id = self.trace.canonical_value(*index);
                let length_id = self.trace.canonical_value(*length);
                let index_value = self.lower_value(*index)?;
                let length_value = self.lower_value(*length)?;
                if self.bounds_proofs.insert((index_id, length_id)) {
                    let zero = V { op: "0".to_owned(), ty: index_value.ty };
                    let negative = self.icmp("slt", &index_value, &zero);
                    let too_large = self.icmp("sge", &index_value, &length_value);
                    let failure = self.or1(&negative, &too_large);
                    self.emit_side_exit(*exit, &failure)?;
                }
                if let Some(&list) = self.list_lengths.get(&length_id) {
                    let normalized = self.zext(&index_value, Ty::I64);
                    self.list_indices.insert((list, index_id), normalized);
                }
            }

            GuardEq { lhs, rhs, exit } => {
                let lhs_value = self.lower_value(*lhs)?;
                let rhs_value = self.lower_value(*rhs)?;
                let failure = if self.value_rep(*lhs)? == TraceRep::F64 {
                    self.fcmp("une", &lhs_value, &rhs_value)
                } else {
                    self.icmp("ne", &lhs_value, &rhs_value)
                };
                self.emit_side_exit(*exit, &failure)?;
            }

            GuardSwitch { tag, observed, exit } => {
                let tag_value = self.lower_value(*tag)?;
                let rep = self.value_rep(*tag)?;
                if !matches!(rep, TraceRep::Int32 | TraceRep::UInt32 | TraceRep::Bool) {
                    return Err(self.error("tagged Switch lowering is not enabled"));
                }
                // Case labels are compared on the 32-bit representation bits.
                let case_bits = |case: i64| -> Option<i32> {
                    (i32::MIN as i64..=u32::MAX as i64).contains(&case).then_some(case as i32)
                };
                match observed {
                    TraceSwitchGuard::Case(0) if rep == TraceRep::Bool => {
                        let failure = self.cond(&tag_value);
                        self.emit_side_exit(*exit, &failure)?;
                    }
                    TraceSwitchGuard::Case(1) if rep == TraceRep::Bool => {
                        let condition = self.cond(&tag_value);
                        self.emit_conditional_side_exit(*exit, &condition, false)?;
                    }
                    TraceSwitchGuard::Case(case) => {
                        let failure = match case_bits(*case) {
                            Some(bits) => self.icmp("ne", &tag_value, &V::i32(bits)),
                            None => V { op: "true".to_owned(), ty: Ty::I1 },
                        };
                        self.emit_side_exit(*exit, &failure)?;
                    }
                    TraceSwitchGuard::Default { excluded } => {
                        let mut matches_any = V { op: "false".to_owned(), ty: Ty::I1 };
                        for case in excluded {
                            if let Some(bits) = case_bits(*case) {
                                let equal = self.icmp("eq", &tag_value, &V::i32(bits));
                                matches_any = self.or1(&matches_any, &equal);
                            }
                        }
                        self.emit_side_exit(*exit, &matches_any)?;
                    }
                }
            }

            GuardType { value, expected, exit } => {
                let value = self.lower_value(*value)?;
                match expected {
                    TraceType::Any => {}
                    TraceType::Num => {
                        let ok = self.tagged_is_num(&value);
                        let failure = self.not1(&ok);
                        self.emit_side_exit(*exit, &failure)?;
                    }
                    TraceType::Null => {
                        let word = self.word(&value);
                        let failure = self.icmp("ne", &word, &V::i64(TAG_NULL as i64));
                        self.emit_side_exit(*exit, &failure)?;
                    }
                    TraceType::Bool => {
                        let word = self.word(&value);
                        let shifted = self.bin("add", &word, &V::i64((TAG_FALSE as i64).wrapping_neg()));
                        let failure = self.icmp("ugt", &shifted, &V::i64(1));
                        self.emit_side_exit(*exit, &failure)?;
                    }
                    _ => return Err(self.error(format!("unsupported native GuardType {expected:?}"))),
                }
                if let Some(result) = node.results.first().copied() {
                    self.cache_value(result, value);
                }
            }

            GuardClass { value, classes, exit } => {
                let (entry, _) = self.live_heap_entry_for_value(*value, *exit)?;
                let tagged = self.lower_value(*value)?;
                let class = self.heap_entry_receiver_class(&entry);
                let matches = self.class_matches(&class, classes)?;
                let failure = self.not1(&matches);
                self.emit_side_exit(*exit, &failure)?;
                self.cache_value(self.one_result(node)?, tagged);
            }

            GuardMethodEpoch { expected, exit } => {
                let epoch = self.method_epoch_value();
                let expected = self.vm_binding_word(VmBindingKind::MethodEpoch, *expected)?;
                let failure = self.icmp("ne", &epoch, &expected);
                self.emit_side_exit(*exit, &failure)?;
            }

            IntUnary { op, value } => {
                let value = self.lower_value(*value)?;
                let zero = V { op: "0".to_owned(), ty: value.ty };
                let minus_one = V { op: "-1".to_owned(), ty: value.ty };
                let result = match op {
                    IntUnaryOp::Neg => self.bin("sub", &zero, &value),
                    IntUnaryOp::BitNot => self.bin("xor", &value, &minus_one),
                };
                self.cache_value(self.one_result(node)?, result);
            }

            IntBinary { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let mask = |state: &mut Self, rhs: &V| state.bin("and", rhs, &V { op: "31".to_owned(), ty: rhs.ty });
                let result = match op {
                    IntBinOp::Add => self.bin("add", &lhs, &rhs),
                    IntBinOp::Sub => self.bin("sub", &lhs, &rhs),
                    IntBinOp::Mul => self.bin("mul", &lhs, &rhs),
                    IntBinOp::BitAnd => self.bin("and", &lhs, &rhs),
                    IntBinOp::BitOr => self.bin("or", &lhs, &rhs),
                    IntBinOp::BitXor => self.bin("xor", &lhs, &rhs),
                    IntBinOp::Shl => {
                        let rhs = mask(self, &rhs);
                        self.bin("shl", &lhs, &rhs)
                    }
                    IntBinOp::ShrU => {
                        let rhs = mask(self, &rhs);
                        self.bin("lshr", &lhs, &rhs)
                    }
                    IntBinOp::Sar => {
                        let rhs = mask(self, &rhs);
                        self.bin("ashr", &lhs, &rhs)
                    }
                    IntBinOp::DivS | IntBinOp::RemS => {
                        return Err(self.error("unchecked integer division/remainder disabled"));
                    }
                };
                self.cache_value(self.one_result(node)?, result);
            }

            IntBinaryChecked { op, lhs, rhs, exit } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let with_overflow = |state: &mut Self, name: &str| -> Result<V, String> {
                    let pair = state.def(
                        Ty::I64, // placeholder type, only used through extractvalue
                        &format!("call {{ i32, i1 }} @llvm.{name}.with.overflow.i32(i32 {}, i32 {})", lhs.op, rhs.op),
                    );
                    let result = state.def(Ty::I32, &format!("extractvalue {{ i32, i1 }} {}, 0", pair.op));
                    let overflow = state.def(Ty::I1, &format!("extractvalue {{ i32, i1 }} {}, 1", pair.op));
                    state.emit_side_exit(*exit, &overflow)?;
                    Ok(result)
                };
                let result = match op {
                    IntBinOp::Add => with_overflow(self, "sadd")?,
                    IntBinOp::Sub => with_overflow(self, "ssub")?,
                    IntBinOp::Mul => with_overflow(self, "smul")?,
                    IntBinOp::DivS | IntBinOp::RemS => {
                        let zero = self.icmp("eq", &rhs, &V::i32(0));
                        self.emit_side_exit(*exit, &zero)?;
                        let is_min = self.icmp("eq", &lhs, &V::i32(i32::MIN));
                        let is_minus_one = self.icmp("eq", &rhs, &V::i32(-1));
                        let overflow = self.and1(&is_min, &is_minus_one);
                        self.emit_side_exit(*exit, &overflow)?;
                        if matches!(op, IntBinOp::DivS) {
                            self.bin("sdiv", &lhs, &rhs)
                        } else {
                            self.bin("srem", &lhs, &rhs)
                        }
                    }
                    _ => return Err(self.error(format!("checked {op:?} is not meaningful for this integer operation"))),
                };
                self.cache_value(self.one_result(node)?, result);
            }

            F64Unary { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    F64UnaryOp::Neg => self.def(Ty::F64, &format!("fneg double {}", value.op)),
                };
                self.cache_value(self.one_result(node)?, result);
            }

            F64Math { op, value } => {
                let value = self.lower_value(*value)?;
                let result = match op {
                    TraceMathOp::Abs => self.intrinsic_f64("fabs", &value),
                    TraceMathOp::Floor => self.intrinsic_f64("floor", &value),
                    TraceMathOp::Ceil => self.intrinsic_f64("ceil", &value),
                    TraceMathOp::Sqrt => self.intrinsic_f64("sqrt", &value),
                    TraceMathOp::Round => self.call_pure_f64("napalm_jit_round", &[&value]),
                    TraceMathOp::Sin => self.call_pure_f64("napalm_jit_sin", &[&value]),
                    TraceMathOp::Cos => self.call_pure_f64("napalm_jit_cos", &[&value]),
                    TraceMathOp::Tan => self.call_pure_f64("napalm_jit_tan", &[&value]),
                    TraceMathOp::Log => self.call_pure_f64("napalm_jit_log", &[&value]),
                    TraceMathOp::Exp => self.call_pure_f64("napalm_jit_exp", &[&value]),
                };
                self.cache_value(self.one_result(node)?, result);
            }

            F64Binary { op, lhs, rhs } => {
                let divisor = self
                    .trace
                    .value(self.trace.canonical_value(*rhs))
                    .and_then(|value| match value.origin {
                        TraceValueOrigin::Constant(TraceConstant::F64Bits(bits)) => Some(f64::from_bits(bits)),
                        _ => None,
                    });
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = match op {
                    F64BinOp::Add => self.bin("fadd", &lhs, &rhs),
                    F64BinOp::Sub => self.bin("fsub", &lhs, &rhs),
                    F64BinOp::Mul => self.bin("fmul", &lhs, &rhs),
                    F64BinOp::Div => {
                        if let Some(reciprocal) = divisor.and_then(Self::exact_power_of_two_reciprocal) {
                            self.bin("fmul", &lhs, &V::f64(reciprocal))
                        } else {
                            self.bin("fdiv", &lhs, &rhs)
                        }
                    },
                    F64BinOp::Rem => self.f64_remainder(&lhs, &rhs, divisor),
                };
                self.cache_value(self.one_result(node)?, result);
            }

            IntCompare { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let pred = match op {
                    CmpKind::Eq => "eq",
                    CmpKind::Ne => "ne",
                    CmpKind::Lt => "slt",
                    CmpKind::Le => "sle",
                    CmpKind::Gt => "sgt",
                    CmpKind::Ge => "sge",
                };
                let result = self.icmp(pred, &lhs, &rhs);
                let result = self.bool32(&result);
                self.cache_value(self.one_result(node)?, result);
            }

            F64Compare { op, lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let pred = match op {
                    CmpKind::Eq => "oeq",
                    CmpKind::Ne => "une",
                    CmpKind::Lt => "olt",
                    CmpKind::Le => "ole",
                    CmpKind::Gt => "ogt",
                    CmpKind::Ge => "oge",
                };
                let result = self.fcmp(pred, &lhs, &rhs);
                let result = self.bool32(&result);
                self.cache_value(self.one_result(node)?, result);
            }

            RefEq { lhs, rhs } => {
                let lhs = self.lower_value(*lhs)?;
                let rhs = self.lower_value(*rhs)?;
                let result = self.icmp("eq", &lhs, &rhs);
                let result = self.bool32(&result);
                self.cache_value(self.one_result(node)?, result);
            }

            IsNull { value } => {
                let value = self.lower_value(*value)?;
                let word = self.word(&value);
                let result = self.icmp("eq", &word, &V::i64(TAG_NULL as i64));
                let result = self.bool32(&result);
                self.cache_value(self.one_result(node)?, result);
            }

            Select { condition, if_true, if_false } => {
                let condition = self.lower_value(*condition)?;
                let if_true = self.lower_value(*if_true)?;
                let if_false = self.lower_value(*if_false)?;
                let result = self.select(&condition, &if_true, &if_false);
                self.cache_value(self.one_result(node)?, result);
            }

            BoxInt32 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.cast("sitofp", &value, Ty::F64);
                let tagged = self.bitcast(&number, Ty::I64);
                self.cache_value(self.one_result(node)?, tagged);
            }

            BoxUInt32 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.cast("uitofp", &value, Ty::F64);
                let tagged = self.bitcast(&number, Ty::I64);
                self.cache_value(self.one_result(node)?, tagged);
            }

            BoxF64 { value } => {
                let value = self.lower_value(*value)?;
                let tagged = self.box_f64(&value);
                self.cache_value(self.one_result(node)?, tagged);
            }

            Int32ToF64 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.cast("sitofp", &value, Ty::F64);
                self.cache_value(self.one_result(node)?, number);
            }

            UInt32ToF64 { value } => {
                let value = self.lower_value(*value)?;
                let number = self.cast("uitofp", &value, Ty::F64);
                self.cache_value(self.one_result(node)?, number);
            }

            F64ToInt32 { value, exit } => {
                let number = self.lower_value(*value)?;
                let bits = self.bitcast(&number, Ty::I64);
                let negative_zero = self.icmp("eq", &bits, &V::i64((-0.0f64).to_bits() as i64));
                self.emit_side_exit(*exit, &negative_zero)?;
                let ge = self.fcmp("oge", &number, &V::f64(i32::MIN as f64));
                let le = self.fcmp("ole", &number, &V::f64(i32::MAX as f64));
                let in_range = self.and1(&ge, &le);
                let fail = self.not1(&in_range);
                self.emit_side_exit(*exit, &fail)?;
                let result = self.cast("fptosi", &number, Ty::I32);
                let back = self.cast("sitofp", &result, Ty::F64);
                let exact = self.fcmp("oeq", &number, &back);
                let not_exact = self.not1(&exact);
                self.emit_side_exit(*exit, &not_exact)?;
                self.cache_value(self.one_result(node)?, result);
            }

            TaggedToHeapHandle { value, exit } => {
                let tagged = self.lower_value(*value)?;
                let is_heap = self.tagged_is_heap(&tagged);
                let fail = self.not1(&is_heap);
                self.emit_side_exit(*exit, &fail)?;
                let word = self.word(&tagged);
                let handle = self.bin("and", &word, &V::i64(PAYLOAD_MASK as i64));
                self.cache_value(self.one_result(node)?, handle);
            }

            GuardHeapObject { value, kind, exit } => {
                let (entry, address) =
                    self.heap_entry_for_value(*value, CraneliftLoweringState::runtime_heap_kind(*kind), *exit)?;
                let result = self.one_result(node)?;
                self.cache_value(result, entry);
                if let Some(slot) = self.heap_addresses.get_mut(result.0 as usize) {
                    *slot = Some(address);
                }
            }

            HeapLoadField { object, slot, exit } => {
                let (entry, _) = self.heap_entry_for_value(*object, TraceHeapKind::Instance, *exit)?;
                let len = self.heap_entry_len(&entry);
                self.guard_constant_index(&len, *slot, *exit)?;
                let location = Fwd::Field { entry: entry.clone(), slot: *slot };
                let result = if let Some(value) = self.forwarded_heap_loads.get(&location) {
                    value.clone()
                } else {
                    let data = self.heap_entry_data(&entry);
                    let index = self.vm_binding_word(VmBindingKind::FieldSlot, *slot as u64)?;
                    let address = self.gep_words(&data, &index);
                    let value = self.load(Ty::I64, &address, Mem::Heap);
                    self.forwarded_heap_loads.insert(location, value.clone());
                    value
                };
                self.cache_value(self.one_result(node)?, result);
            }

            HeapStoreField { object, value, slot, exit } => {
                let tagged = self.lower_value(*value)?;
                let tagged = self.word(&tagged);
                let (entry, owner) = self.heap_entry_for_value(*object, TraceHeapKind::Instance, *exit)?;
                let len = self.heap_entry_len(&entry);
                self.guard_constant_index(&len, *slot, *exit)?;
                let data = self.heap_entry_data(&entry);
                let index = self.vm_binding_word(VmBindingKind::FieldSlot, *slot as u64)?;
                let address = self.gep_words(&data, &index);
                self.emit_heap_write_barrier(&owner, *value, &tagged)?;
                self.store(&tagged, &address, Mem::Heap);
                self.publish_heap_store(Fwd::Field { entry, slot: *slot }, tagged);
            }

            HeapLoadCell { cell, exit } => {
                let (entry, _) = self.heap_entry_for_value(*cell, TraceHeapKind::Cell, *exit)?;
                let location = Fwd::Cell { entry: entry.clone() };
                let result = if let Some(value) = self.forwarded_heap_loads.get(&location) {
                    value.clone()
                } else {
                    let data = self.heap_entry_data(&entry);
                    let value = self.load(Ty::I64, &data, Mem::Heap);
                    self.forwarded_heap_loads.insert(location, value.clone());
                    value
                };
                self.cache_value(self.one_result(node)?, result);
            }

            HeapStoreCell { cell, value, exit } => {
                let tagged = self.lower_value(*value)?;
                let tagged = self.word(&tagged);
                let (entry, owner) = self.heap_entry_for_value(*cell, TraceHeapKind::Cell, *exit)?;
                let data = self.heap_entry_data(&entry);
                self.emit_heap_write_barrier(&owner, *value, &tagged)?;
                self.store(&tagged, &data, Mem::Heap);
                self.publish_heap_store(Fwd::Cell { entry }, tagged);
            }

            HeapWriteBarrier { object, value, exit } => {
                if self.trace_value_may_be_heap(*value)? {
                    let tagged = self.lower_value(*value)?;
                    let is_heap = self.tagged_is_heap(&tagged);
                    let barrier = self.new_block();
                    let done = self.new_block();
                    self.cond_br(&is_heap, &barrier, &done, None);
                    self.begin(&barrier);
                    // Branch-local owner proof (see the Cranelift tier).
                    let tagged_object = self.lower_value(*object)?;
                    let (_, owner) = self.guard_live_heap_entry(&tagged_object, *exit)?;
                    self.call_heap_barrier(&owner);
                    self.br(&done);
                    self.begin(&done);
                }
            }

            HeapListLen { list, exit } => {
                let (entry, _) = self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                let len = self.heap_entry_len(&entry);
                let len = self.trunc(&len, Ty::I32);
                let result = self.one_result(node)?;
                self.cache_value(result, len);
                let list = self.trace.canonical_value(*list);
                self.list_lengths.insert(result, list);
            }

            HeapListGet { list, index, exit } => {
                let (entry, _) = self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                let index = self.validated_list_index(*list, *index, &entry, *exit)?;
                let location = Fwd::Indexed { entry: entry.clone(), index: index.clone() };
                let result = if let Some(value) = self.forwarded_heap_loads.get(&location) {
                    value.clone()
                } else {
                    let data = self.heap_entry_data(&entry);
                    let address = self.gep_words(&data, &index);
                    let value = self.load(Ty::I64, &address, Mem::Heap);
                    self.forwarded_heap_loads.insert(location, value.clone());
                    value
                };
                self.cache_value(self.one_result(node)?, result);
            }

            GuardNoContract { value, kind, exit } => {
                let heap_kind = match kind {
                    TraceHeapObjectKind::List => TraceHeapKind::List,
                    TraceHeapObjectKind::Map => TraceHeapKind::Map,
                    _ => return Err(self.error("invalid contract guard kind")),
                };
                let (entry, _) = self.heap_entry_for_value(*value, heap_kind, *exit)?;
                self.guard_uncontracted_store(&entry, *exit)?;
            }

            HeapListSet { list, index, value, exit } => {
                let tagged = self.lower_value(*value)?;
                let tagged = self.word(&tagged);
                let (entry, owner) = self.heap_entry_for_value(*list, TraceHeapKind::List, *exit)?;
                self.guard_uncontracted_store(&entry, *exit)?;
                let index = self.validated_list_index(*list, *index, &entry, *exit)?;
                let data = self.heap_entry_data(&entry);
                let address = self.gep_words(&data, &index);
                self.emit_heap_write_barrier(&owner, *value, &tagged)?;
                self.store(&tagged, &address, Mem::Heap);
                self.publish_heap_store(Fwd::Indexed { entry, index }, tagged);
            }

            HeapMapLen { map, exit } => {
                let (entry, _) = self.heap_entry_for_value(*map, TraceHeapKind::Map, *exit)?;
                let len = self.heap_entry_len(&entry);
                let len = self.trunc(&len, Ty::I32);
                self.cache_value(self.one_result(node)?, len);
            }

            HeapMapGetShaped { map, key, expected_key, shape, slot, exit } => {
                let tagged_key = self.lower_value(*key)?;
                let (entry, owner, metadata) = self.heap_entry_with_metadata_for_value(*map, TraceHeapKind::Map, *exit)?;
                self.guard_map_shape_once(*map, &metadata, *shape, *exit)?;
                self.guard_string_bytes_once(*key, &tagged_key, expected_key, *exit)?;
                let location = Fwd::ShapedMap { entry: entry.clone(), slot: *slot };
                let result = if let Some(value) = self.forwarded_heap_loads.get(&location) {
                    value.clone()
                } else {
                    let data = self.heap_entry_map_data(&entry, &owner);
                    let index = self.vm_binding_word(VmBindingKind::DenseSlot, *slot as u64)?;
                    let address = self.gep_words(&data, &index);
                    let value = self.load(Ty::I64, &address, Mem::Heap);
                    self.forwarded_heap_loads.insert(location, value.clone());
                    value
                };
                self.cache_value(self.one_result(node)?, result);
            }

            HeapMapGetPic { map, key, expected_key, entries, exit } => {
                let tagged_key = self.lower_value(*key)?;
                let (entry, owner, metadata) = self.heap_entry_with_metadata_for_value(*map, TraceHeapKind::Map, *exit)?;
                self.guard_string_bytes_once(*key, &tagged_key, expected_key, *exit)?;
                let shape = self.heap_metadata_payload(&metadata);
                let mut matched = V { op: "false".to_owned(), ty: Ty::I1 };
                let mut selected = V::i64(0);
                for candidate in entries {
                    let expected_shape = self.vm_binding_i32(VmBindingKind::ShapeId, candidate.shape)?;
                    let is_shape = self.icmp("eq", &shape, &expected_shape);
                    matched = self.or1(&matched, &is_shape);
                    let slot = self.vm_binding_word(VmBindingKind::DenseSlot, candidate.slot as u64)?;
                    selected = self.select(&is_shape, &slot, &selected);
                }
                let miss = self.not1(&matched);
                self.emit_side_exit(*exit, &miss)?;
                let data = self.heap_entry_map_data(&entry, &owner);
                let address = self.gep_words(&data, &selected);
                let value = self.load(Ty::I64, &address, Mem::Heap);
                self.cache_value(self.one_result(node)?, value);
            }

            HeapMapSetShaped { map, key, value, expected_key, shape, slot, exit } => {
                let tagged_key = self.lower_value(*key)?;
                let tagged = self.lower_value(*value)?;
                let tagged = self.word(&tagged);
                let (entry, owner, metadata) = self.heap_entry_with_metadata_for_value(*map, TraceHeapKind::Map, *exit)?;
                self.guard_map_shape_once(*map, &metadata, *shape, *exit)?;
                self.guard_string_bytes_once(*key, &tagged_key, expected_key, *exit)?;
                let data = self.heap_entry_map_data(&entry, &owner);
                let index = self.vm_binding_word(VmBindingKind::DenseSlot, *slot as u64)?;
                let address = self.gep_words(&data, &index);
                self.emit_heap_write_barrier(&owner, *value, &tagged)?;
                self.store(&tagged, &address, Mem::Heap);
                self.publish_heap_store(Fwd::ShapedMap { entry, slot: *slot }, tagged);
            }

            HeapStringLen { string, exit } => {
                let (entry, _) = self.heap_entry_for_value(*string, TraceHeapKind::String, *exit)?;
                let len = self.heap_entry_len(&entry);
                let len = self.trunc(&len, Ty::I32);
                self.cache_value(self.one_result(node)?, len);
            }

            HeapStringAsciiCodePointAt { string, index, exit } => {
                let (entry, _, metadata) = self.heap_entry_with_metadata_for_value(*string, TraceHeapKind::String, *exit)?;
                let code_points = self.heap_entry_len(&entry);
                let byte_len32 = self.heap_metadata_payload(&metadata);
                let byte_len = self.zext(&byte_len32, Ty::I64);
                let same_length = self.icmp("eq", &code_points, &byte_len);
                let metadata_exact = self.icmp("ne", &byte_len32, &V::i32(TRACE_HEAP_METADATA_MAX as i32));
                let ascii = self.and1(&same_length, &metadata_exact);
                let not_ascii = self.not1(&ascii);
                self.emit_side_exit(*exit, &not_ascii)?;
                let index = self.lower_value(*index)?;
                let index = self.normalize_list_index(&index, &byte_len, *exit)?;
                let data = self.heap_entry_data(&entry);
                let address = self.gep8_dyn(&data, &index);
                let byte = self.load(Ty::I8, &address, Mem::Str);
                let code_point = self.zext(&byte, Ty::I32);
                self.cache_value(self.one_result(node)?, code_point);
            }

            HeapStringCodePointAt { string, index, exit } => {
                let (entry, _, metadata) = self.heap_entry_with_metadata_for_value(*string, TraceHeapKind::String, *exit)?;
                let byte_len32 = self.heap_metadata_payload(&metadata);
                let saturated = self.icmp("eq", &byte_len32, &V::i32(TRACE_HEAP_METADATA_MAX as i32));
                self.emit_side_exit(*exit, &saturated)?;
                let byte_len = self.zext(&byte_len32, Ty::I64);
                let index = self.lower_value(*index)?;
                let index = self.normalize_list_index(&index, &byte_len, *exit)?;
                let data = self.heap_entry_data(&entry);
                let data = self.as_ptr(&data);
                let result = self.def(
                    Ty::I32,
                    &format!(
                        "call i32 @napalm_jit_decode_code_point(ptr {}, i64 {}, i64 {}) #2",
                        data.op, byte_len.op, index.op
                    ),
                );
                self.cache_value(self.one_result(node)?, result);
            }

            HeapAwaitFulfilled { promise, exit } => {
                let (entry, _, metadata) = self.heap_entry_with_metadata_for_value(*promise, TraceHeapKind::Promise, *exit)?;
                let state = self.heap_metadata_payload(&metadata);
                let not_fulfilled = self.icmp("ne", &state, &V::i32(TRACE_PROMISE_FULFILLED as i32));
                self.emit_side_exit(*exit, &not_fulfilled)?;
                let value = self.load_at(Ty::I64, &entry, offset_of!(TraceHeapEntry, promise_value) as i64, Mem::Sidecar);
                self.cache_value(self.one_result(node)?, value);
            }

            ReadBarrier { value } => {
                self.invalidate_heap_proofs();
                let value = self.lower_value(*value)?;
                self.cache_value(self.one_result(node)?, value);
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- function

    fn lower_function(&mut self, native_budget: u32) -> Result<(), String> {
        let trace = self.trace;
        let native_budget_enabled =
            native_budget != 0 && matches!(trace.terminator, TraceTerminator::LoopBack);

        // `start` is the first non-entry block.
        let _ = writeln!(self.body, "start:");
        self.current = "start".to_owned();

        for index in 0..trace.values.len() {
            let value = &trace.values[index];
            if matches!(
                &value.origin,
                TraceValueOrigin::EntrySlot { .. } | TraceValueOrigin::GuardedEntry { .. } | TraceValueOrigin::Constant(_)
            ) {
                self.lower_value(value.id)?;
            }
        }

        let mut preheader_nodes = vec![false; trace.nodes.len()];
        for &id in &trace.preheader {
            preheader_nodes[id] = true;
            let node = &trace.nodes[id];
            if !matches!(node.kind, TraceInst::Nop) {
                self.lower_node(node)?;
            }
        }

        // Same conservative invariant-guard prefix hoisting as the Cranelift tier.
        let epoch_free_loop = matches!(trace.terminator, TraceTerminator::LoopBack)
            && !trace.nodes.iter().any(|node| {
                matches!(node.kind, TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. } | TraceInst::ReadBarrier { .. })
            });
        if epoch_free_loop {
            for node in &trace.nodes {
                if preheader_nodes[node.id] || matches!(node.kind, TraceInst::Nop) {
                    continue;
                }
                let Some(exit) = node.kind.exit() else { break; };
                if !node.kind.is_entry_guard()
                    || !trace_inst_available_in_preheader(trace, &node.kind, &preheader_nodes)
                    || !trace_exit_available_in_preheader(trace, exit, &preheader_nodes)
                {
                    break;
                }
                self.lower_node(node)?;
                preheader_nodes[node.id] = true;
            }
        }

        // Loop-carried state lives in entry-block allocas; SROA/mem2reg turns them
        // into SSA phis (the LLVM analogue of Cranelift's frontend Variables).
        let mut phi_slots = Vec::with_capacity(trace.loop_phis.len());
        for (index, phi) in trace.loop_phis.iter().enumerate() {
            let ty = Ty::of_rep(phi.rep);
            let slot = format!("%phi{index}");
            let _ = writeln!(self.entry, "  {slot} = alloca {}, align {}", ty.ir(), ty.align());
            let mut initial = self.lower_value(phi.initial)?;
            if let Some(exit) = phi.entry_exit {
                initial = match phi.rep {
                    TraceRep::F64 => self.guard_numeric(&initial, exit)?,
                    TraceRep::Int32 => self.guard_int32_numeric(&initial, exit)?,
                    TraceRep::UInt32 => self.guard_uint32_numeric(&initial, exit)?,
                    other => return Err(self.error(format!("unsupported promoted LoopPhi representation {other:?}"))),
                };
            }
            let initial = self.coerce(&initial, ty);
            self.line(&format!("store {} {}, ptr {slot}, align {}", ty.ir(), initial.op, ty.align()));
            phi_slots.push((slot, ty));
        }
        let budget_slot = if native_budget_enabled {
            let _ = writeln!(self.entry, "  %budget = alloca i64, align 8");
            self.line(&format!("store i64 {native_budget}, ptr %budget, align 8"));
            Some("%budget".to_owned())
        } else {
            None
        };

        self.br("loop");
        self.begin("loop");

        if trace.nodes.iter().any(|node| {
            matches!(node.kind, TraceInst::Runtime { .. } | TraceInst::RuntimeCall { .. } | TraceInst::ReadBarrier { .. })
        }) {
            self.invalidate_heap_proofs();
        }

        for (phi, (slot, ty)) in trace.loop_phis.iter().zip(phi_slots.iter()) {
            let value = self.def(*ty, &format!("load {}, ptr {slot}, align {}", ty.ir(), ty.align()));
            self.cache_value(phi.value, value);
        }
        self.native_budget_remaining = budget_slot
            .as_ref()
            .map(|slot| self.def(Ty::I64, &format!("load i64, ptr {slot}, align 8")));

        for node in &trace.nodes {
            if !preheader_nodes[node.id] && !matches!(node.kind, TraceInst::Nop) {
                self.lower_node(node)?;
            }
        }

        match &trace.terminator {
            TraceTerminator::RuntimeReturn => self.ret(TRACE_RESULT_RUNTIME),
            TraceTerminator::LoopBack => {
                let mut backedges = Vec::with_capacity(trace.loop_phis.len());
                for phi in &trace.loop_phis {
                    let backedge = phi.backedge.ok_or_else(|| self.error(format!("{:?} has no backedge", phi.value)))?;
                    backedges.push(self.lower_value(backedge)?);
                }
                for (value, (slot, ty)) in backedges.iter().zip(phi_slots.iter()) {
                    let value = self.coerce(value, *ty);
                    self.line(&format!("store {} {}, ptr {slot}, align {}", ty.ir(), value.op, ty.align()));
                }
                if let Some(slot) = &budget_slot {
                    let budget = self
                        .native_budget_remaining
                        .clone()
                        .ok_or_else(|| self.error("native loop budget has no SSA value"))?;
                    let next_budget = self.bin("add", &budget, &V::i64(-1));
                    self.line(&format!("store i64 {}, ptr {slot}, align 8", next_budget.op));
                    let exhausted = self.icmp("eq", &next_budget, &V::i64(0));
                    let stop = self.new_block();
                    self.cond_br(&exhausted, &stop, "loop", Some(MD_COLD_TRUE));
                    self.begin(&stop);
                    let exit = trace
                        .native_budget_exit
                        .ok_or_else(|| self.error("native loop budget enabled without watchdog side exit"))?;
                    self.native_budget_remaining = Some(next_budget);
                    self.emit_unconditional_side_exit(exit)?;
                } else {
                    self.br("loop");
                }
            }
            TraceTerminator::Exit(exit) => self.emit_unconditional_side_exit(*exit)?,
            TraceTerminator::Return => return Err(self.error("native Return terminator is not enabled")),
            TraceTerminator::Link => return Err(self.error("direct native trace links are installed after compilation")),
            TraceTerminator::Unreachable => return Err(self.error("trace has unreachable terminator")),
        }
        Ok(())
    }

    /// Representation coercion for loop-carried slots.
    fn coerce(&mut self, value: &V, ty: Ty) -> V {
        match (value.ty, ty) {
            (a, b) if a == b => value.clone(),
            (Ty::Ptr, Ty::I64) => self.word(value),
            (Ty::I1, Ty::I32) => self.zext(value, Ty::I32),
            (Ty::I32, Ty::I64) => self.zext(value, Ty::I64),
            (Ty::I64, Ty::I32) => self.trunc(value, Ty::I32),
            (Ty::F64, Ty::I64) | (Ty::I64, Ty::F64) => self.bitcast(value, ty),
            _ => value.clone(),
        }
    }

    fn finish_module(self, symbol: &str, host: &HostInfo) -> String {
        let mut out = String::with_capacity(self.entry.len() + self.body.len() + 4096);
        let _ = writeln!(out, "; Napalm trace {:?} (LLVM tier)", self.trace.id);
        let _ = writeln!(out, "source_filename = \"napalm\"");
        let _ = writeln!(out, "target datalayout = \"{}\"", host.data_layout);
        let _ = writeln!(out, "target triple = \"{}\"", host.triple);
        out.push('\n');
        out.push_str(concat!(
            "declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)\n",
            "declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)\n",
            "declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)\n",
            "declare double @llvm.fabs.f64(double)\n",
            "declare double @llvm.floor.f64(double)\n",
            "declare double @llvm.ceil.f64(double)\n",
            "declare double @llvm.sqrt.f64(double)\n",
            "declare double @llvm.copysign.f64(double, double)\n",
            "declare double @llvm.trunc.f64(double)\n",
            "declare double @napalm_jit_frem(double, double) #1\n",
            "declare double @napalm_jit_round(double) #1\n",
            "declare double @napalm_jit_sin(double) #1\n",
            "declare double @napalm_jit_cos(double) #1\n",
            "declare double @napalm_jit_tan(double) #1\n",
            "declare double @napalm_jit_log(double) #1\n",
            "declare double @napalm_jit_exp(double) #1\n",
            "declare i32 @napalm_jit_decode_code_point(ptr, i64, i64) #2\n",
            "\n",
        ));
        let _ = writeln!(
            out,
            "define i32 @{symbol}(ptr noundef nonnull align 16 dereferenceable({}) %act, ptr noundef nonnull align 4 %links) #0 {{",
            std::mem::size_of::<TraceActivation>()
        );
        out.push_str("entry:\n");
        out.push_str(&self.entry);
        out.push_str("  br label %start\n");
        out.push_str(&self.body);
        out.push_str("}\n\n");
        let escape = |text: &str| text.replace('\\', "\\5C").replace('"', "\\22");
        let _ = writeln!(
            out,
            "attributes #0 = {{ nounwind hot \"target-cpu\"=\"{}\" \"target-features\"=\"{}\" \"frame-pointer\"=\"none\" }}",
            escape(&host.cpu),
            escape(&host.features)
        );
        out.push_str("attributes #1 = { nounwind willreturn nofree nosync memory(none) }\n");
        out.push_str("attributes #2 = { nounwind willreturn nofree nosync memory(argmem: read) }\n");
        out.push_str("attributes #3 = { nounwind }\n\n");
        out.push_str(concat!(
            "!0 = !{!\"napalm.tbaa\"}\n",
            "!1 = !{!\"napalm.act\", !0, i64 0}\n",
            "!2 = !{!\"napalm.reg\", !0, i64 0}\n",
            "!3 = !{!\"napalm.sidecar\", !0, i64 0}\n",
            "!4 = !{!\"napalm.heap\", !0, i64 0}\n",
            "!5 = !{!\"napalm.str\", !0, i64 0}\n",
            "!6 = !{!\"napalm.binding\", !0, i64 0}\n",
            "!7 = !{!\"napalm.links\", !0, i64 0}\n",
            "!11 = !{!1, !1, i64 0}\n",
            "!12 = !{!2, !2, i64 0}\n",
            "!13 = !{!3, !3, i64 0}\n",
            "!14 = !{!4, !4, i64 0}\n",
            "!15 = !{!5, !5, i64 0}\n",
            "!16 = !{!6, !6, i64 0}\n",
            "!17 = !{!7, !7, i64 0}\n",
            "!20 = !{!\"branch_weights\", i32 1, i32 2000000}\n",
            "!21 = !{!\"branch_weights\", i32 2000000, i32 1}\n",
            "!22 = !{}\n",
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_by_one_is_exact_with_signed_zero_and_nonfinite_values() {
        let values = [
            0.0_f64, -0.0, 3.25, -3.25, 1.0, -1.0,
            f64::MAX, -f64::MAX, f64::MIN_POSITIVE,
            f64::from_bits(1), -f64::from_bits(1),
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
        ];
        for number in values {
            let magnitude = number.abs();
            let actual = (magnitude - magnitude.trunc()).copysign(number);
            let expected = number % 1.0;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn exact_power_of_two_division_handles_signed_and_extreme_operands() {
        for exponent in [-1022, -1000, -7, -1, 0, 1, 10, 1023] {
            let divisor = 2.0_f64.powi(exponent);
            for signed in [divisor, -divisor] {
                let reciprocal = LlvmLowering::exact_power_of_two_reciprocal(signed).unwrap();
                for numerator in [
                    -0.0, 0.0, -3.0, 3.25, f64::MIN_POSITIVE,
                    f64::from_bits(1), f64::MAX, f64::NEG_INFINITY, f64::INFINITY,
                ] {
                    assert_eq!(
                        (numerator / signed).to_bits(),
                        (numerator * reciprocal).to_bits(),
                        "numerator {numerator:?}, divisor {signed:?}"
                    );
                }
            }
        }
        for not_a_power in [0.0, -0.0, 3.0, -6.0, f64::INFINITY, f64::NAN, f64::from_bits(1)] {
            assert!(LlvmLowering::exact_power_of_two_reciprocal(not_a_power).is_none());
        }
    }

    #[test]
    fn custom_llvm_pipeline_is_preserved() {
        assert_eq!(effective_optimization_pipeline("default<O2>"), "default<O2>");
        assert!(effective_optimization_pipeline("default<O3>").contains("default<O3>"));
    }

    #[test]
    fn llvm_compiles_and_runs_a_trace_shaped_function() {
        let host = host_info().expect("LLVM host");
        let machine = create_host_target_machine(host).expect("target machine");
        let context: &'static Context = Box::leak(Box::new(Context::create()));
        let mut engines = Vec::new();
        let symbol = "napalm_llvm_smoke";
        let ir = format!(
            "target datalayout = \"{}\"\ntarget triple = \"{}\"\n\
             define i32 @{symbol}(ptr %act, ptr %links) {{\n\
             entry:\n  %v = load i32, ptr %links, align 4\n  %r = add i32 %v, 40\n  ret i32 %r\n}}\n",
            host.data_layout, host.triple
        );
        let code = compile_ir(
            context,
            &machine,
            symbol,
            &ir,
            "default<O3>",
            false,
            false,
            &mut engines,
            std::time::Instant::now(),
        )
        .expect("compiled");
        let function: NativeTraceFn = unsafe { std::mem::transmute(code.address) };
        let links = [2u32];
        let result = unsafe { function(std::ptr::null_mut(), links.as_ptr()) };
        assert_eq!(result, 42);
        std::mem::forget(engines); // the leaked test context outlives them
    }
}

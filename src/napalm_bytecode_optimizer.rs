//! Post-lowering, control-flow-aware optimizer for Napalm register bytecode.
//!
//! Unlike the HIR optimizer, this pipeline preserves the baseline VM's opcode
//! semantics and physical-register assignments. All code movement is applied
//! to a temporary program and publishes only after bytecode verification.
//!
//! In particular, relative word-addressed branches, switch tables, exception
//! ranges, precise stack maps, source ranges, OSR/deopt resume PCs and inlined
//! resume PCs in *other* functions are relocated together.
//!
//! Specialized super-instructions are selected by the bytecode compiler
//! (`CompileOptions::emit_specialized_bytecode`), not by this pass pipeline.

use crate::napalm_collections::{HashMap, HashSet};
use std::fmt;
use smallbox::{SmallBox, smallbox, space::S1};
use smallvec::SmallVec;

use crate::napalm_bytecode::{
    BytecodePc, BytecodeProgram, BytecodeVerifier, Decoder, Opcode, Word,
};
use crate::napalm_ir::{Constant, FuncId};

#[derive(Debug, Clone)]
pub struct OptimizationError {
    pub pass: &'static str,
    pub message: String,
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.pass, self.message)
    }
}
impl std::error::Error for OptimizationError {}

fn error(pass: &'static str, message: impl Into<String>) -> OptimizationError {
    OptimizationError { pass, message: message.into() }
}

/// A pass can change every function. It must preserve the existing instruction
/// boundary layout; `CompactBytecode` alone may change word offsets, as it
/// remaps the entire program's metadata and relative branches atomically.
pub trait BytecodeOptimization {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError>;

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

#[derive(Clone, Debug, Default)]
pub struct OptimizationStats {
    pub input_words: usize,
    pub output_words: usize,
    pub passes_changed: usize,
    pub words_removed: usize,
}

pub struct Optimize {
    passes: Vec<SmallBox<dyn BytecodeOptimization, S1>>,
    verify_after_each_pass: bool,
}

impl Default for Optimize {
    fn default() -> Self {
        Self { passes: Vec::new(), verify_after_each_pass: cfg!(debug_assertions) }
    }
}

impl Optimize {
    pub fn new() -> Self { Self::default() }

    pub fn with_pass(mut self, pass: impl BytecodeOptimization + 'static) -> Self {
        self.passes.push(smallbox!(pass));
        self
    }

    pub fn verify_after_each_pass(mut self, enabled: bool) -> Self {
        self.verify_after_each_pass = enabled;
        self
    }

    pub fn optimize_program(&mut self, program: &mut BytecodeProgram) -> Result<OptimizationStats, OptimizationError> {
        // Publish a fully verified program or leave the caller's bytecode intact.
        BytecodeVerifier::verify_program(program)
            .map_err(|e| error("pre-optimization verifier", e.to_string()))?;
        let mut work = program.clone();
        let input_words = work.functions.iter().map(|f| f.chunk.code.len()).sum::<usize>();
        let mut passes_changed = 0;
        for pass in &mut self.passes {
            let changed = pass.optimize(&mut work)?;
            if changed { passes_changed += 1; }
            if self.verify_after_each_pass {
                BytecodeVerifier::verify_program(&work)
                    .map_err(|e| error(pass.name(), e.to_string()))?;
            }
        }
        BytecodeVerifier::verify_program(&work)
            .map_err(|e| error("final bytecode verifier", e.to_string()))?;
        let output_words = work.functions.iter().map(|f| f.chunk.code.len()).sum::<usize>();
        *program = work;
        Ok(OptimizationStats {
            input_words,
            output_words,
            passes_changed,
            words_removed: input_words.saturating_sub(output_words),
        })
    }
}

/// Conservative default: all transforms preserve the execution semantics of
/// existing opcodes; no VM dispatch-loop update is required.
pub fn optimization_passes() -> Optimize {
    Optimize::new()
        .with_pass(RemoveIdentityMoves)
        .with_pass(NormalizeIntegerLoads)
        .with_pass(RemoveOverwrittenLoads)
        .with_pass(CollapseDoubleSwaps)
        .with_pass(CollapseDuplicateMoves)
        .with_pass(FoldIntegerSelfOperands)
        .with_pass(PropagateAdjacentCopies)
        .with_pass(FoldIntegerIdentities)
        .with_pass(FoldAdjacentConstants)
        .with_pass(FoldMetadataConstants)
        .with_pass(FoldRepeatedGuards)
        .with_pass(FoldLiteralBranches)
        .with_pass(ThreadUnconditionalJumps)
        .with_pass(InvertBranchOverJump)
        .with_pass(RemoveFallthroughJumps)
        .with_pass(CompactBytecode)
}

/// A minimal pipeline for debugging or differential testing.
pub fn conservative_optimization_passes() -> Optimize {
    Optimize::new()
        .with_pass(RemoveIdentityMoves)
        .with_pass(NormalizeIntegerLoads)
        .with_pass(RemoveFallthroughJumps)
        .with_pass(CompactBytecode)
}

/// The optimizer never edits a bytecode word in the middle of an instruction.
#[derive(Clone, Copy, Debug)]
struct InstView {
    pc: usize,
    len: usize,
    op: Opcode,
    first: Word,
}

fn decode(code: &[Word], pass: &'static str) -> Result<Vec<InstView>, OptimizationError> {
    Decoder::new(code)
        .map(|result| {
            let i = result.map_err(|e| error(pass, e.to_string()))?;
            Ok(InstView { pc: i.pc.word_index(), len: i.len_words(), op: i.opcode, first: i.first })
        })
        .collect()
}

fn replace(code: &mut [Word], pc: usize, op: Opcode, a: u8, b: u8, c: u8) {
    code[pc] = Word::abc(op, a, b, c);
}

fn nop(code: &mut [Word], pc: usize) {
    code[pc] = Word::abc(Opcode::Nop, 0, 0, 0);
}

/// Resume PCs that inlined deopt frames (owned by *other* functions) target in
/// each function. Computed once per pass: rescanning every function's deopts
/// for each function made entry discovery O(functions²) per pass. Non-compacting
/// passes preserve instruction boundaries, so the index stays valid while a pass
/// edits code.
type InlinedResumes = HashMap<FuncId, SmallVec<[usize; 4]>>;

fn inlined_resumes(program: &BytecodeProgram) -> InlinedResumes {
    let mut result = InlinedResumes::default();
    for f in &program.functions {
        for d in &f.chunk.deopts {
            for frame in &d.inlined_frames {
                result.entry(frame.function).or_default().push(frame.resume_pc.word_index());
            }
        }
    }
    result
}

/// Dense set of instruction-boundary PCs (`0..=code.len()`).
struct Boundaries(Vec<bool>);

impl Boundaries {
    #[inline]
    fn insert(&mut self, pc: usize) {
        if pc >= self.0.len() { self.0.resize(pc + 1, false); }
        self.0[pc] = true;
    }

    #[inline]
    fn contains(&self, pc: &usize) -> bool { self.0.get(*pc).copied().unwrap_or(false) }
}

/// Mark every possible non-fallthrough entry. A local rewrite must not rely on
/// the preceding instruction if the current PC can be entered directly.
fn entries(program: &BytecodeProgram, function_index: usize, inlined: &InlinedResumes) -> Result<Boundaries, OptimizationError> {
    let chunk = &program.functions[function_index].chunk;
    let mut result = Boundaries(vec![false; chunk.code.len() + 1]);
    result.insert(0);
    for i in decode(&chunk.code, "CFG entry discovery")? {
        if matches!(i.op, Opcode::Jump | Opcode::JumpIfFalse | Opcode::JumpIfTrue) {
            let delta = chunk.code[i.pc + 1].0 as i32 as i64;
            let target = i.pc as i64 + i.len as i64 + delta;
            if target < 0 || target > chunk.code.len() as i64 {
                return Err(error("CFG entry discovery", "invalid relative target"));
            }
            result.insert(target as usize);
        }
    }
    for table in &chunk.switch_tables {
        result.insert(table.default_target.word_index());
        for c in &table.cases { result.insert(c.target.word_index()); }
    }
    for h in &chunk.exception_handlers {
        result.insert(h.handler_pc.word_index());
        result.insert(h.start_pc.word_index());
        result.insert(h.end_pc.word_index());
    }
    for e in &chunk.resume_points { result.insert(e.pc.word_index()); }
    for d in &chunk.deopts { result.insert(d.resume_pc.word_index()); }
    if let Some(pcs) = inlined.get(&program.functions[function_index].id) {
        for &pc in pcs { result.insert(pc); }
    }
    Ok(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Literal {
    Null,
    Bool(bool),
    Int(i32),
}

fn literal_at(chunk: &crate::napalm_bytecode::BytecodeChunk, inst: InstView) -> Option<Literal> {
    match inst.op {
        Opcode::LoadNull => Some(Literal::Null),
        Opcode::LoadFalse => Some(Literal::Bool(false)),
        Opcode::LoadTrue => Some(Literal::Bool(true)),
        Opcode::LoadI8 => Some(Literal::Int(inst.first.b() as i8 as i32)),
        Opcode::LoadI32 => Some(Literal::Int(chunk.code[inst.pc + 1].0 as i32)),
        Opcode::LoadConst => match chunk.constants.get(chunk.code[inst.pc + 1].0 as usize)? {
            Constant::Null => Some(Literal::Null),
            Constant::Bool(v) => Some(Literal::Bool(*v)),
            Constant::Int32(v) => Some(Literal::Int(*v)),
            _ => None,
        },
        _ => None,
    }
}

fn encoded_i8(op: Opcode, dst: u8, value: i32) -> Word {
    Word::abc(op, dst, value as i8 as u8, 0)
}

pub struct RemoveIdentityMoves;
impl BytecodeOptimization for RemoveIdentityMoves {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        for f in &mut program.functions {
            for i in decode(&f.chunk.code, self.name())? {
                if matches!(i.op, Opcode::Move | Opcode::Swap) && i.first.a() == i.first.b() {
                    nop(&mut f.chunk.code, i.pc);
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
}

/// Replace wide literal instructions by their short form and leave an inert
/// extension word for the final compactor. This preserves all PCs until the
/// global relocation phase.
pub struct NormalizeIntegerLoads;
impl BytecodeOptimization for NormalizeIntegerLoads {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        for f in &mut program.functions {
            for i in decode(&f.chunk.code, self.name())? {
                let Some(lit) = literal_at(&f.chunk, i) else { continue; };
                let (op, b) = match lit {
                    Literal::Null => (Opcode::LoadNull, 0),
                    Literal::Bool(false) => (Opcode::LoadFalse, 0),
                    Literal::Bool(true) => (Opcode::LoadTrue, 0),
                    Literal::Int(v) if i8::try_from(v).is_ok() => (Opcode::LoadI8, v as i8 as u8),
                    _ => continue,
                };
                if i.op == op && (i.op != Opcode::LoadI8 || i.first.b() == b) { continue; }
                replace(&mut f.chunk.code, i.pc, op, i.first.a(), b, 0);
                for pc in i.pc + 1..i.pc + i.len { nop(&mut f.chunk.code, pc); }
                changed = true;
            }
        }
        Ok(changed)
    }
}

/// A pure load immediately overwritten in the same physical register has no
/// effects and can be eliminated without any liveness inference.
pub struct RemoveOverwrittenLoads;
impl BytecodeOptimization for RemoveOverwrittenLoads {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi, &inlined)?;
            let insts = decode(&program.functions[fi].chunk.code, self.name())?;
            let mut edits = Vec::new();
            for pair in insts.windows(2) {
                let (a,b) = (pair[0], pair[1]);
                if !matches!(a.op, Opcode::Move | Opcode::LoadNull | Opcode::LoadFalse | Opcode::LoadTrue | Opcode::LoadI8 | Opcode::LoadI32 | Opcode::LoadConst)
                    || !matches!(b.op, Opcode::LoadNull | Opcode::LoadFalse | Opcode::LoadTrue | Opcode::LoadI8 | Opcode::LoadI32 | Opcode::LoadConst)
                    || a.first.a() != b.first.a() || boundaries.contains(&b.pc)
                { continue; }
                edits.push((a.pc, a.len));
            }
            if !edits.is_empty() {
                let code = &mut program.functions[fi].chunk.code;
                for (pc,len) in edits {
                    for offset in 0..len { nop(code, pc + offset); }
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
}

/// Two adjacent swaps of the same registers cancel each other. A second
/// instruction with an incoming branch/OSR/deopt edge is never eliminated.
pub struct CollapseDoubleSwaps;
impl BytecodeOptimization for CollapseDoubleSwaps {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries=entries(program, fi, &inlined)?;
            let insts=decode(&program.functions[fi].chunk.code,self.name())?;
            let mut edits=Vec::new();
            let mut used=HashSet::default();
            for pair in insts.windows(2) {
                let (a,b)=(pair[0],pair[1]);
                if a.op!=Opcode::Swap || b.op!=Opcode::Swap
                    || boundaries.contains(&b.pc) || used.contains(&a.pc) {continue;}
                if (a.first.a()==b.first.a() && a.first.b()==b.first.b())
                    || (a.first.a()==b.first.b() && a.first.b()==b.first.a()) {
                    edits.push((a.pc,b.pc));
                    used.insert(a.pc);used.insert(b.pc);
                }
            }
            let code=&mut program.functions[fi].chunk.code;
            for (a,b) in edits {nop(code,a);nop(code,b);changed=true;}
        }
        Ok(changed)
    }
}

pub struct CollapseDuplicateMoves;
impl BytecodeOptimization for CollapseDuplicateMoves {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries=entries(program, fi, &inlined)?;
            let insts=decode(&program.functions[fi].chunk.code,self.name())?;
            let removals=insts.windows(2).filter_map(|pair| {
                let (a,b)=(pair[0],pair[1]);
                if a.op==Opcode::Move && b.op==Opcode::Move
                    && a.first==b.first && !boundaries.contains(&b.pc) {
                    Some(b.pc)
                } else {None}
            }).collect::<Vec<_>>();
            let code=&mut program.functions[fi].chunk.code;
            for pc in removals {nop(code,pc);changed=true;}
        }
        Ok(changed)
    }
}

/// Identities that are valid for any signed i32 value and do not require
/// speculative type information or new register uses.
pub struct FoldIntegerSelfOperands;
impl BytecodeOptimization for FoldIntegerSelfOperands {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        for f in &mut program.functions {
            for i in decode(&f.chunk.code,self.name())? {
                if i.first.b()!=i.first.c() {continue;}
                let word=match i.op {
                    Opcode::ISub|Opcode::IBitXor => Some(encoded_i8(Opcode::LoadI8,i.first.a(),0)),
                    Opcode::ICmpEq|Opcode::ICmpLe|Opcode::ICmpGe =>
                        Some(Word::abc(Opcode::LoadTrue,i.first.a(),0,0)),
                    Opcode::ICmpNe|Opcode::ICmpLt|Opcode::ICmpGt =>
                        Some(Word::abc(Opcode::LoadFalse,i.first.a(),0,0)),
                    Opcode::IBitAnd|Opcode::IBitOr =>
                        Some(Word::abc(Opcode::Move,i.first.a(),i.first.b(),0)),
                    _=>None,
                };
                if let Some(word)=word {
                    if f.chunk.code[i.pc]!=word {f.chunk.code[i.pc]=word;changed=true;}
                }
            }
        }
        Ok(changed)
    }
}

/// Adjacent copies of a stable literal can be rewritten to a single-word
/// materialization without adding new register uses to the JIT live maps.
pub struct PropagateAdjacentCopies;
impl BytecodeOptimization for PropagateAdjacentCopies {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi, &inlined)?;
            let insts = decode(&program.functions[fi].chunk.code, self.name())?;
            let mut edits = Vec::new();
            for pair in insts.windows(2) {
                let (a,b) = (pair[0], pair[1]);
                if b.op != Opcode::Move || b.first.b() != a.first.a() || boundaries.contains(&b.pc) { continue; }
                let replacement = match a.op {
                    Opcode::LoadI8 | Opcode::LoadNull | Opcode::LoadFalse | Opcode::LoadTrue =>
                        Some(Word::abc(a.op, b.first.a(), a.first.b(), 0)),
                    _ => None,
                };
                if let Some(word) = replacement { edits.push((b.pc, word)); }
            }
            let code = &mut program.functions[fi].chunk.code;
            for (pc, word) in edits {
                if code[pc] != word { code[pc] = word; changed = true; }
            }
        }
        Ok(changed)
    }
}

pub struct FoldIntegerIdentities;
impl BytecodeOptimization for FoldIntegerIdentities {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi, &inlined)?;
            let insts = decode(&program.functions[fi].chunk.code, self.name())?;
            let mut edits = Vec::new();
            for pair in insts.windows(2) {
                let (previous, op) = (pair[0], pair[1]);
                if boundaries.contains(&op.pc) { continue; }
                let Some(Literal::Int(k)) = literal_at(&program.functions[fi].chunk, previous) else { continue; };
                let temp = previous.first.a();
                let l = op.first.b();
                let r = op.first.c();
                let other = if l == temp { Some(r) } else if r == temp { Some(l) } else { None };
                let Some(other) = other else { continue; };
                let result = match op.op {
                    Opcode::IAdd if k == 0 => Some(Word::abc(Opcode::Move, op.first.a(), other, 0)),
                    Opcode::ISub if k == 0 && r == temp => Some(Word::abc(Opcode::Move, op.first.a(), l, 0)),
                    Opcode::IMul if k == 1 => Some(Word::abc(Opcode::Move, op.first.a(), other, 0)),
                    Opcode::IMul if k == 0 => Some(encoded_i8(Opcode::LoadI8, op.first.a(), 0)),
                    Opcode::IBitOr | Opcode::IBitXor if k == 0 => Some(Word::abc(Opcode::Move, op.first.a(), other, 0)),
                    Opcode::IBitAnd if k == 0 => Some(encoded_i8(Opcode::LoadI8, op.first.a(), 0)),
                    Opcode::IShl | Opcode::IShrU | Opcode::ISar if k == 0 && r == temp =>
                        Some(Word::abc(Opcode::Move, op.first.a(), l, 0)),
                    _ => None,
                };
                if let Some(word) = result { edits.push((op.pc, word)); }
            }
            let code = &mut program.functions[fi].chunk.code;
            for (pc, word) in edits { if code[pc] != word { code[pc] = word; changed = true; } }
        }
        Ok(changed)
    }
}

fn fold_i32(op: Opcode, a: i32, b: i32) -> Option<Literal> {
    use Opcode::*;
    let value = match op {
        IAdd => Literal::Int(a.checked_add(b)?),
        ISub => Literal::Int(a.checked_sub(b)?),
        IMul => Literal::Int(a.checked_mul(b)?),
        IDivS => Literal::Int(a.checked_div(b)?),
        IRemS => Literal::Int(a.checked_rem(b)?),
        IBitAnd => Literal::Int(a & b),
        IBitOr => Literal::Int(a | b),
        IBitXor => Literal::Int(a ^ b),
        IShl if (0..32).contains(&b) => Literal::Int(a.wrapping_shl(b as u32)),
        IShrU if (0..32).contains(&b) => Literal::Int(((a as u32) >> b) as i32),
        ISar if (0..32).contains(&b) => Literal::Int(a >> b),
        ICmpEq => Literal::Bool(a == b),
        ICmpNe => Literal::Bool(a != b),
        ICmpLt => Literal::Bool(a < b),
        ICmpLe => Literal::Bool(a <= b),
        ICmpGt => Literal::Bool(a > b),
        ICmpGe => Literal::Bool(a >= b),
        _ => return None,
    };
    Some(value)
}

/// One-sided identities using guaranteed register constants. The surviving
/// operand is always one of the original instruction's inputs, so the
/// baseline JIT liveness map remains conservative and valid.
fn fold_one_known_i32(op: Opcode, dst: u8, left: u8, right: u8,
    left_value: Option<i32>, right_value: Option<i32>) -> Option<Word> {
    let mv = |reg| Word::abc(Opcode::Move,dst,reg,0);
    let li = |value| encoded_i8(Opcode::LoadI8,dst,value);
    match op {
        Opcode::IAdd|Opcode::IBitOr|Opcode::IBitXor if left_value==Some(0) => Some(mv(right)),
        Opcode::IAdd|Opcode::IBitOr|Opcode::IBitXor if right_value==Some(0) => Some(mv(left)),
        Opcode::ISub if right_value==Some(0) => Some(mv(left)),
        Opcode::IMul if left_value==Some(1) => Some(mv(right)),
        Opcode::IMul if right_value==Some(1) => Some(mv(left)),
        Opcode::IMul|Opcode::IBitAnd if left_value==Some(0)||right_value==Some(0) => Some(li(0)),
        Opcode::IBitAnd if left_value==Some(-1) => Some(mv(right)),
        Opcode::IBitAnd if right_value==Some(-1) => Some(mv(left)),
        Opcode::IBitOr if left_value==Some(-1)||right_value==Some(-1) => Some(li(-1)),
        Opcode::IShl|Opcode::IShrU|Opcode::ISar if right_value==Some(0) => Some(mv(left)),
        Opcode::IDivS if right_value==Some(1) => Some(mv(left)),
        Opcode::IRemS if right_value==Some(1) => Some(li(0)),
        _ => None,
    }
}

/// Folds two consecutive constant definitions followed by an integer opcode.
/// Only representable small results are folded, avoiding a code-size increase.
/// The original definitions remain available to GC/deopt snapshots until the
/// later dead-load and compaction passes.
pub struct FoldAdjacentConstants;
impl BytecodeOptimization for FoldAdjacentConstants {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi, &inlined)?;
            let code = &program.functions[fi].chunk.code;
            let insts = decode(code, self.name())?;
            let mut edits = Vec::new();
            for triple in insts.windows(3) {
                let (a,b,op) = (triple[0],triple[1],triple[2]);
                if boundaries.contains(&b.pc) || boundaries.contains(&op.pc) { continue; }
                let (Some(Literal::Int(ak)), Some(Literal::Int(bk))) =
                    (literal_at(&program.functions[fi].chunk, a), literal_at(&program.functions[fi].chunk, b)) else { continue; };
                if a.first.a() == b.first.a() { continue; }
                let Some(result) = (if op.first.b() == a.first.a() && op.first.c() == b.first.a() {
                    fold_i32(op.op, ak, bk)
                } else if op.first.b() == b.first.a() && op.first.c() == a.first.a() {
                    fold_i32(op.op, bk, ak)
                } else { None }) else { continue; };
                let word = match result {
                    Literal::Int(v) if i8::try_from(v).is_ok() => encoded_i8(Opcode::LoadI8, op.first.a(), v),
                    Literal::Bool(v) => Word::abc(if v {Opcode::LoadTrue} else {Opcode::LoadFalse}, op.first.a(), 0, 0),
                    _ => continue,
                };
                edits.push((op.pc,word));
            }
            let code = &mut program.functions[fi].chunk.code;
            for (pc,word) in edits { if code[pc] != word { code[pc]=word; changed=true; } }
        }
        Ok(changed)
    }
}

/// Exploit the compiler's *proven* live SSA constant facts. Unlike a local
/// peephole, this works across basic blocks and after register allocation.
pub struct FoldMetadataConstants;
impl BytecodeOptimization for FoldMetadataConstants {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        for f in &mut program.functions {
            let insts = decode(&f.chunk.code, self.name())?;
            for i in insts {
                // Finish all immutable borrows of the side table before
                // changing the executable words of this chunk.
                let (replacement, branch_taken) = {
                    let facts = f.chunk.constants_at(BytecodePc(i.pc as u32));
                    let int = |reg: u8| facts.iter().find_map(|(r,c)| {
                        if r.0 != reg { return None; }
                        match c { Constant::Int32(value) => Some(*value), _ => None }
                    });
                    let bool_value = |reg: u8| facts.iter().find_map(|(r,c)| {
                        if r.0 != reg { return None; }
                        match c { Constant::Bool(value) => Some(*value), _ => None }
                    });
                    let left=int(i.first.b());
                    let right=int(i.first.c());
                    let replacement = if let (Some(l),Some(r)) = (left,right) {
                        match fold_i32(i.op,l,r) {
                            Some(Literal::Int(v)) if i8::try_from(v).is_ok() =>
                                Some(encoded_i8(Opcode::LoadI8,i.first.a(),v)),
                            Some(Literal::Bool(v)) =>
                                Some(Word::abc(if v {Opcode::LoadTrue} else {Opcode::LoadFalse},i.first.a(),0,0)),
                            _ => fold_one_known_i32(i.op,i.first.a(),i.first.b(),i.first.c(),left,right),
                        }
                    } else {
                        fold_one_known_i32(i.op,i.first.a(),i.first.b(),i.first.c(),left,right)
                    };
                    let branch_taken = if matches!(i.op,Opcode::JumpIfTrue|Opcode::JumpIfFalse) {
                        bool_value(i.first.a()).map(|v|if i.op == Opcode::JumpIfTrue {v} else {!v})
                    } else { None };
                    (replacement,branch_taken)
                };
                if let Some(word) = replacement {
                    if word != f.chunk.code[i.pc] { f.chunk.code[i.pc] = word; changed = true; }
                } else if let Some(taken) = branch_taken {
                    if taken { replace(&mut f.chunk.code,i.pc,Opcode::Jump,0,0,0); }
                    else { nop(&mut f.chunk.code,i.pc); nop(&mut f.chunk.code,i.pc+1); }
                    changed = true;
                }
            }
        }
        Ok(changed)
    }
}

/// A guard with precisely the same operands and deopt ID, executed directly
/// after an identical guard, cannot fail if the first already succeeded.
pub struct FoldRepeatedGuards;
impl BytecodeOptimization for FoldRepeatedGuards {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries=entries(program, fi, &inlined)?;
            let chunk=&program.functions[fi].chunk;
            let insts=decode(&chunk.code,self.name())?;
            let mut removals=Vec::new();
            for pair in insts.windows(2) {
                let (left,right)=(pair[0],pair[1]);
                if boundaries.contains(&right.pc)
                    || !matches!(right.op,Opcode::GuardType|Opcode::GuardClass|Opcode::GuardBounds|
                                     Opcode::GuardNotNull|Opcode::GuardInt32|Opcode::GuardF64)
                    || left.op!=right.op || left.first!=right.first || left.len!=right.len {continue;}
                if chunk.code[left.pc..left.pc+left.len] == chunk.code[right.pc..right.pc+right.len] {
                    removals.push((right.pc,right.len));
                }
            }
            let code=&mut program.functions[fi].chunk.code;
            for (pc,len) in removals {
                for offset in 0..len {nop(code,pc+offset);}
                changed=true;
            }
        }
        Ok(changed)
    }
}

pub struct FoldLiteralBranches;
impl BytecodeOptimization for FoldLiteralBranches {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi, &inlined)?;
            let chunk = &program.functions[fi].chunk;
            let insts = decode(&chunk.code, self.name())?;
            let mut edits = Vec::new();
            for pair in insts.windows(2) {
                let (load,branch)=(pair[0],pair[1]);
                if boundaries.contains(&branch.pc) || load.first.a() != branch.first.a() { continue; }
                let Some(Literal::Bool(value))=literal_at(chunk, load) else {continue;};
                let taken = match branch.op {
                    Opcode::JumpIfTrue => value,
                    Opcode::JumpIfFalse => !value,
                    _ => continue,
                };
                edits.push((branch.pc,taken));
            }
            let code=&mut program.functions[fi].chunk.code;
            for (pc,taken) in edits {
                if taken { replace(code,pc,Opcode::Jump,0,0,0); }
                else { nop(code,pc); nop(code,pc+1); }
                changed=true;
            }
        }
        Ok(changed)
    }
}

/// Redirect branches through an unconditional jump, without deleting the jump
/// (which may also be reachable from another control-flow predecessor).
pub struct ThreadUnconditionalJumps;
impl BytecodeOptimization for ThreadUnconditionalJumps {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        for f in &mut program.functions {
            let insts=decode(&f.chunk.code,self.name())?;
            // `insts` is sorted by PC: binary-search it instead of building a
            // per-function HashMap; jump chains are short, so `visited` is too.
            let len=f.chunk.code.len() as i64;
            let mut edits=Vec::new();
            for i in &insts {
                if !matches!(i.op,Opcode::Jump|Opcode::JumpIfFalse|Opcode::JumpIfTrue) {continue;}
                let old_delta=f.chunk.code[i.pc+1].0 as i32 as i64;
                let mut target=i.pc as i64+i.len as i64+old_delta;
                let mut visited=SmallVec::<[i64; 4]>::new();
                while target>=0 && target<len && !visited.contains(&target) {
                    visited.push(target);
                    let Ok(found)=insts.binary_search_by_key(&(target as usize),|i|i.pc) else {break;};
                    let next=&insts[found];
                    if next.op!=Opcode::Jump {break;}
                    target=next.pc as i64+next.len as i64+f.chunk.code[next.pc+1].0 as i32 as i64;
                }
                if target>=0 && target<=len && target!=i.pc as i64+i.len as i64+old_delta {
                    let new_delta=target-(i.pc+i.len) as i64;
                    if let Ok(delta)=i32::try_from(new_delta) {edits.push((i.pc+1,delta as u32));}
                }
            }
            for (pc,delta) in edits {
                if f.chunk.code[pc].0!=delta {f.chunk.code[pc]=Word(delta); changed=true;}
            }
        }
        Ok(changed)
    }
}

/// Pattern:
///     jump-if-true  r,  AFTER_JUMP
///     jump                ELSEWHERE
/// AFTER_JUMP:
/// =>  jump-if-false r, ELSEWHERE
/// The second jump is converted to NOPs; compaction removes them.
pub struct InvertBranchOverJump;
impl BytecodeOptimization for InvertBranchOverJump {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let inlined = inlined_resumes(program);
        for fi in 0..program.functions.len() {
            let boundaries=entries(program, fi, &inlined)?;
            let chunk=&program.functions[fi].chunk;
            let insts=decode(&chunk.code,self.name())?;
            let mut edits=Vec::new();
            for pair in insts.windows(2) {
                let (branch,jump)=(pair[0],pair[1]);
                if !matches!(branch.op,Opcode::JumpIfFalse|Opcode::JumpIfTrue)
                    || jump.op!=Opcode::Jump || boundaries.contains(&jump.pc) {continue;}
                let taken=branch.pc as i64+branch.len as i64+chunk.code[branch.pc+1].0 as i32 as i64;
                if taken!=jump.pc as i64+jump.len as i64 {continue;}
                let other=jump.pc as i64+jump.len as i64+chunk.code[jump.pc+1].0 as i32 as i64;
                let delta=other-(branch.pc+branch.len) as i64;
                let Ok(delta)=i32::try_from(delta) else {continue;};
                edits.push((branch.pc, jump.pc, delta));
            }
            let code=&mut program.functions[fi].chunk.code;
            for (branch,jump,delta) in edits {
                let op=if code[branch].opcode()==Some(Opcode::JumpIfTrue) {Opcode::JumpIfFalse} else {Opcode::JumpIfTrue};
                let condition=code[branch].a();
                replace(code,branch,op,condition,0,0);
                code[branch+1]=Word(delta as u32);
                nop(code,jump);
                nop(code,jump+1);
                changed=true;
            }
        }
        Ok(changed)
    }
}

pub struct RemoveFallthroughJumps;
impl BytecodeOptimization for RemoveFallthroughJumps {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        for f in &mut program.functions {
            for i in decode(&f.chunk.code,self.name())? {
                if i.op==Opcode::Jump && f.chunk.code[i.pc+1].0==0 {
                    nop(&mut f.chunk.code,i.pc);
                    nop(&mut f.chunk.code,i.pc+1);
                    changed=true;
                }
            }
        }
        Ok(changed)
    }
}

#[derive(Clone, Debug)]
struct Relocation {
    /// Dense old-PC -> new-PC table; `NOT_A_BOUNDARY` marks interior words.
    old_to_new: Vec<u32>,
}

const NOT_A_BOUNDARY: u32 = u32::MAX;

impl Relocation {
    fn map(&self, pc: BytecodePc) -> Result<BytecodePc, OptimizationError> {
        self.old_to_new.get(pc.word_index()).copied()
            .filter(|&new| new != NOT_A_BOUNDARY).map(BytecodePc)
            .ok_or_else(||error("CompactBytecode",format!("PC {} is not an instruction boundary",pc.0)))
    }
}

pub struct CompactBytecode;
impl BytecodeOptimization for CompactBytecode {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let mut relocations=HashMap::<FuncId,Relocation>::default();
        for f in &mut program.functions {
            let old_code=&f.chunk.code;
            let insts=decode(old_code,self.name())?;
            let mut mapping=vec![NOT_A_BOUNDARY; old_code.len()+1];
            let mut new_code=Vec::with_capacity(old_code.len());
            // Store old instruction start and new start for later branch patching.
            let mut branches=Vec::new();
            for i in &insts {
                mapping[i.pc]=new_code.len() as u32;
                if i.op==Opcode::Nop {continue;}
                if matches!(i.op,Opcode::Jump|Opcode::JumpIfTrue|Opcode::JumpIfFalse) {
                    let target=(i.pc+i.len) as i64+old_code[i.pc+1].0 as i32 as i64;
                    branches.push((new_code.len(),target));
                }
                new_code.extend_from_slice(&old_code[i.pc..i.pc+i.len]);
            }
            mapping[old_code.len()]=new_code.len() as u32;
            for (new_pc,target) in branches {
                let Ok(old_target)=usize::try_from(target) else {
                    return Err(error(self.name(),"negative branch target"));
                };
                let Some(new_target)=mapping.get(old_target).copied().filter(|&new| new != NOT_A_BOUNDARY) else {
                    return Err(error(self.name(),format!("branch target {old_target} is not a boundary")));
                };
                let delta=new_target as i64-(new_pc+2) as i64;
                let delta=i32::try_from(delta).map_err(|_|error(self.name(),"branch displacement overflow"))?;
                new_code[new_pc+1]=Word(delta as u32);
            }
            let relocation=Relocation {old_to_new:mapping};
            for table in &mut f.chunk.switch_tables {
                table.default_target=relocation.map(table.default_target)?;
                for c in &mut table.cases {c.target=relocation.map(c.target)?;}
            }
            // A removed instruction may have had a *different* pre-execution
            // value state from its successor (especially when dead writes or
            // two cancelling swaps have been eliminated). Never reassign its
            // old GC/liveness facts to a new PC: preserve only the snapshots
            // originally attached to surviving instructions.
            let mut old_nops=Boundaries(vec![false; old_code.len()+1]);
            for i in insts.iter().filter(|i|i.op==Opcode::Nop) { old_nops.insert(i.pc); }
            f.chunk.safepoints.retain(|e|!old_nops.contains(&e.pc.word_index()));
            for e in &mut f.chunk.safepoints {e.pc=relocation.map(e.pc)?;}
            f.chunk.safepoints.sort_by_key(|e|e.pc.0);
            f.chunk.live_registers.retain(|e|!old_nops.contains(&e.pc.word_index()));
            for e in &mut f.chunk.live_registers {e.pc=relocation.map(e.pc)?;}
            f.chunk.live_registers.sort_by_key(|e|e.pc.0);
            for e in &mut f.chunk.resume_points {e.pc=relocation.map(e.pc)?;}
            f.chunk.resume_points.sort_by_key(|e|e.pc.0);
            for d in &mut f.chunk.deopts {d.resume_pc=relocation.map(d.resume_pc)?;}
            for h in &mut f.chunk.exception_handlers {
                h.start_pc=relocation.map(h.start_pc)?;
                h.end_pc=relocation.map(h.end_pc)?;
                h.handler_pc=relocation.map(h.handler_pc)?;
            }
            f.chunk.exception_handlers.retain(|h|h.start_pc.0<h.end_pc.0);
            for s in &mut f.chunk.source_map {
                s.start_pc=relocation.map(s.start_pc)?;
                s.end_pc=relocation.map(s.end_pc)?;
            }
            f.chunk.source_map.retain(|s|s.start_pc.0<s.end_pc.0);
            changed|=new_code.len()!=old_code.len();
            f.chunk.code=new_code;
            relocations.insert(f.id,relocation);
        }
        // Inlined frame snapshots refer to other functions' PCs. They must be
        // fixed only *after* all functions' relocation maps are available.
        for f in &mut program.functions {
            for d in &mut f.chunk.deopts {
                for frame in &mut d.inlined_frames {
                    let r=relocations.get(&frame.function).ok_or_else(||
                        error(self.name(),format!("unknown inline frame function {:?}",frame.function)))?;
                    frame.resume_pc=r.map(frame.resume_pc)?;
                }
            }
        }
        Ok(changed)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::napalm_bytecode::Assembler;

    fn stub_function(code: Vec<Word>) -> BytecodeProgram {
        use crate::napalm_bytecode::{BytecodeFunction, BytecodeChunk, FrameLayout, RegisterKind};
        use crate::napalm_ir::{ModuleId, StringId, FunctionKind, MemoizationKind, Type};
        let mut program = BytecodeProgram::new();
        let mut chunk = BytecodeChunk::default();
        chunk.code = code;
        program.push_function(BytecodeFunction {
            id: FuncId(0), module: ModuleId(0), kind: FunctionKind::ModuleBody,
            is_async: false, is_concurrent: false, memoization: MemoizationKind::None,
            field_owner: None, name: StringId(0), arity: 0, result_type: Type::Top,
            capture_count: 0,
            frame: FrameLayout {
                general_register_count: 8, parameters: Vec::new(), captures: Vec::new(),
                return_kind: RegisterKind::Value,
            },
            chunk,
        }).unwrap();
        program
    }

    #[test]
    fn small_constant_folding_only_uses_representable_results() {
        assert_eq!(fold_i32(Opcode::IAdd, 1, 2),Some(Literal::Int(3)));
        assert_eq!(fold_i32(Opcode::ICmpLt, 1, 2),Some(Literal::Bool(true)));
        assert_eq!(fold_i32(Opcode::IDivS, 3, 0),None);
        assert_eq!(fold_i32(Opcode::IAdd, i32::MAX, 1),None);
        assert_eq!(fold_i32(Opcode::IShrU, -1, 1),Some(Literal::Int(i32::MAX)));
    }

    #[test]
    fn decoded_extension_words_are_never_treated_as_instructions() {
        let mut a=Assembler::new();
        a.emit_x32(Opcode::LoadI32,0,0,0,0x0000_00ff);
        a.emit_a(Opcode::Return,0);
        let code=a.finish().unwrap().code;
        let insts=decode(&code,"test").unwrap();
        assert_eq!(insts.len(),2);
        assert_eq!(insts[0].len,2);
    }
}

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
//! Superinstructions requiring new interpreter handlers are intentionally not
//! emitted here: the baseline interpreter must support an opcode before it is
//! safe to make it executable. `superinstruction_candidates` supplies a
//! descriptor-driven catalog for the next runtime integration phase.

use std::collections::{HashMap, HashSet};
use std::fmt;
use smallbox::{SmallBox, smallbox, space::S1};

use crate::napalm_bytecode::{
    BytecodePc, BytecodeProgram, BytecodeVerifier, Decoder, Opcode, Word,
    BytecodeFusion, SpecializedDescriptor, FusionContinuation, Reg, pack_registers,
    decode_call_conv,
};
use crate::napalm_ir::{Constant, FuncId, ClassId, StructuralLayoutId, PropertySlotId, DeoptId, StringId};

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

/// Mark every possible non-fallthrough entry. A local rewrite must not rely on
/// the preceding instruction if the current PC can be entered directly.
fn entries(program: &BytecodeProgram, function_index: usize) -> Result<HashSet<usize>, OptimizationError> {
    let chunk = &program.functions[function_index].chunk;
    let mut result = HashSet::from([0usize]);
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
    let id = program.functions[function_index].id;
    for f in &program.functions {
        for d in &f.chunk.deopts {
            for frame in &d.inlined_frames {
                if frame.function == id { result.insert(frame.resume_pc.word_index()); }
            }
        }
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
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries=entries(program,fi)?;
            let insts=decode(&program.functions[fi].chunk.code,self.name())?;
            let mut edits=Vec::new();
            let mut used=HashSet::new();
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
        for fi in 0..program.functions.len() {
            let boundaries=entries(program,fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries=entries(program,fi)?;
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
        for fi in 0..program.functions.len() {
            let boundaries = entries(program, fi)?;
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
            let by_pc=insts.iter().map(|i|(i.pc,*i)).collect::<HashMap<_,_>>();
            let len=f.chunk.code.len() as i64;
            let mut edits=Vec::new();
            for i in &insts {
                if !matches!(i.op,Opcode::Jump|Opcode::JumpIfFalse|Opcode::JumpIfTrue) {continue;}
                let old_delta=f.chunk.code[i.pc+1].0 as i32 as i64;
                let mut target=i.pc as i64+i.len as i64+old_delta;
                let mut visited=HashSet::new();
                while target>=0 && target<len && visited.insert(target) {
                    let Some(next)=by_pc.get(&(target as usize)) else {break;};
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
        for fi in 0..program.functions.len() {
            let boundaries=entries(program,fi)?;
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
    old_to_new: HashMap<usize, usize>,
}

impl Relocation {
    fn map(&self, pc: BytecodePc) -> Result<BytecodePc, OptimizationError> {
        self.old_to_new.get(&pc.word_index()).copied()
            .and_then(|new| u32::try_from(new).ok()).map(BytecodePc)
            .ok_or_else(||error("CompactBytecode",format!("PC {} is not an instruction boundary",pc.0)))
    }
}

pub struct CompactBytecode;
impl BytecodeOptimization for CompactBytecode {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed=false;
        let mut relocations=HashMap::<FuncId,Relocation>::new();
        for f in &mut program.functions {
            let old_code=&f.chunk.code;
            let insts=decode(old_code,self.name())?;
            let mut mapping=HashMap::<usize,usize>::new();
            let mut new_code=Vec::with_capacity(old_code.len());
            // Store old instruction start and new start for later branch patching.
            let mut branches=Vec::new();
            for i in &insts {
                mapping.insert(i.pc,new_code.len());
                if i.op==Opcode::Nop {continue;}
                if matches!(i.op,Opcode::Jump|Opcode::JumpIfTrue|Opcode::JumpIfFalse) {
                    let target=(i.pc+i.len) as i64+old_code[i.pc+1].0 as i32 as i64;
                    branches.push((new_code.len(),target));
                }
                new_code.extend_from_slice(&old_code[i.pc..i.pc+i.len]);
            }
            mapping.insert(old_code.len(),new_code.len());
            for (new_pc,target) in branches {
                let Ok(old_target)=usize::try_from(target) else {
                    return Err(error(self.name(),"negative branch target"));
                };
                let Some(&new_target)=mapping.get(&old_target) else {
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
            let old_nops=insts.iter().filter(|i|i.op==Opcode::Nop)
                .map(|i|i.pc).collect::<HashSet<_>>();
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


/// Fused bytecode emission pass. It works on completed physical bytecode, so
/// operands are already allocated, profiles have been lowered, and the original
/// PC-indexed GC/deopt information is available for validation.
///
/// This pass maintains the original code length with `Nop` padding. The later
/// `CompactBytecode` pass performs a single relocation of every PC-indexed
/// table, including references from other functions' inlined frame snapshots.
///
/// The second instruction's GC/liveness snapshot is *not thrown away*: it is
/// captured by `FusionContinuation`, which the future VM handler must consult
/// when it reaches that internal stage. No external control-flow entry is
/// allowed at the eliminated boundary.
pub struct FuseSurvivorSuperinstructions;

struct FusionRewrite {
    first_pc: usize,
    second_pc: usize,
    total_len: usize,
    words: Vec<Word>,
    semantic: BytecodeFusion,
}

fn build_survivor_fusion(
    program: &BytecodeProgram,
    fidx: usize,
    first: InstView,
    second: InstView,
) -> Option<FusionRewrite> {
    use Opcode::*;
    let function = &program.functions[fidx];
    let code = &function.chunk.code;
    let d = first.first.a();
    let special = |view: InstView| function.chunk.specializations
        .get(code.get(view.pc + 1)?.0 as usize);
    let typed_list = match first.op {
        Opcode::AllocList => Some((None, None)),
        Opcode::DefineArrayWithGenerics => match special(first)? {
            SpecializedDescriptor::ArrayType { element, with_capacity } => Some((
                Some(element.clone()), if *with_capacity { Some(Reg(first.first.b())) } else { None },
            )),
            _ => None,
        },
        _ => None,
    };
    let field_load = match first.op {
        Opcode::LoadField => Some((None, code[first.pc + 1].0)),
        Opcode::GetFieldClass => match special(first)? {
            SpecializedDescriptor::FieldClass { class, local_slot } => Some((Some(*class), *local_slot)),
            _ => None,
        },
        _ => None,
    };
    let property_load = match first.op {
        Opcode::LoadProperty => Some((
            StructuralLayoutId(code[first.pc + 1].0),
            PropertySlotId(u16::try_from(code[first.pc + 2].0).ok()?),
        )),
        Opcode::GetFieldShape => match special(first)? {
            SpecializedDescriptor::FieldShape { layout, slot } => Some((*layout, *slot)),
            _ => None,
        },
        _ => None,
    };
    let field_store_slot = match second.op {
        Opcode::StoreField => Some(code[second.pc + 1].0),
        Opcode::SetFieldClass => match special(second)? {
            SpecializedDescriptor::FieldClass { class, local_slot }
                if *class == ClassId(code[first.pc + 1].0) => Some(*local_slot),
            _ => None,
        },
        _ => None,
    };
    let property_store_slot = match second.op {
        Opcode::StoreProperty => Some((
            StructuralLayoutId(code[second.pc + 1].0),
            PropertySlotId(u16::try_from(code[second.pc + 2].0).ok()?),
        )),
        Opcode::SetFieldShape => match special(second)? {
            SpecializedDescriptor::FieldShape { layout, slot } => Some((*layout, *slot)),
            _ => None,
        },
        _ => None,
    };
    let (opcode, a, b, c, semantic, args): (Opcode, u8, u8, u8, BytecodeFusion, Vec<Reg>) =
        match (first.op, second.op) {
        (LoadI8, IAdd | ISub | IMul | IDivS | IRemS | IBitAnd | IBitOr
            | IBitXor | IShl | IShrU | ISar | ICmpEq | ICmpNe | ICmpLt | ICmpLe
            | ICmpGt | ICmpGe)
            if second.first.a() == d
                && (second.first.b() == d) != (second.first.c() == d) => {
            let immediate_first = second.first.b() == d;
            let other = if immediate_first { second.first.c() } else { second.first.b() };
            let opcode = if matches!(second.op, ICmpEq | ICmpNe | ICmpLt | ICmpLe | ICmpGt | ICmpGe) {
                LoadI8IntegerComparison
            } else {
                LoadI8IntegerArithmetic
            };
            (opcode, second.first.a(), other, first.first.b(),
                BytecodeFusion::IntegerImmediate {operation: second.op, immediate_first}, Vec::new())
        }
        (LoadI8, Return) if second.first.a() == d =>
            (LoadI8Return, first.first.b(), 0, 0, BytecodeFusion::ReturnInteger, Vec::new()),
        (LoadNull | LoadFalse | LoadTrue, Return) if second.first.a() == d => {
            let tag = match first.op { LoadNull => 0, LoadFalse => 1, _ => 2 };
            (LoadLiteralReturn, tag, 0, 0, BytecodeFusion::ReturnLiteral, Vec::new())
        }
        (AllocList | DefineArrayWithGenerics, ListAppend)
            if second.first.a() == d && typed_list.is_some() => {
            let (element, capacity) = typed_list?;
            (AllocListAndAppend, d, second.first.b(), 0,
                BytecodeFusion::ListWithFirstElement { element, capacity }, Vec::new())
        },
        (AllocInstance, StoreField | SetFieldClass)
            if second.first.a() == d && field_store_slot.is_some() =>
            (AllocInstanceAndInitializeField, d, second.first.b(), 0,
                BytecodeFusion::InstanceWithField {
                    class: ClassId(code[first.pc + 1].0), slot: field_store_slot?,
                }, Vec::new()),
        (AllocInstance, StoreProperty | SetFieldShape)
            if second.first.a() == d && property_store_slot.is_some() => {
            let (layout, slot) = property_store_slot?;
            (AllocInstanceAndInitializeProperty, d, second.first.c(), second.first.b(),
                BytecodeFusion::InstanceWithProperty {
                    class: ClassId(code[first.pc + 1].0), layout, slot,
                }, Vec::new())
        },
        (LoadField | GetFieldClass, CallClosure)
            if second.first.b() == d && second.first.a() == d && field_load.is_some() =>
            (LoadFieldAndCall, second.first.a(), first.first.b(), second.first.c(),
                BytecodeFusion::FieldClosureCall {
                    class: field_load?.0,
                    slot: field_load?.1,
                    convention: decode_call_conv(code[second.pc + 1].0 as u8)?,
                }, crate::napalm_bytecode::unpack_registers(
                    &code[second.pc + 2..second.pc + second.len], second.first.c() as usize).ok()?),
        (LoadProperty | GetFieldShape, CallClosure)
            if second.first.b() == d && second.first.a() == d && property_load.is_some() =>
            (LoadPropertyAndCall, second.first.a(), first.first.b(), second.first.c(),
                BytecodeFusion::PropertyClosureCall {
                    layout: property_load?.0,
                    slot: property_load?.1,
                    key: Reg(first.first.c()),
                    convention: decode_call_conv(code[second.pc + 1].0 as u8)?,
                }, crate::napalm_bytecode::unpack_registers(
                    &code[second.pc + 2..second.pc + second.len], second.first.c() as usize).ok()?),
        (GuardType | GuardClass, LoadField | GetFieldClass)
            if first.first.a() == first.first.b() && second.first.b() == d =>
            (GuardThenFieldLoad, second.first.a(), first.first.b(), 0,
                BytecodeFusion::GuardedField {
                    class_guard: first.op == GuardClass,
                    guard_id: code[first.pc + 1].0,
                    deopt: DeoptId(code[first.pc + 2].0),
                    slot: match second.op {
                        GetFieldClass => match special(second)? {
                            SpecializedDescriptor::FieldClass { local_slot, .. } => *local_slot,
                            _ => return None,
                        },
                        _ => code[second.pc + 1].0,
                    },
                }, Vec::new()),
        (GuardType | GuardClass, LoadProperty | GetFieldShape)
            if first.first.a() == first.first.b() && second.first.b() == d =>
            (GuardThenPropertyLoad, second.first.a(), first.first.b(), second.first.c(),
                BytecodeFusion::GuardedProperty {
                    class_guard: first.op == GuardClass,
                    guard_id: code[first.pc + 1].0,
                    deopt: DeoptId(code[first.pc + 2].0),
                    layout: match second.op {
                        GetFieldShape => match special(second)? {
                            SpecializedDescriptor::FieldShape { layout, .. } => *layout,
                            _ => return None,
                        },
                        _ => StructuralLayoutId(code[second.pc + 1].0),
                    },
                    slot: match second.op {
                        GetFieldShape => match special(second)? {
                            SpecializedDescriptor::FieldShape { slot, .. } => *slot,
                            _ => return None,
                        },
                        _ => PropertySlotId(u16::try_from(code[second.pc + 2].0).ok()?),
                    },
                }, Vec::new()),
        (ImportModule, ImportVariable)
            if second.first.b() == d && second.first.a() == d =>
            (ImportAndReadVariable, d, 0, 0,
                BytecodeFusion::ImportAndRead {
                    module: StringId(code[first.pc + 1].0),
                    variable: StringId(code[second.pc + 1].0),
                }, Vec::new()),
        (CallDirect, Return) if second.first.a() == d =>
            (DirectCallAndReturn, d, 0, first.first.c(),
                BytecodeFusion::DirectCallAndReturn {
                    function: FuncId(code[first.pc + 1].0),
                    convention: decode_call_conv(first.first.b())?,
                }, crate::napalm_bytecode::unpack_registers(
                    &code[first.pc + 2..first.pc + first.len], first.first.c() as usize).ok()?),
        (CallClosure, Return) if second.first.a() == d =>
            (ClosureCallAndReturn, d, first.first.b(), first.first.c(),
                BytecodeFusion::ClosureCallAndReturn {
                    convention: decode_call_conv(code[first.pc + 1].0 as u8)?,
                }, crate::napalm_bytecode::unpack_registers(
                    &code[first.pc + 2..first.pc + first.len], first.first.c() as usize).ok()?),
        (Launch | LaunchConcurrentTask, Await | AwaitPromise)
            if second.first.b() == d && second.first.a() == d =>
            (ConcurrentLaunchAndAwait, d, first.first.b(), first.first.c(),
                BytecodeFusion::ConcurrentLaunchAndAwait {
                    target: if first.op == LaunchConcurrentTask {
                        match special(first)? {
                            SpecializedDescriptor::Concurrent { function } => Some(*function),
                            _ => return None,
                        }
                    } else { None },
                },
                crate::napalm_bytecode::unpack_registers(
                    &code[first.pc + 2..first.pc + first.len], first.first.c() as usize).ok()?),
        _ => return None,
    };

    if args.len() > crate::napalm_bytecode::MAX_DIRECT_FRAME_VALUES { return None; }
    let mut words = vec![Word::abc(opcode, a, b, c), Word(0)];
    words.extend(pack_registers(&args));
    let total_len = first.len + second.len;
    if words.len() > total_len { return None; }
    Some(FusionRewrite { first_pc: first.pc, second_pc: second.pc, total_len, words, semantic })
}

impl BytecodeOptimization for FuseSurvivorSuperinstructions {
    fn optimize(&mut self, program: &mut BytecodeProgram) -> Result<bool, OptimizationError> {
        let mut changed = false;
        for index in 0..program.functions.len() {
            let barriers = entries(program, index)?;
            let code = &program.functions[index].chunk.code;
            let instructions = decode(code, self.name())?;
            let mut rewrites = Vec::<FusionRewrite>::new();
            let mut next_free = 0;
            for pair in instructions.windows(2) {
                let first = pair[0];
                let second = pair[1];
                if first.pc < next_free || barriers.contains(&second.pc) { continue; }
                let Some(rewrite) = build_survivor_fusion(program, index, first, second) else { continue; };
                // Two different exception handlers must never be collapsed into
                // one dispatch. An exception-range boundary was already treated
                // as a barrier in `entries` above.
                next_free = second.pc + second.len;
                rewrites.push(rewrite);
            }
            if rewrites.is_empty() { continue; }
            let chunk = &mut program.functions[index].chunk;
            for rewrite in rewrites {
                // Capture snapshots of the second primitive instruction before
                // removing its code and PC-keyed metadata.
                let second_pc = BytecodePc(rewrite.second_pc as u32);
                let second_safepoint = chunk.safepoints.iter().find(|item| item.pc == second_pc)
                    .map(|item| (item.kind, item.references.clone()));
                let second_live = chunk.live_registers.iter().find(|item| item.pc == second_pc);
                let continuation = FusionContinuation {
                    gc: second_safepoint,
                    live: second_live.map(|entry| entry.live),
                    constants: second_live.map_or_else(Vec::new, |entry| entry.constants.clone()),
                    types: second_live.map_or_else(Vec::new, |entry| entry.types.clone()),
                };
                let descriptor = chunk.push_specialization(SpecializedDescriptor::Fusion {
                    operation: rewrite.semantic,
                    continuation,
                });
                let mut words = rewrite.words;
                words[1] = Word(descriptor);
                let first_pc = BytecodePc(rewrite.first_pc as u32);
                chunk.code[rewrite.first_pc..rewrite.first_pc + words.len()].copy_from_slice(&words);
                for pc in rewrite.first_pc + words.len()..rewrite.first_pc + rewrite.total_len {
                    nop(&mut chunk.code, pc);
                }
                chunk.safepoints.retain(|entry| entry.pc != second_pc);
                chunk.live_registers.retain(|entry| entry.pc != second_pc);
                // Only source spans are allowed to refer to an eliminated PC.
                // An external control entry, resume or exception boundary is
                // forbidden by the barrier test performed before the rewrite.
                for entry in &mut chunk.source_map {
                    if entry.start_pc == second_pc { entry.start_pc = first_pc; }
                    if entry.end_pc == second_pc { entry.end_pc = first_pc; }
                }
                chunk.source_map.retain(|entry| entry.start_pc.0 < entry.end_pc.0);
                changed = true;
            }
        }
        Ok(changed)
    }
}

/// Enables emitted, descriptor-backed superinstructions. Do NOT run against an
/// interpreter that lacks the matching native handlers. `CompactBytecode` must
/// follow the fuser so the reserved Nop padding is removed and all PCs relocate.
pub fn superinstruction_emission_passes() -> Optimize {
    Optimize::new().with_pass(FuseSurvivorSuperinstructions).with_pass(CompactBytecode)
}

/// Deduplicated candidate set. Each item maps to its own opcode ID 164..179.
/// `MapLookupOrInsert` has a first-class emitter but is not inferred from an
/// unconditional MapGet/MapSet pair, which would change runtime semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuperinstructionCandidate {
    LoadI8IntegerArithmetic,
    LoadI8IntegerComparison,
    LoadI8Return,
    LoadLiteralReturn,
    AllocListAndAppend,
    AllocInstanceAndInitializeField,
    AllocInstanceAndInitializeProperty,
    LoadFieldAndCall,
    LoadPropertyAndCall,
    GuardThenFieldLoad,
    GuardThenPropertyLoad,
    MapLookupOrInsert,
    ImportAndReadVariable,
    DirectCallAndReturn,
    ClosureCallAndReturn,
    ConcurrentLaunchAndAwait,
}

pub const fn superinstruction_candidates() -> &'static [SuperinstructionCandidate] {
    use SuperinstructionCandidate::*;
    &[
        LoadI8IntegerArithmetic, LoadI8IntegerComparison, LoadI8Return,
        LoadLiteralReturn, AllocListAndAppend, AllocInstanceAndInitializeField,
        AllocInstanceAndInitializeProperty, LoadFieldAndCall, LoadPropertyAndCall,
        GuardThenFieldLoad, GuardThenPropertyLoad, MapLookupOrInsert,
        ImportAndReadVariable, DirectCallAndReturn, ClosureCallAndReturn,
        ConcurrentLaunchAndAwait,
    ]
}

/// An actionable fusion site. A hit is **not** proof that it can be lowered
/// without changing GC maps, exception behavior, or runtime side effects.
/// `has_external_entry` must be false before a multi-instruction rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperinstructionSite {
    pub function: FuncId,
    pub start_pc: BytecodePc,
    pub end_pc: BytecodePc,
    pub candidate: SuperinstructionCandidate,
    pub has_external_entry: bool,
    pub crosses_gc_boundary: bool,
}

/// Finds raw, unproven candidate sites. Only `FuseSurvivorSuperinstructions`
/// performs the checked, metadata-preserving emission described below.
pub fn find_superinstruction_sites(program: &BytecodeProgram) -> Result<Vec<SuperinstructionSite>, OptimizationError> {
    use SuperinstructionCandidate::*;
    let mut sites = Vec::new();
    for fi in 0..program.functions.len() {
        let f = &program.functions[fi];
        let chunk = &f.chunk;
        let insts = decode(&chunk.code, "superinstruction planner")?;
        let boundaries = entries(program, fi)?;
        for pair in insts.windows(2) {
            let (first, second) = (pair[0],pair[1]);
            let d = first.first.a();
            let kind = match (first.op,second.op) {
                (Opcode::LoadI8, Opcode::IAdd|Opcode::ISub|Opcode::IMul|
                    Opcode::IDivS|Opcode::IRemS|Opcode::IBitAnd|Opcode::IBitOr|
                    Opcode::IBitXor|Opcode::IShl|Opcode::IShrU|Opcode::ISar)
                    if second.first.b() == d || second.first.c() == d => Some(LoadI8IntegerArithmetic),
                (Opcode::LoadI8, Opcode::ICmpEq|Opcode::ICmpNe|Opcode::ICmpLt|
                    Opcode::ICmpLe|Opcode::ICmpGt|Opcode::ICmpGe)
                    if second.first.b() == d || second.first.c() == d => Some(LoadI8IntegerComparison),
                (Opcode::LoadI8,Opcode::Return) if second.first.a() == d => Some(LoadI8Return),
                (Opcode::LoadNull|Opcode::LoadTrue|Opcode::LoadFalse,Opcode::Return)
                    if second.first.a()==d => Some(LoadLiteralReturn),
                (Opcode::AllocList,Opcode::ListAppend) if second.first.a()==d => Some(AllocListAndAppend),
                (Opcode::AllocInstance,Opcode::StoreProperty) if second.first.a()==d => Some(AllocInstanceAndInitializeProperty),
                (Opcode::AllocInstance,Opcode::StoreField) if second.first.a()==d => Some(AllocInstanceAndInitializeField),
                (Opcode::LoadField,Opcode::CallClosure) if second.first.b()==d => Some(LoadFieldAndCall),
                (Opcode::LoadProperty,Opcode::CallClosure) if second.first.b()==d => Some(LoadPropertyAndCall),
                (Opcode::GuardType|Opcode::GuardClass,Opcode::LoadProperty)
                    if second.first.b()==d => Some(GuardThenPropertyLoad),
                (Opcode::GuardType|Opcode::GuardClass,Opcode::LoadField)
                    if second.first.b()==d => Some(GuardThenFieldLoad),
                (Opcode::ImportModule,Opcode::ImportVariable) if second.first.b()==d => Some(ImportAndReadVariable),
                (Opcode::CallDirect,Opcode::Return) if second.first.a()==d => Some(DirectCallAndReturn),
                (Opcode::CallClosure,Opcode::Return) if second.first.a()==d => Some(ClosureCallAndReturn),
                (Opcode::Launch,Opcode::Await) if second.first.b()==d => Some(ConcurrentLaunchAndAwait),
                _ => None,
            };
            if let Some(candidate)=kind {
                sites.push(SuperinstructionSite {
                    function:f.id,
                    start_pc:BytecodePc(first.pc as u32),
                    end_pc:BytecodePc((second.pc+second.len) as u32),
                    candidate,
                    has_external_entry:boundaries.contains(&second.pc),
                    crosses_gc_boundary:first.op.may_gc()||second.op.may_gc(),
                });
            }
        }

    }
    Ok(sites)
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
    fn fused_arithmetic_has_one_dispatch_and_no_spurious_temp() {
        let mut program = stub_function(vec![
            Word::abc(Opcode::LoadI8, 2, 7, 0),
            Word::abc(Opcode::IAdd, 2, 2, 1),
            Word::abc(Opcode::Return, 2, 0, 0),
        ]);
        assert!(FuseSurvivorSuperinstructions.optimize(&mut program).unwrap());
        let insts = decode(&program.functions[0].chunk.code, "test").unwrap();
        assert_eq!(insts.len(), 2);
        assert_eq!(insts[0].op, Opcode::LoadI8IntegerArithmetic);
        assert_eq!(insts[0].first.c(), 7);
        assert_eq!(insts[1].op, Opcode::Return);
    }

    #[test]
    fn refuses_fusion_when_register_lifetime_would_change() {
        let mut program = stub_function(vec![
            Word::abc(Opcode::LoadI8, 2, 7, 0),
            Word::abc(Opcode::IAdd, 3, 2, 1),
            Word::abc(Opcode::Return, 2, 0, 0),
        ]);
        assert!(!FuseSurvivorSuperinstructions.optimize(&mut program).unwrap());
    }

    #[test]
    fn preserves_second_stage_precise_gc_metadata() {
        use crate::napalm_bytecode::{SafepointEntry, SafepointKind, ReferenceMap};
        let mut program = stub_function(vec![
            Word::abc(Opcode::AllocList, 2, 0, 0),
            Word::abc(Opcode::ListAppend, 2, 3, 0),
            Word::abc(Opcode::Return, 2, 0, 0),
        ]);
        let mut stage_one = ReferenceMap::default();
        stage_one.set_general(Reg(3), true);
        let mut stage_two = ReferenceMap::default();
        stage_two.set_general(Reg(2), true);
        stage_two.set_general(Reg(3), true);
        program.functions[0].chunk.safepoints = vec![
            SafepointEntry { pc: BytecodePc(0), kind: SafepointKind::Allocation, references: stage_one },
            SafepointEntry { pc: BytecodePc(1), kind: SafepointKind::RuntimeMutation, references: stage_two.clone() },
        ];
        assert!(FuseSurvivorSuperinstructions.optimize(&mut program).unwrap());
        let chunk = &program.functions[0].chunk;
        assert_eq!(chunk.safepoints.len(), 1);
        let decoded = decode(&chunk.code, "test").unwrap();
        assert_eq!(decoded[0].op, Opcode::AllocListAndAppend);
        let descriptor = &chunk.specializations[chunk.code[1].0 as usize];
        match descriptor {
            SpecializedDescriptor::Fusion { continuation, .. } => {
                assert_eq!(continuation.gc.as_ref().map(|entry| &entry.1), Some(&stage_two));
            }
            _ => panic!("expected a fused continuation"),
        }
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

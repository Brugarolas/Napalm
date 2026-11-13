//! Ahead-of-time passes. Only the build pipeline installs these transforms.
//! Budgets limit code growth/evaluation work, never the outer HIR fixed point.

use super::*;
use crate::napalm_ir_optimizer::InstFactory;

#[path = "aot/scalar.rs"]
mod scalar;
#[path = "aot/interprocedural.rs"]
mod interprocedural;
#[path = "aot/reachability.rs"]
mod reachability;
pub use scalar::{Sccp, Licm, Gvn, Pre};
pub use interprocedural::{Ipcp, Specialization, Inlining, PartialEvaluation};
pub use reachability::TreeShaking;
#[cfg(test)]
#[path = "aot/tests.rs"]
mod tests;

fn replace_uses(function: &mut Function, from: ValueId, to: ValueId) {
    for block in &function.blocks {
        for &id in &block.insts {
            function.nodes[id.0 as usize].kind.replace_operand(from, to);
        }
    }
    for point in &mut function.deopts {
        for slot in &mut point.snapshot.slots {
            if slot.value == from { slot.value = to; }
        }
        for frame in &mut point.snapshot.inlined_frames {
            for slot in &mut frame.slots {
                if slot.value == from { slot.value = to; }
            }
        }
    }
}

fn constant_eq(a: &Constant, b: &Constant) -> bool {
    match (a, b) {
        (Constant::Num(a), Constant::Num(b)) => a.to_bits() == b.to_bits(),
        _ => a == b,
    }
}

/// Whether a constant can stand for an SSA value of `ty` without changing its
/// register representation. Int32 constants only replace Int32 values, and no
/// boxed constant replaces an unboxed internal representation.
fn constant_fits(constant: &Constant, ty: &Type) -> bool {
    match constant {
        Constant::Int32(_) => *ty == Type::Int32,
        _ => !matches!(ty, Type::Int32 | Type::F64 | Type::RawPtr),
    }
}

/// Materialize a proven constant for an SSA parameter without changing its ABI.
fn bind_constant(function: &mut Function, param: ValueId, constant: Constant) -> bool {
    if !constant_fits(&constant, &function.values[param.0 as usize].ty) { return false; }
    if !function.blocks.iter().flat_map(|b| &b.insts)
        .any(|id| function.nodes[id.0 as usize].kind.operands().contains(&param)) {
        return false;
    }
    let ValueDef::BlockParam(block, _) = function.values[param.0 as usize].def else { return false };
    let ty = function.values[param.0 as usize].ty.clone();
    let (id, values) = InstFactory::new(function).create(Inst::Const(constant), &[ty], None);
    replace_uses(function, param, values[0]);
    function.blocks[block.0 as usize].insts.insert(0, id);
    true
}

/// Speculation whitelist: no memory reads, checks, allocation, division or
/// representation checks. These operations are safe even on a zero-trip path.
fn speculatable(kind: &Inst) -> bool {
    matches!(kind, Inst::Const(_) | Inst::Move(_) | Inst::IntUnary { .. }
        | Inst::IntBin { op: IntBinOp::Add | IntBinOp::Sub | IntBinOp::Mul
            | IntBinOp::BitAnd | IntBinOp::BitOr | IntBinOp::BitXor, .. }
        | Inst::F64Unary { .. } | Inst::F64Bin { .. }
        | Inst::CmpInt { .. } | Inst::CmpF64 { .. } | Inst::RefEq { .. }
        | Inst::Int32ToNum { .. } | Inst::F64ToNum { .. })
}

struct Cfg {
    reachable: Vec<bool>,
    preds: Vec<Vec<usize>>,
    dom: Vec<Vec<bool>>,
    owner: Vec<Option<usize>>,
}

impl Cfg {
    fn new(function: &Function) -> Self {
        let n = function.blocks.len();
        let reachable = compute_reachable(function);
        let mut preds = vec![Vec::new(); n];
        let mut owner = vec![None; function.values.len()];
        for (b, block) in function.blocks.iter().enumerate() {
            for &param in &block.params { owner[param.0 as usize] = Some(b); }
            for &id in &block.insts {
                for &result in &function.nodes[id.0 as usize].results { owner[result.0 as usize] = Some(b); }
            }
            if !reachable[b] { continue; }
            if let Some(term) = block.insts.last() {
                for edge in function.nodes[term.0 as usize].kind.edges() {
                    let target = edge.target.0 as usize;
                    if !preds[target].contains(&b) { preds[target].push(b); }
                }
            }
        }
        let entry = function.entry.0 as usize;
        let mut dom = vec![reachable.clone(); n];
        dom[entry].fill(false);
        dom[entry][entry] = true;
        loop {
            let mut changed = false;
            for b in 0..n {
                if b == entry || !reachable[b] { continue; }
                let mut next = vec![true; n];
                for &pred in &preds[b] {
                    for i in 0..n { next[i] &= dom[pred][i]; }
                }
                next[b] = true;
                if next != dom[b] { dom[b] = next; changed = true; }
            }
            if !changed { break; }
        }
        Self { reachable, preds, dom, owner }
    }

    fn available(&self, function: &Function, value: ValueId, block: usize) -> bool {
        matches!(function.values[value.0 as usize].def, ValueDef::CaptureParam(_))
            || self.owner[value.0 as usize].is_some_and(|owner| self.dom[block][owner])
    }
}

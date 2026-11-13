use super::*;
use crate::napalm_ir::MethodSymbol;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Scalar {
    Null, Bool(bool), Num(u64), Int(i32), String(crate::napalm_ir::StringId), F64(u64),
}
impl Scalar {
    pub(super) fn from_constant(c: &Constant) -> Self {
        match c { Constant::Null => Self::Null, Constant::Bool(v) => Self::Bool(*v),
            Constant::Num(v) => Self::Num(v.to_bits()), Constant::Int32(v) => Self::Int(*v),
            Constant::String(v) => Self::String(*v) }
    }
    pub(super) fn constant(&self) -> Option<Constant> {
        Some(match self { Self::Null => Constant::Null, Self::Bool(v) => Constant::Bool(*v),
            Self::Num(v) => Constant::Num(f64::from_bits(*v)), Self::Int(v) => Constant::Int32(*v),
            Self::String(v) => Constant::String(*v), Self::F64(_) => return None })
    }
    fn int(&self) -> Option<i32> { if let Self::Int(v) = self { Some(*v) } else { None } }
    fn float(&self) -> Option<f64> { if let Self::F64(v) = self { Some(f64::from_bits(*v)) } else { None } }
    fn num(&self) -> Option<f64> { if let Self::Num(v) = self { Some(f64::from_bits(*v)) } else { None } }
    pub(super) fn truthy(&self) -> bool { !matches!(self, Self::Null | Self::Bool(false)) }
}

#[derive(Clone, Copy, Debug)]
enum Operator { Add, Sub, Mul, Div, Mod, Lt, Le, Gt, Ge, Eq, Ne, Neg, Not }

/// Builtin scalar operators by method symbol. Builtin dispatch tables are bound
/// only from the build's own manifest during VM setup and user code cannot
/// rebind them, so an `Invoke` of one of these symbols on Number (or Bool/Null
/// for `!`) receivers always runs the fixed primitive of napalm_builtin
/// (NumAdd, NumMod, NumEq, BoolNot, ...). Folding needs concrete operand values;
/// it never relies on static types.
#[derive(Default)]
pub(super) struct Operators(HashMap<MethodSymbol, Operator>);
impl Operators {
    pub(super) fn new(program: &Program) -> Self {
        use Operator::*;
        let mut operators = HashMap::default();
        for index in 0u32.. {
            let symbol = MethodSymbol(index);
            let Some(signature) = program.method(symbol) else { break };
            if signature.kind != MethodKind::Method { continue; }
            let operator = match (program.string(signature.name), signature.arity) {
                (Some("+"), 1) => Add, (Some("-"), 1) => Sub, (Some("*"), 1) => Mul,
                (Some("/"), 1) => Div, (Some("%"), 1) => Mod, (Some("<"), 1) => Lt,
                (Some("<="), 1) => Le, (Some(">"), 1) => Gt, (Some(">="), 1) => Ge,
                (Some("=="), 1) => Eq, (Some("!="), 1) => Ne, (Some("-"), 0) => Neg,
                (Some("!"), 0) => Not,
                _ => continue,
            };
            operators.insert(symbol, operator);
        }
        Self(operators)
    }

    fn is_operator(&self, kind: &Inst) -> bool {
        matches!(kind, Inst::Invoke { method, .. } if self.0.contains_key(method))
    }
}

/// NaN payloads are not guaranteed to survive the runtime's number boxing, so
/// NaN-producing operations are left to execute at run time.
fn number(value: f64) -> Option<Scalar> {
    (!value.is_nan()).then(|| Scalar::Num(value.to_bits()))
}

/// Shared scalar semantics for SCCP and bounded compile-time evaluation.
/// Unsupported operations, overflow and trapping divisions stay in the program.
pub(super) fn evaluate(kind: &Inst, operators: &Operators, get: impl Fn(ValueId) -> Option<Scalar>) -> Option<Scalar> {
    use Scalar as S;
    Some(match kind {
        Inst::Const(c) => S::from_constant(c),
        Inst::Move(v) => get(*v)?,
        Inst::IntUnary { op, value } => S::Int(fold_int_unary(*op, get(*value)?.int()?)?),
        Inst::IntBin { op, left, right } => S::Int(fold_int_bin(*op, get(*left)?.int()?, get(*right)?.int()?)?),
        Inst::CmpInt { kind, left, right } => S::Bool(compare_i32(*kind, get(*left)?.int()?, get(*right)?.int()?)),
        Inst::CmpF64 { kind, left, right } => S::Bool(compare_f64(*kind, get(*left)?.float()?, get(*right)?.float()?)),
        Inst::Int32ToNum { value } => S::Num((get(*value)?.int()? as f64).to_bits()),
        Inst::NumToInt32 { value } => S::Int(exact_i32(get(*value)?.num()?)?),
        Inst::NumToF64 { value } => S::F64(get(*value)?.num()?.to_bits()),
        Inst::F64ToNum { value } => S::Num(get(*value)?.float()?.to_bits()),
        Inst::F64Unary { op: F64UnaryOp::Neg, value } => S::F64((-get(*value)?.float()?).to_bits()),
        Inst::F64Bin { op, left, right } => {
            let (a, b) = (get(*left)?.float()?, get(*right)?.float()?);
            S::F64(match op { F64BinOp::Add => a+b, F64BinOp::Sub => a-b, F64BinOp::Mul => a*b,
                F64BinOp::Div => a/b, F64BinOp::Rem => a%b }.to_bits())
        }
        Inst::RefEq { left, right } => S::Bool(fold_ref_eq_constants(&get(*left)?.constant()?, &get(*right)?.constant()?)?),
        // A satisfied scalar contract returns its operand; a violated one never
        // returns, so it is simply not folded.
        Inst::BindInterface { value, expected, .. } => {
            let value = get(*value)?;
            match (expected, &value) {
                (Type::Top, _) | (Type::Num, S::Num(_)) | (Type::Bool, S::Bool(_))
                | (Type::Null, S::Null) | (Type::String, S::String(_)) => value,
                _ => return None,
            }
        }
        Inst::Invoke { receiver, method, args, .. } => {
            let operator = *operators.0.get(method)?;
            let receiver = get(*receiver)?;
            match (operator, args.as_slice()) {
                (Operator::Not, []) => match receiver { S::Bool(v) => S::Bool(!v), S::Null => S::Bool(true), _ => return None },
                (Operator::Neg, []) => number(-receiver.num()?)?,
                (operator, [argument]) => {
                    let (a, b) = (receiver.num()?, get(*argument)?.num()?);
                    match operator {
                        Operator::Add => number(a + b)?, Operator::Sub => number(a - b)?,
                        Operator::Mul => number(a * b)?, Operator::Div => number(a / b)?,
                        Operator::Mod => number(a % b)?,
                        Operator::Lt => S::Bool(a < b), Operator::Le => S::Bool(a <= b),
                        Operator::Gt => S::Bool(a > b), Operator::Ge => S::Bool(a >= b),
                        Operator::Eq => S::Bool(a == b), Operator::Ne => S::Bool(a != b),
                        Operator::Neg | Operator::Not => return None,
                    }
                }
                _ => return None,
            }
        }
        _ => return None,
    })
}

#[derive(Clone, PartialEq, Eq)]
enum Lattice { Unknown, Constant(Scalar), Varying }
fn join(old: &mut Lattice, new: Lattice) -> bool {
    let next = match (&*old, new) {
        (Lattice::Varying, _) | (_, Lattice::Varying) => Lattice::Varying,
        (Lattice::Unknown, new) => new,
        (old, Lattice::Unknown) => old.clone(),
        (Lattice::Constant(a), Lattice::Constant(b)) if *a == b => Lattice::Constant(b),
        _ => Lattice::Varying,
    };
    if *old == next { false } else { *old = next; true }
}

fn has_uses(f: &Function, value: ValueId) -> bool {
    f.blocks.iter().flat_map(|b| &b.insts).any(|id| f.nodes[id.0 as usize].kind.operands().contains(&value))
}

/// Sparse conditional constant propagation over executable SSA edges. The
/// lattice is monotone; an initially unknown edge is never treated as dead.
/// Builtin scalar operator invokes are evaluated (see `Operators`), and
/// exception handlers become executable, with unknown incoming values, as soon
/// as any block they protect is executable.
#[derive(Default)]
pub struct Sccp;
impl ProgramOptimization for Sccp {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let operators = Operators::new(program);
        let mut changed = false;
        for f in &mut program.functions { changed |= sccp(f, &operators); }
        changed
    }
}

pub(super) fn sccp(f: &mut Function, operators: &Operators) -> bool {
    if !f.deopts.is_empty() { return false; }
    let mut lattice = vec![Lattice::Unknown; f.values.len()];
    let entry = f.entry.0 as usize;
    for &v in &f.blocks[entry].params { lattice[v.0 as usize] = Lattice::Varying; }
    for c in &f.captures { lattice[c.value.0 as usize] = Lattice::Varying; }
    let mut blocks = vec![false; f.blocks.len()];
    blocks[entry] = true;
    let mut edges = HashSet::default();
    loop {
        let mut changed = false;
        for region in &f.exception_regions {
            let handler = region.handler.0 as usize;
            if !blocks[handler] && region.protected_blocks.iter().any(|b| blocks[b.0 as usize]) {
                blocks[handler] = true;
                for &param in &f.blocks[handler].params { lattice[param.0 as usize] = Lattice::Varying; }
                changed = true;
            }
        }
        for b in 0..f.blocks.len() {
            if !blocks[b] { continue; }
            for &id in &f.blocks[b].insts {
                let node = &f.nodes[id.0 as usize];
                if let [result] = node.results.as_slice() {
                    let value = evaluate(&node.kind, operators, |v| match &lattice[v.0 as usize] {
                        Lattice::Constant(c) => Some(c.clone()), _ => None,
                    });
                    let optimistic = node.kind.effect_behavior() == EffectBehavior::None
                        || operators.is_operator(&node.kind)
                        || matches!(node.kind, Inst::BindInterface { .. });
                    let next = if let Some(v) = value { Lattice::Constant(v) }
                        else if optimistic && node.kind.operands().iter().any(|v| lattice[v.0 as usize] == Lattice::Unknown) {
                            Lattice::Unknown
                        } else { Lattice::Varying };
                    changed |= join(&mut lattice[result.0 as usize], next);
                }
                let outgoing = node.kind.edges();
                let chosen = match &node.kind {
                    Inst::Branch { cond, .. } => match &lattice[cond.0 as usize] {
                        Lattice::Constant(v) => Some(if v.truthy() { 0 } else { 1 }),
                        Lattice::Unknown => continue, _ => None,
                    },
                    Inst::Switch { tag, cases, .. } => match &lattice[tag.0 as usize] {
                        Lattice::Constant(Scalar::Int(v)) => Some(cases.iter().position(|c| c.value == *v as i64).map_or(0, |i| i+1)),
                        Lattice::Unknown => continue, _ => None,
                    },
                    _ => None,
                };
                for (index, edge) in outgoing.iter().enumerate() {
                    if chosen.is_some_and(|chosen| chosen != index) { continue; }
                    changed |= edges.insert((b, index));
                    let target = edge.target.0 as usize;
                    if !blocks[target] { blocks[target] = true; changed = true; }
                    for (&param, &arg) in f.blocks[target].params.iter().zip(&edge.args) {
                        let incoming = lattice[arg.0 as usize].clone();
                        changed |= join(&mut lattice[param.0 as usize], incoming);
                    }
                }
            }
        }
        if !changed { break; }
    }
    let handlers: HashSet<usize> = f.exception_regions.iter().map(|r| r.handler.0 as usize).collect();
    // Values created by the rewrite below (materialized constants) are not in
    // the lattice and are never constants to fold again.
    let constant = |lattice: &[Lattice], f: &Function, value: ValueId| match lattice.get(value.0 as usize) {
        Some(Lattice::Constant(scalar)) => scalar.constant().filter(|c| constant_fits(c, &f.values[value.0 as usize].ty)),
        _ => None,
    };
    let mut changed = false;
    for b in 0..f.blocks.len() {
        if !blocks[b] { continue; }
        if !handlers.contains(&b) {
            for param in f.blocks[b].params.clone() {
                if let Some(c) = constant(&lattice, f, param) { changed |= bind_constant(f, param, c); }
            }
        }
        for id in f.blocks[b].insts.clone() {
            let node = &f.nodes[id.0 as usize];
            if let [result] = node.results.as_slice() {
                let result = *result;
                if let Some(c) = constant(&lattice, f, result).filter(|_| !matches!(node.kind, Inst::Const(_))) {
                    if node.kind.effect_behavior() == EffectBehavior::None || operators.is_operator(&node.kind) {
                        // Pure, or a builtin operator proven to run on scalars.
                        f.nodes[id.0 as usize].kind = Inst::Const(c);
                        changed = true;
                    } else if has_uses(f, result) {
                        // The instruction stays (e.g. a contract check); its
                        // users read the proven constant instead.
                        let ty = f.values[result.0 as usize].ty.clone();
                        let source = f.nodes[id.0 as usize].source;
                        let (const_id, values) = InstFactory::new(f).create(Inst::Const(c), &[ty], source);
                        replace_uses(f, result, values[0]);
                        let pos = f.blocks[b].insts.iter().position(|&other| other == id).unwrap();
                        f.blocks[b].insts.insert(pos + 1, const_id);
                        changed = true;
                    }
                }
            }
            let folded = {
                let outgoing = f.nodes[id.0 as usize].kind.edges();
                let mut taken = outgoing.iter().enumerate().filter(|(i, _)| edges.contains(&(b, *i)));
                match (outgoing.len() > 1, taken.next(), taken.next()) {
                    (true, Some((_, edge)), None) => Some((*edge).clone()),
                    _ => None,
                }
            };
            if let Some(edge) = folded {
                f.nodes[id.0 as usize].kind = Inst::Jump { edge };
                changed = true;
            }
        }
        repair_effect_chain(f, b);
    }
    if changed {
        CopyPropagation.optimize_function(f);
        ControlFlowCleanup.optimize_function(f);
        DeadCodeElimination.optimize_function(f);
    }
    changed
}

/// Dominance-scoped value numbering, revisited after operand canonicalization.
#[derive(Default)]
pub struct Gvn;
impl FunctionOptimization for Gvn {
    fn optimize_function(&mut self, f: &mut Function) -> bool {
        let copies = CopyPropagation.optimize_function(f);
        let numbered = DominatorCSE.optimize_function(f);
        if numbered { CopyPropagation.optimize_function(f); DeadCodeElimination.optimize_function(f); }
        copies | numbered
    }
}

/// Loop-invariant code motion of speculatable (pure, non-trapping) computations
/// into a dedicated preheader that ends in an unconditional jump to the header.
#[derive(Default)]
pub struct Licm;
impl FunctionOptimization for Licm {
    fn optimize_function(&mut self, f: &mut Function) -> bool {
        if !f.exception_regions.is_empty() || !f.deopts.is_empty() { return false; }
        let mut cfg = Cfg::new(f);
        let mut changed = false;
        for header in 0..f.blocks.len() {
            if !cfg.reachable[header] { continue; }
            let tails: Vec<_> = cfg.preds[header].iter().copied().filter(|&p| cfg.dom[p][header]).collect();
            if tails.is_empty() { continue; }
            let mut members = HashSet::default();
            members.insert(header);
            let mut work = tails;
            while let Some(b) = work.pop() {
                if members.insert(b) { work.extend(cfg.preds[b].iter().copied()); }
            }
            let outside: Vec<_> = cfg.preds[header].iter().copied().filter(|p| !members.contains(p)).collect();
            let [preheader] = outside.as_slice() else { continue };
            let preheader = *preheader;
            let Some(&term) = f.blocks[preheader].insts.last() else { continue };
            if !matches!(&f.nodes[term.0 as usize].kind, Inst::Jump { edge } if edge.target.0 as usize == header) { continue; }
            loop {
                let mut moved = false;
                for b in 0..f.blocks.len() {
                    if !members.contains(&b) { continue; }
                    for id in f.blocks[b].insts.clone() {
                        let node = &f.nodes[id.0 as usize];
                        // Constants and copies are free to rematerialize; hoisting
                        // them only lengthens live ranges (register pressure in the
                        // interpreter, loop-carried state in JIT traces).
                        if node.results.len() != 1 || !speculatable(&node.kind)
                            || matches!(node.kind, Inst::Const(_) | Inst::Move(_))
                            || !node.kind.operands().iter().all(|&v| cfg.available(f, v, preheader)) { continue; }
                        let result = node.results[0];
                        f.blocks[b].insts.retain(|&other| other != id);
                        let pos = f.blocks[preheader].insts.len()-1;
                        f.blocks[preheader].insts.insert(pos, id);
                        cfg.owner[result.0 as usize] = Some(preheader);
                        moved = true;
                        changed = true;
                    }
                }
                if !moved { break; }
            }
        }
        changed
    }
}

/// SSA PRE at joins with unconditional predecessors. Translate join parameters
/// along each incoming edge, reuse available values, and insert only on missing
/// paths. Critical edges/exception regions are left for a future extension.
#[derive(Default)]
pub struct Pre;
impl FunctionOptimization for Pre {
    fn optimize_function(&mut self, f: &mut Function) -> bool {
        if !f.exception_regions.is_empty() || !f.deopts.is_empty() { return false; }
        // Each step moves one expression strictly up an acyclic join edge, so
        // the local loop terminates; the CFG itself never changes.
        let mut changed = false;
        while pre_step(f) { changed = true; }
        changed
    }
}

fn pre_step(f: &mut Function) -> bool {
    let cfg = Cfg::new(f);
    for b in 0..f.blocks.len() {
        if !cfg.reachable[b] || cfg.preds[b].len() < 2 || f.blocks[b].params.len() >= u16::MAX as usize { continue; }
        if cfg.preds[b].iter().any(|&p| cfg.dom[p][b] || !matches!(
            &f.nodes[f.blocks[p].insts.last().unwrap().0 as usize].kind,
            Inst::Jump { edge } if edge.target.0 as usize == b)) { continue; }
        for id in f.blocks[b].insts.clone() {
            let node = &f.nodes[id.0 as usize];
            let [result] = node.results.as_slice() else { continue };
            let result = *result;
            if !speculatable(&node.kind) || matches!(node.kind, Inst::Const(_) | Inst::Move(_)) { continue; }
            let result_ty = f.values[result.0 as usize].ty.clone();
            let mut plans = Vec::new();
            for &pred in &cfg.preds[b] {
                let term = *f.blocks[pred].insts.last().unwrap();
                let Inst::Jump { edge } = &f.nodes[term.0 as usize].kind else { unreachable!() };
                let mut translated = node.kind.clone();
                translated.for_each_operand_mut(|v| {
                    if let Some(i) = f.blocks[b].params.iter().position(|p| p == v) { *v = edge.args[i]; }
                });
                if !translated.operands().iter().all(|&v| cfg.available(f, v, pred)) { break; }
                let Some(key) = pure_key(&translated) else { break };
                let available = f.blocks.iter().enumerate().filter(|(owner, _)| cfg.dom[pred][*owner])
                    .flat_map(|(_, block)| &block.insts).find_map(|&other| {
                        let n = &f.nodes[other.0 as usize];
                        (pure_key(&n.kind).as_ref() == Some(&key)).then(|| n.results.first().copied()).flatten()
                            .filter(|v| f.values[v.0 as usize].ty == result_ty)
                    });
                plans.push((pred, term, translated, available));
            }
            if plans.len() != cfg.preds[b].len() || plans.iter().all(|(_, _, _, v)| v.is_none()) { continue; }
            let param = ValueId(f.values.len() as u32);
            f.values.push(crate::napalm_ir::Value { id: param, ty: result_ty.clone(),
                def: ValueDef::BlockParam(BlockId(b as u32), f.blocks[b].params.len() as u16) });
            f.blocks[b].params.push(param);
            for (pred, term, expression, value) in plans {
                let value = value.unwrap_or_else(|| {
                    let (new_id, results) = InstFactory::new(f).create(expression, &[result_ty.clone()], None);
                    let pos = f.blocks[pred].insts.len()-1;
                    f.blocks[pred].insts.insert(pos, new_id);
                    results[0]
                });
                let Inst::Jump { edge } = &mut f.nodes[term.0 as usize].kind else { unreachable!() };
                edge.args.push(value);
            }
            f.nodes[id.0 as usize].kind = Inst::Move(param);
            CopyPropagation.optimize_function(f);
            return true;
        }
    }
    false
}

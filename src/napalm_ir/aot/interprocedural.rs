use super::*;
use super::scalar::{Operators, Scalar, evaluate, sccp};
use crate::napalm_ir::{Block, EffectDef, EffectId, EffectValue, SpecializationOrigin, Value};

fn eligible(f: &Function) -> bool {
    !f.is_async && !f.is_concurrent && f.captures.is_empty() && f.deopts.is_empty()
        && !f.memoization.is_reactive()
        && matches!(f.kind, FunctionKind::Closure | FunctionKind::StaticMethod | FunctionKind::Method)
}

fn core(program: &Program, f: &Function) -> bool {
    program.module(f.module).and_then(|m| program.string(m.name)).is_some_and(|name| name.starts_with("<core:"))
}

fn size(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// Instructions that do work at run time (constants and copies are free).
fn work(f: &Function) -> usize {
    f.blocks.iter().flat_map(|b| &b.insts)
        .filter(|id| !matches!(f.nodes[id.0 as usize].kind, Inst::Const(_) | Inst::Move(_))).count()
}

/// Whether `from` can reach `target` through direct calls (including itself).
fn reaches(program: &Program, from: FuncId, target: FuncId, seen: &mut HashSet<FuncId>) -> bool {
    if from == target { return true; }
    if !seen.insert(from) { return false; }
    let Some(function) = program.function(from) else { return true };
    function.blocks.iter().flat_map(|b| &b.insts).any(|id| match &function.nodes[id.0 as usize].kind {
        Inst::CallDirect { callee, .. } | Inst::TailCallDirect { callee, .. } => reaches(program, *callee, target, seen),
        _ => false,
    })
}

/// Whether `id` belongs to a direct-call cycle; a specialization clone counts
/// as its origin. Inlining or specializing such a function would unroll one
/// recursion level per HIR round (clone, inline, shake the dead clone, repeat).
fn recursive(program: &Program, id: FuncId) -> bool {
    let Some(function) = program.function(id) else { return true };
    let origin = function.specialization.map_or(id, |s| s.origin);
    function.blocks.iter().flat_map(|b| &b.insts).any(|inst| match &function.nodes[inst.0 as usize].kind {
        Inst::CallDirect { callee, .. } | Inst::TailCallDirect { callee, .. } =>
            reaches(program, *callee, id, &mut HashSet::default())
                || reaches(program, *callee, origin, &mut HashSet::default()),
        _ => false,
    })
}

/// IPCP for closed direct-call targets. Address-taken targets keep their generic
/// entry semantics; per-callsite cloning below handles those safely instead.
#[derive(Default)]
pub struct Ipcp;
impl ProgramOptimization for Ipcp {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let mut exposed = HashSet::default();
        for f in &program.functions {
            for block in &f.blocks {
                for &id in &block.insts {
                    match &f.nodes[id.0 as usize].kind {
                        Inst::MakeClosure { function, .. } | Inst::BindMethod { function, .. } => { exposed.insert(*function); }
                        _ => {}
                    }
                }
            }
        }
        for m in &program.modules { for binding in &m.worker_function_bindings { exposed.insert(binding.function); } }
        let mut plans = Vec::new();
        for target in &program.functions {
            if !eligible(target) || core(program, target) || exposed.contains(&target.id) { continue; }
            let params = &target.blocks[target.entry.0 as usize].params;
            let mut facts: Option<Vec<Option<Constant>>> = None;
            for caller in &program.functions {
                for &id in caller.blocks.iter().flat_map(|b| &b.insts) {
                    let args = match &caller.nodes[id.0 as usize].kind {
                        Inst::CallDirect { conv: CallConv::Napalm, callee, args }
                        | Inst::TailCallDirect { conv: CallConv::Napalm, callee, args } if *callee == target.id => args,
                        _ => continue,
                    };
                    if args.len() != params.len() { continue; }
                    let incoming: Vec<_> = args.iter().map(|v| resolve_constant(caller, *v, 0)).collect();
                    if let Some(facts) = &mut facts {
                        for (fact, value) in facts.iter_mut().zip(incoming) {
                            if !matches!((&*fact, value.as_ref()), (Some(a), Some(b)) if constant_eq(a,b)) { *fact = None; }
                        }
                    } else { facts = Some(incoming); }
                }
            }
            if let Some(facts) = facts { plans.push((target.id, params.clone(), facts)); }
        }
        let mut changed = false;
        for (id, params, facts) in plans {
            let f = program.function_mut(id).unwrap();
            for (param, fact) in params.into_iter().zip(facts) {
                if let Some(c) = fact { changed |= bind_constant(f, param, c); }
            }
        }
        changed
    }
}

/// Canonical call-site facts. Constants are keyed by exact bits (NaN payloads,
/// -0.0), never by a lossy textual float rendering.
fn facts_key(target: FuncId, constants: &[Option<Constant>], objects: &[Option<Type>]) -> String {
    let constants: Vec<_> = constants.iter().map(|c| c.as_ref().map(Scalar::from_constant)).collect();
    let objects = serde_json::to_string(objects).expect("object specialization facts");
    format!("{}:{constants:?}:{objects}", target.0)
}

const MAX_SPECIALIZATIONS: usize = 128;
const MAX_VERSIONS_PER_FUNCTION: usize = 8;
const MAX_SPECIALIZED_SIZE: usize = 256;

/// Callsite specialization for constant parameters AND exact persistent object
/// contracts/class sets (interprocedural object specialization): a narrowed
/// parameter type lets StaticTypeSpecialization devirtualize invokes and resolve
/// structural property slots inside the clone on the next HIR round. Generic
/// originals remain valid for escaping closures. Clones keep the original name
/// and record their facts in `Function::specialization`, so repeated HIR rounds
/// reuse versions instead of re-cloning.
#[derive(Default)]
pub struct Specialization;
impl ProgramOptimization for Specialization {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let mut plans = Vec::new();
        for caller in &program.functions {
            for &id in caller.blocks.iter().flat_map(|b| &b.insts) {
                let (target, args) = match &caller.nodes[id.0 as usize].kind {
                    Inst::CallDirect { conv: CallConv::Napalm, callee, args }
                    | Inst::TailCallDirect { conv: CallConv::Napalm, callee, args } => (*callee, args),
                    _ => continue,
                };
                let Some(callee) = program.function(target) else { continue };
                if target == caller.id || !eligible(callee) || core(program, callee)
                    // A clone would bypass the original's memo table.
                    || callee.memoization.enabled()
                    || callee.specialization.is_some() || recursive(program, target)
                    || callee.signature.params.len() != args.len()
                    || size(callee) > MAX_SPECIALIZED_SIZE { continue; }
                let params = &callee.blocks[callee.entry.0 as usize].params;
                let used: HashSet<_> = callee.blocks.iter().flat_map(|b| &b.insts)
                    .flat_map(|id| callee.nodes[id.0 as usize].kind.operands()).collect();
                // A narrower object type only pays off where type specialization
                // can use it: dispatch, property/field access, contract checks.
                let receivers: HashSet<ValueId> = callee.blocks.iter().flat_map(|b| &b.insts)
                    .filter_map(|id| match &callee.nodes[id.0 as usize].kind {
                        Inst::Invoke { receiver, .. } | Inst::InvokeSuper { receiver, .. }
                        | Inst::TailInvoke { receiver, .. } | Inst::TailInvokeSuper { receiver, .. } => Some(*receiver),
                        Inst::LoadProperty { object, .. } | Inst::StoreProperty { object, .. }
                        | Inst::LoadField { object, .. } | Inst::StoreField { object, .. } => Some(*object),
                        Inst::BindInterface { value, .. } => Some(*value),
                        _ => None,
                    }).collect();
                let constants: Vec<_> = args.iter().zip(params).map(|(v,p)| {
                    used.contains(p).then(|| resolve_constant(caller, *v, 0)).flatten()
                        .filter(|c| constant_fits(c, &callee.values[p.0 as usize].ty))
                }).collect();
                let objects: Vec<_> = args.iter().zip(params).map(|(arg,param)| {
                    let actual = &caller.values[arg.0 as usize].ty;
                    let expected = &callee.values[param.0 as usize].ty;
                    (receivers.contains(param) && actual != expected && expected.accepts(actual)
                        && matches!(actual, Type::Record(_) | Type::Interface(_) | Type::Instance(_) | Type::GenericInstance { .. }))
                        .then(|| actual.clone())
                }).collect();
                if constants.iter().all(Option::is_none) && objects.iter().all(Option::is_none) { continue; }
                let key = facts_key(target, &constants, &objects);
                plans.push((caller.id, id, target, key, constants, objects));
            }
        }
        let operators = Operators::new(program);
        let mut changed = false;
        for (caller, call, original, key, constants, objects) in plans {
            let facts = program.intern_string(key);
            let wanted = SpecializationOrigin { origin: original, facts };
            let existing = program.functions.iter().find(|f| f.specialization == Some(wanted)).map(|f| f.id);
            let version = if let Some(id) = existing { id } else {
                // Bound growth across repeated outer rounds, not just one call.
                let count = program.functions.iter().filter(|f| f.specialization.is_some()).count();
                let versions = program.functions.iter()
                    .filter(|f| f.specialization.is_some_and(|s| s.origin == original)).count();
                if count >= MAX_SPECIALIZATIONS || versions >= MAX_VERSIONS_PER_FUNCTION { continue; }
                // Build and judge the clone before touching the program: ID and
                // site counters are part of the HIR snapshot, so reserving them
                // for a discarded clone would keep the fixed point from settling.
                let mut clone = program.function(original).unwrap().clone();
                clone.specialization = Some(wanted);
                clone.profiles = Default::default();
                clone.memoization = crate::napalm_ir::MemoizationKind::None;
                let params = clone.blocks[clone.entry.0 as usize].params.clone();
                for (i, param) in params.into_iter().enumerate() {
                    if let Some(c) = &constants[i] { bind_constant(&mut clone, param, c.clone()); }
                    if let Some(ty) = &objects[i] {
                        clone.signature.params[i] = ty.clone();
                        clone.values[param.0 as usize].ty = ty.clone();
                    }
                }
                // Constant facts must remove work. An unprofitable clone still
                // costs code size and splits JIT warm-up/profiles across copies
                // (e.g. a benchmark's warm-up call and its measured call).
                if objects.iter().all(Option::is_none) {
                    sccp(&mut clone, &operators);
                    if work(&clone) >= work(program.function(original).unwrap()) { continue; }
                }
                clone.id = program.reserve_function_id();
                for node in &mut clone.nodes {
                    match &mut node.kind {
                        Inst::Invoke { site, .. } | Inst::InvokeSuper { site, .. }
                        | Inst::TailInvoke { site, .. } | Inst::TailInvokeSuper { site, .. } => *site = program.fresh_site(),
                        _ => {}
                    }
                }
                program.push_function(clone)
            };
            let node = &mut program.function_mut(caller).unwrap().nodes[call.0 as usize];
            match &mut node.kind {
                Inst::CallDirect { callee, .. } | Inst::TailCallDirect { callee, .. } => *callee = version,
                _ => unreachable!(),
            }
            changed = true;
        }
        changed
    }
}

/// Straight-line helpers are copied in place (no CFG split); any remaining
/// small, closed callee is then inlined with its whole CFG.
#[derive(Default)]
pub struct Inlining;
impl ProgramOptimization for Inlining {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let straight = !super::super::static_type_specialization::inline_aot_calls(program).is_empty();
        straight | inline_cfg_calls(program)
    }
}

/// Callee size limit (instructions, all blocks) for whole-CFG inlining.
const MAX_CFG_INLINE_INSTS: usize = 64;
/// A caller stops receiving inlined code once it reaches this size.
const MAX_INLINE_CALLER_INSTS: usize = 512;

/// Callee-side legality of whole-CFG inlining. Fiber switches, awaits and
/// deopt-based speculation depend on the callee's own frame; memoized functions
/// must keep their call boundary (memo table / reactive validation). `throw`
/// (FiberAbort) is fine: an inlined throw reaches exactly the handlers a throw
/// propagating out of the call would. Context that depends on the executing
/// function (field layout, module-relative primitives) is checked per caller in
/// `cfg_inlinable_into`.
fn cfg_inlinable(program: &Program, callee: &Function) -> bool {
    eligible(callee) && !core(program, callee) && callee.exception_regions.is_empty()
        && !callee.memoization.enabled() && size(callee) <= MAX_CFG_INLINE_INSTS
        && !has_loop(callee)
        && callee.blocks.iter().flat_map(|b| &b.insts).all(|id| !matches!(
            callee.nodes[id.0 as usize].kind,
            Inst::GuardType { .. } | Inst::GuardClass { .. } | Inst::GuardBounds { .. }
                | Inst::GuardNotNull { .. } | Inst::GuardInt32 { .. } | Inst::GuardF64 { .. }
                | Inst::Bailout { .. } | Inst::Await { .. } | Inst::MakeFiber { .. }
                | Inst::FiberCall { .. } | Inst::FiberTry { .. } | Inst::FiberTransfer { .. }
                | Inst::FiberTransferError { .. } | Inst::FiberYield { .. } | Inst::FiberSuspend
        ))
}

/// A loop amortizes its call overhead, while inlining it moves the hot loop into
/// the caller (often a module body), where traces carry the caller's whole
/// register state and the callee loses its own JIT entry/OSR anchor.
fn has_loop(f: &Function) -> bool {
    let cfg = Cfg::new(f);
    (0..f.blocks.len()).any(|header| cfg.reachable[header] && cfg.preds[header].iter().any(|&p| cfg.dom[p][header]))
}

fn cfg_inlinable_into(caller: &Function, callee: &Function) -> bool {
    let insts = || callee.blocks.iter().flat_map(|b| &b.insts).map(|id| &callee.nodes[id.0 as usize].kind);
    // FieldSlots are local to the compiling function's field owner.
    (callee.field_owner == caller.field_owner
        || insts().all(|kind| !matches!(kind, Inst::LoadField { .. } | Inst::StoreField { .. })))
    // Keep anything that might resolve relative to the running module at home.
    && (callee.module == caller.module
        || insts().all(|kind| !matches!(kind, Inst::CallPrimitive { .. } | Inst::CallForeign { .. }
            | Inst::ImportModule { .. } | Inst::ImportVariable { .. })))
}

fn inline_cfg_calls(program: &mut Program) -> bool {
    let mut changed = false;
    for index in 0..program.functions.len() {
        let caller = &program.functions[index];
        if !caller.deopts.is_empty() { continue; }
        let caller_id = caller.id;
        let mut remaining = MAX_INLINE_CALLER_INSTS.saturating_sub(size(caller));
        let mut sites = Vec::new();
        for &id in caller.blocks.iter().flat_map(|b| &b.insts) {
            let node = &caller.nodes[id.0 as usize];
            let Inst::CallDirect { conv: CallConv::Napalm, callee, args } = &node.kind else { continue };
            let Some(target) = program.function(*callee) else { continue };
            let cost = size(target);
            if *callee == caller_id || node.results.len() != 1 || args.len() != target.signature.params.len()
                || cost > remaining || !cfg_inlinable(program, target) || !cfg_inlinable_into(caller, target)
                || reaches(program, *callee, caller_id, &mut HashSet::default())
                || recursive(program, *callee) { continue; }
            remaining -= cost;
            sites.push((id, *callee));
        }
        for (call, callee) in sites {
            let body = program.function(callee).unwrap().clone();
            let invokes = body.nodes.iter().filter(|n| matches!(n.kind,
                Inst::Invoke { .. } | Inst::InvokeSuper { .. } | Inst::TailInvoke { .. } | Inst::TailInvokeSuper { .. })).count();
            let fresh: Vec<_> = (0..invokes).map(|_| program.fresh_site()).collect();
            inline_cfg_site(&mut program.functions[index], call, &body, &mut fresh.into_iter());
            changed = true;
        }
    }
    changed
}

fn new_block(f: &mut Function, types: impl IntoIterator<Item = Type>) -> BlockId {
    let id = BlockId(f.blocks.len() as u32);
    let effect = EffectId(f.effects.len() as u32);
    f.effects.push(EffectValue { id: effect, def: EffectDef::BlockParam(id) });
    let mut params = Vec::new();
    for (index, ty) in types.into_iter().enumerate() {
        let value = ValueId(f.values.len() as u32);
        f.values.push(Value { id: value, ty, def: ValueDef::BlockParam(id, index as u16) });
        params.push(value);
    }
    f.blocks.push(Block { id, params, effect, insts: Vec::new(), sealed: true });
    id
}

fn push_node(f: &mut Function, kind: Inst, results: SmallVec<[ValueId; 1]>, source: Option<crate::napalm_ir::SourceSpan>) -> InstId {
    let id = InstId(f.nodes.len() as u32);
    for (index, value) in results.iter().enumerate() {
        f.values[value.0 as usize].def = ValueDef::InstResult(id, index as u8);
    }
    f.nodes.push(Node { id, kind, results, effect_in: None, effect_out: None, source });
    id
}

/// Split the caller block at `call`, copy every callee block, route returns to
/// a continuation block whose parameter replaces the call result, and convert
/// tail calls into ordinary calls that return through the continuation. The
/// call node itself survives as a Move, so all uses of its result stay valid.
/// The new blocks join every exception region protecting the call.
fn inline_cfg_site(caller: &mut Function, call: InstId, callee: &Function, sites: &mut impl Iterator<Item = SiteId>) {
    let (b, pos) = caller.blocks.iter().enumerate()
        .find_map(|(b, block)| block.insts.iter().position(|&id| id == call).map(|pos| (b, pos)))
        .expect("inlined call site in a live block");
    let Inst::CallDirect { args, .. } = caller.nodes[call.0 as usize].kind.clone() else { unreachable!() };
    let result = caller.nodes[call.0 as usize].results[0];
    let result_ty = caller.values[result.0 as usize].ty.clone();
    let call_source = caller.nodes[call.0 as usize].source;
    let first_new = caller.blocks.len();

    let cont = new_block(caller, [result_ty.clone()]);
    let tail = caller.blocks[b].insts.split_off(pos + 1);
    caller.blocks[b].insts.pop();
    caller.nodes[call.0 as usize].kind = Inst::Move(caller.blocks[cont.0 as usize].params[0]);
    caller.blocks[cont.0 as usize].insts = std::iter::once(call).chain(tail).collect();

    let mut values: HashMap<ValueId, ValueId> = HashMap::default();
    let mut blocks = Vec::with_capacity(callee.blocks.len());
    for block in &callee.blocks {
        let types: Vec<_> = block.params.iter().map(|v| callee.values[v.0 as usize].ty.clone()).collect();
        let id = new_block(caller, types);
        for (old, new) in block.params.iter().zip(caller.blocks[id.0 as usize].params.clone()) { values.insert(*old, new); }
        blocks.push(id);
    }
    // Results first: operands may be defined in blocks laid out later.
    for &id in callee.blocks.iter().flat_map(|b| &b.insts) {
        for &old in &callee.nodes[id.0 as usize].results {
            let new = ValueId(caller.values.len() as u32);
            caller.values.push(Value { id: new, ty: callee.values[old.0 as usize].ty.clone(), def: ValueDef::BlockParam(cont, 0) });
            values.insert(old, new);
        }
    }
    let to_cont = |value: ValueId| Inst::Jump { edge: Edge { target: cont, args: vec![value], effect: EffectId(0) } };
    for (block, &new_block_id) in callee.blocks.iter().zip(&blocks) {
        let mut insts = Vec::with_capacity(block.insts.len());
        for &id in &block.insts {
            let node = &callee.nodes[id.0 as usize];
            let mut kind = node.kind.clone();
            kind.for_each_operand_mut(|v| *v = values[v]);
            match &mut kind {
                Inst::Jump { edge } => edge.target = blocks[edge.target.0 as usize],
                Inst::Branch { then_edge, else_edge, .. } => {
                    then_edge.target = blocks[then_edge.target.0 as usize];
                    else_edge.target = blocks[else_edge.target.0 as usize];
                }
                Inst::Switch { default_edge, cases, .. } => {
                    default_edge.target = blocks[default_edge.target.0 as usize];
                    for case in cases { case.edge.target = blocks[case.edge.target.0 as usize]; }
                }
                Inst::Invoke { site, .. } | Inst::InvokeSuper { site, .. }
                | Inst::TailInvoke { site, .. } | Inst::TailInvokeSuper { site, .. } => *site = sites.next().expect("fresh site"),
                _ => {}
            }
            let call_kind = match kind {
                Inst::Return(value) => { insts.push(push_node(caller, to_cont(value), SmallVec::new(), node.source)); continue; }
                Inst::TailCallDirect { conv, callee, args } => Inst::CallDirect { conv, callee, args },
                Inst::TailCallClosure { conv, closure, args } => Inst::CallClosure { conv, closure, args },
                Inst::TailInvoke { site, receiver, method, args } => Inst::Invoke { site, receiver, method, args },
                Inst::TailInvokeSuper { site, home_class, receiver, method, args } =>
                    Inst::InvokeSuper { site, home_class, receiver, method, args },
                kind => {
                    let results = node.results.iter().map(|v| values[v]).collect();
                    insts.push(push_node(caller, kind, results, node.source));
                    continue;
                }
            };
            let value = ValueId(caller.values.len() as u32);
            caller.values.push(Value { id: value, ty: result_ty.clone(), def: ValueDef::BlockParam(cont, 0) });
            insts.push(push_node(caller, call_kind, smallvec::smallvec![value], node.source));
            insts.push(push_node(caller, to_cont(value), SmallVec::new(), node.source));
        }
        caller.blocks[new_block_id.0 as usize].insts = insts;
    }
    // The call boundary trusts the callee's parameter types without a runtime
    // check. A narrowing Move expresses the same contract on the entry edge;
    // copy propagation never substitutes a value of a wider type through it.
    let entry_block = blocks[callee.entry.0 as usize];
    let mut entry_args = Vec::with_capacity(args.len());
    for (arg, param) in args.into_iter().zip(caller.blocks[entry_block.0 as usize].params.clone()) {
        let expected = caller.values[param.0 as usize].ty.clone();
        if expected.accepts(&caller.values[arg.0 as usize].ty) { entry_args.push(arg); continue; }
        let (id, results) = InstFactory::new(caller).create(Inst::Move(arg), &[expected], call_source);
        caller.blocks[b].insts.push(id);
        entry_args.push(results[0]);
    }
    let entry = Inst::Jump { edge: Edge { target: entry_block, args: entry_args, effect: EffectId(0) } };
    let jump = push_node(caller, entry, SmallVec::new(), call_source);
    caller.blocks[b].insts.push(jump);

    let added: Vec<BlockId> = (first_new..caller.blocks.len()).map(|i| BlockId(i as u32)).collect();
    for region in &mut caller.exception_regions {
        if region.protected_blocks.contains(&BlockId(b as u32)) { region.protected_blocks.extend(&added); }
    }
    repair_effect_chain(caller, b);
    for block in added { repair_effect_chain(caller, block.0 as usize); }
}

/// Interpreter for a strict scalar HIR subset. It never enters the VM, executes
/// native/user callbacks, reads module/heap state, allocates objects or removes
/// a call whose evaluated path could trap. Recursion/loops share a fuel budget;
/// tail calls replace the current activation instead of nesting.
fn evaluate_call(program: &Program, operators: &Operators, id: FuncId, args: &[Scalar], fuel: &mut usize, depth: usize) -> Option<Scalar> {
    if depth >= 32 { return None; }
    let (mut f, mut incoming) = (program.function(id)?, args.to_vec());
    'activation: loop {
        if !eligible(f) || !f.exception_regions.is_empty() { return None; }
        let mut values = vec![None; f.values.len()];
        let mut block = f.entry;
        loop {
            let current = &f.blocks[block.0 as usize];
            if current.params.len() != incoming.len() { return None; }
            for (&param, value) in current.params.iter().zip(incoming.drain(..)) { values[param.0 as usize] = Some(value); }
            let mut next = None;
            for (position, &id) in current.insts.iter().enumerate() {
                *fuel = fuel.checked_sub(1)?;
                let node = &f.nodes[id.0 as usize];
                // `return callee(...)` is a tail call even when not marked as one.
                if let Inst::CallDirect { conv: CallConv::Napalm, callee, args } = &node.kind {
                    if matches!(current.insts.get(position + 1).map(|next| &f.nodes[next.0 as usize].kind),
                        Some(Inst::Return(value)) if node.results.first() == Some(value)) {
                        incoming = args.iter().map(|v| values[v.0 as usize].clone()).collect::<Option<Vec<_>>>()?;
                        f = program.function(*callee)?;
                        continue 'activation;
                    }
                }
                match &node.kind {
                    // GC polls / tiering hooks have no value semantics.
                    Inst::Safepoint | Inst::OSRPoint => continue,
                    Inst::Return(value) => return values[value.0 as usize].clone(),
                    Inst::Jump { edge } => { next = Some(edge); break; }
                    Inst::Branch { cond, then_edge, else_edge } => {
                        next = Some(if values[cond.0 as usize].as_ref()?.truthy() { then_edge } else { else_edge }); break;
                    }
                    Inst::Switch { tag, cases, default_edge } => {
                        let Scalar::Int(tag) = values[tag.0 as usize].as_ref()? else { return None };
                        next = Some(cases.iter().find(|c| c.value == *tag as i64).map_or(default_edge, |c| &c.edge)); break;
                    }
                    Inst::TailCallDirect { conv: CallConv::Napalm, callee, args } => {
                        incoming = args.iter().map(|v| values[v.0 as usize].clone()).collect::<Option<Vec<_>>>()?;
                        f = program.function(*callee)?;
                        continue 'activation;
                    }
                    _ => {}
                }
                let [result] = node.results.as_slice() else { return None };
                let value = if let Inst::CallDirect { conv: CallConv::Napalm, callee, args } = &node.kind {
                    let args = args.iter().map(|v| values[v.0 as usize].clone()).collect::<Option<Vec<_>>>()?;
                    evaluate_call(program, operators, *callee, &args, fuel, depth+1)?
                } else { evaluate(&node.kind, operators, |v| values[v.0 as usize].clone())? };
                values[result.0 as usize] = Some(value);
            }
            let edge = next?;
            incoming = edge.args.iter().map(|v| values[v.0 as usize].clone()).collect::<Option<Vec<_>>>()?;
            block = edge.target;
        }
    }
}

/// Interpreted HIR instructions allowed for one folded call (callees included).
const PARTIAL_EVALUATION_FUEL: usize = 8192;

/// Fold direct calls whose arguments are all constants by evaluating the callee
/// at compile time. Tail calls fold into `return constant`.
#[derive(Default)]
pub struct PartialEvaluation;
impl ProgramOptimization for PartialEvaluation {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let operators = Operators::new(program);
        let mut plans = Vec::new();
        // Per call site, so sites that fail to evaluate never starve the rest;
        // the round total only bounds pathological programs.
        let mut fuel = 1usize << 22;
        for f in &program.functions {
            for (b, block) in f.blocks.iter().enumerate() {
                for &id in &block.insts {
                    let node = &f.nodes[id.0 as usize];
                    let (callee, args, ty) = match &node.kind {
                        Inst::CallDirect { conv: CallConv::Napalm, callee, args } =>
                            (*callee, args, &f.values[node.results[0].0 as usize].ty),
                        Inst::TailCallDirect { conv: CallConv::Napalm, callee, args } => (*callee, args, &f.signature.result),
                        _ => continue,
                    };
                    let constants = args.iter().map(|v| resolve_constant(f, *v, 0).map(|c| Scalar::from_constant(&c))).collect::<Option<Vec<_>>>();
                    let Some(args) = constants else { continue };
                    let mut local = fuel.min(PARTIAL_EVALUATION_FUEL);
                    let before = local;
                    let result = evaluate_call(program, &operators, callee, &args, &mut local, 0);
                    fuel -= before-local;
                    if let Some(c) = result.and_then(|v| v.constant()).filter(|c| constant_fits(c, ty)) { plans.push((f.id, b, id, c)); }
                }
            }
        }
        let changed = !plans.is_empty();
        for (f, b, id, c) in plans {
            let function = program.function_mut(f).unwrap();
            if matches!(function.nodes[id.0 as usize].kind, Inst::TailCallDirect { .. }) {
                let ty = function.signature.result.clone();
                let source = function.nodes[id.0 as usize].source;
                let (constant, values) = InstFactory::new(function).create(Inst::Const(c), &[ty], source);
                let block = &mut function.blocks[b].insts;
                let pos = block.iter().position(|&other| other == id).unwrap();
                block.insert(pos, constant);
                function.nodes[id.0 as usize].kind = Inst::Return(values[0]);
            } else {
                function.nodes[id.0 as usize].kind = Inst::Const(c);
            }
            repair_effect_chain(function, b);
        }
        changed
    }
}

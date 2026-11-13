use super::*;

pub(super) fn type_functions(ty: &Type, roots: &mut HashSet<FuncId>) {
    match ty {
        Type::Closure(Some(id)) => { roots.insert(*id); }
        Type::List(Some(ty)) | Type::Set(Some(ty)) | Type::Cell(ty) | Type::Builtin(_, Some(ty)) => type_functions(ty, roots),
        Type::GenericMap(k, v) => { type_functions(k, roots); type_functions(v, roots); }
        Type::Interface(fields) | Type::Record(fields) => for f in fields { type_functions(&f.ty, roots); },
        Type::Union(types) => for ty in types { type_functions(ty, roots); },
        Type::GenericInstance { args, fields, .. } => {
            for ty in args { type_functions(ty, roots); }
            for f in fields { type_functions(&f.ty, roots); }
        }
        _ => {}
    }
}

/// Closed-program reachability. Module initializers, exports, escaped closures,
/// class method bindings, worker entrypoints, deopt frames and builtin helper
/// modules remain roots. Module IDs/slots and function IDs are never renumbered.
#[derive(Default)]
pub struct TreeShaking;
impl ProgramOptimization for TreeShaking {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        if !program.functions.iter().any(|f| f.kind == FunctionKind::ModuleBody) { return false; }
        let mut changed = false;
        // Module-variable reads never throw (`Inst::may_throw`); an unused one
        // is only an ordering token. Converting a closure call into a direct
        // call typically leaves the closure's load behind, and removing it is
        // what lets the binding below become unobserved.
        for f in &mut program.functions {
            let used: HashSet<ValueId> = f.blocks.iter().flat_map(|b| &b.insts)
                .flat_map(|id| f.nodes[id.0 as usize].kind.operands())
                .chain(f.deopts.iter().flat_map(|p| p.snapshot.slots.iter()
                    .chain(p.snapshot.inlined_frames.iter().flat_map(|frame| &frame.slots)).map(|s| s.value)))
                .collect();
            for b in 0..f.blocks.len() {
                let before = f.blocks[b].insts.len();
                f.blocks[b].insts.retain(|id| {
                    let node = &f.nodes[id.0 as usize];
                    !matches!(node.kind, Inst::LoadModuleVar { .. }) || node.results.iter().any(|v| used.contains(v))
                });
                if f.blocks[b].insts.len() != before { repair_effect_chain(f, b); changed = true; }
            }
        }
        let concurrent = program.functions.iter().any(|f| f.is_concurrent);
        if !concurrent {
            let mut observed = HashSet::default();
            for module in &program.modules {
                for export in &module.exports { observed.insert((module.id, export.slot)); }
            }
            for f in &program.functions {
                for &id in f.blocks.iter().flat_map(|b| &b.insts) {
                    if let Inst::LoadModuleVar { module, slot } = f.nodes[id.0 as usize].kind { observed.insert((module, slot)); }
                }
            }
            for f in &mut program.functions {
                let mut remove = HashSet::default();
                for &id in f.blocks.iter().flat_map(|b| &b.insts) {
                    let Inst::StoreModuleVar { module, slot, value } = f.nodes[id.0 as usize].kind else { continue };
                    if observed.contains(&(module, slot)) { continue; }
                    let ValueDef::InstResult(producer, _) = f.values[value.0 as usize].def else { continue };
                    // Only discard closure publication. Other initializers may
                    // be externally observable even when the binding is unread.
                    if matches!(f.nodes[producer.0 as usize].kind, Inst::MakeClosure { .. }) { remove.insert(id); }
                }
                for b in 0..f.blocks.len() {
                    f.blocks[b].insts.retain(|id| !remove.contains(id));
                    repair_effect_chain(f, b);
                }
                changed |= !remove.is_empty();
                let mut used: HashSet<_> = f.blocks.iter().flat_map(|b| &b.insts)
                    .flat_map(|id| f.nodes[id.0 as usize].kind.operands()).collect();
                for point in &f.deopts {
                    used.extend(point.snapshot.slots.iter().map(|s| s.value));
                    for frame in &point.snapshot.inlined_frames { used.extend(frame.slots.iter().map(|s| s.value)); }
                }
                for b in 0..f.blocks.len() {
                    let old = f.blocks[b].insts.len();
                    f.blocks[b].insts.retain(|id| {
                        let node = &f.nodes[id.0 as usize];
                        !matches!(node.kind, Inst::MakeClosure { .. }) || node.results.iter().any(|v| used.contains(v))
                    });
                    changed |= old != f.blocks[b].insts.len();
                    repair_effect_chain(f, b);
                }
            }
        }
        let mut live = HashSet::default();
        for f in &program.functions {
            if f.kind == FunctionKind::ModuleBody
                || program.module(f.module).and_then(|m| program.string(m.name)).is_some_and(|s| s.starts_with("<core:")) {
                live.insert(f.id);
            }
        }
        if concurrent {
            for module in &program.modules { for binding in &module.worker_function_bindings { live.insert(binding.function); } }
        }
        for class in &program.classes {
            for ty in class.own_instance_field_types.iter().chain(&class.static_field_types).chain(&class.implements) { type_functions(ty, &mut live); }
        }
        for layout in &program.structural_layouts { type_functions(&layout.contract, &mut live); }
        for (_, ty) in &program.contract_names { type_functions(ty, &mut live); }
        for declaration in &program.contract_declarations { type_functions(&declaration.contract, &mut live); }
        loop {
            let old = live.len();
            for f in &program.functions {
                if !live.contains(&f.id) { continue; }
                type_functions(&f.signature.result, &mut live);
                for ty in &f.signature.params { type_functions(ty, &mut live); }
                for capture in &f.captures { type_functions(&capture.ty, &mut live); }
                for &id in f.blocks.iter().flat_map(|b| &b.insts) {
                    let node = &f.nodes[id.0 as usize];
                    match &node.kind {
                        Inst::CallDirect { callee, .. } | Inst::TailCallDirect { callee, .. } => { live.insert(*callee); }
                        Inst::MakeClosure { function, .. } | Inst::BindMethod { function, .. } => { live.insert(*function); }
                        Inst::BindInterface { expected, .. } | Inst::GuardType { expected, .. } => type_functions(expected, &mut live),
                        _ => {}
                    }
                    for v in node.results.iter().copied().chain(node.kind.operands()) { type_functions(&f.values[v.0 as usize].ty, &mut live); }
                }
                for point in &f.deopts {
                    for frame in &point.snapshot.inlined_frames { live.insert(frame.function); }
                }
            }
            if live.len() == old { break; }
        }
        if program.functions.iter().any(|f| !live.contains(&f.id)) {
            program.retain_functions(|f| live.contains(&f.id));
            for module in &mut program.modules { module.worker_function_bindings.retain(|b| live.contains(&b.function)); }
            changed = true;
        }
        changed
    }
}

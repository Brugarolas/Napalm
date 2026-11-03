//! Whole-program sharing of immutable constant map literals.
//!
//! A literal such as `{ left: null, right: null }` normally allocates a fresh
//! mutable Map on every evaluation. When the closed program can prove that no
//! object produced by that site is ever mutated, stored outside analyzable SSA
//! flow, passed to unknown code or compared by identity, every evaluation is
//! observationally equivalent to returning one shared instance. The site then
//! reads a hidden module slot initialized at the very start of its module body.
//!
//! The analysis is a flow-insensitive, interprocedural "may hold" propagation.
//! Each candidate site owns one bit. A value tracks the bits it may *be* and the
//! bits it may *reach* through fields of fresh literals (containers). Both flow
//! through moves, block parameters and `CallDirect` arguments/returns; field
//! reads yield only the receiver's reachable bits, so reading a constant field
//! of a shared literal does not taint the scalar result. Any use outside a small
//! read-only whitelist disqualifies every bit the operand may be or reach.
//! Functions whose results carry bits must only be reachable through direct
//! calls; otherwise an unanalyzed caller could observe the shared object.
//! The same proof is used by automatic memoization for maps with computed fields.

use crate::napalm_collections::{HashMap, HashSet};

use crate::napalm_ir::{
    CallConv, Constant, FuncId, Function, FunctionKind, Inst, InstId, MethodKind, ModuleId,
    ModuleSlot, Program, Type, ValueDef, ValueId,
};
use crate::napalm_ir_optimizer::{repair_effect_chain, InstFactory, ProgramOptimization};

use super::discover_module_function_bindings;

/// Bit-per-candidate analysis width.
const MAX_CANDIDATES: usize = 64;

#[derive(Default)]
pub struct SharedConstantLiterals;

impl ProgramOptimization for SharedConstantLiterals {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        share_constant_literals(program)
    }
}

struct Candidate {
    function: usize,
    block: usize,
    alloc: InstId,
    map: ValueId,
    init_stores: Vec<InstId>,
    fields: Vec<(Constant, Type, Constant, Type)>,
}

struct Analysis<'p> {
    program: &'p Program,
    func_index: HashMap<FuncId, usize>,
    /// Initializing `MapSet`s of every fresh map literal, keyed by function.
    init_stores: Vec<HashSet<InstId>>,
    candidate_bits: HashMap<(usize, InstId), u64>,
    taint: Vec<Vec<Taint>>,
    returns: Vec<Taint>,
}

/// Candidate bits a value may *be* (`holds`) or may lead to through the
/// fields of fresh literals (`reach`). Field reads inherit only `reach`, never
/// the identity of their receiver. Nested reads conservatively retain all
/// reachable bits, which can reject safe code but cannot hide an escaping alias.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Taint {
    holds: u64,
    reach: u64,
}

impl Taint {
    fn all(self) -> u64 {
        self.holds | self.reach
    }

    fn fields(self) -> Self {
        Self { holds: self.reach, reach: self.reach }
    }

    fn union(self, other: Self) -> Self {
        Self { holds: self.holds | other.holds, reach: self.reach | other.reach }
    }
}

pub fn share_constant_literals(program: &mut Program) -> bool {
    if !program_is_eligible(program) {
        return false;
    }

    let module_bodies = unique_module_bodies(program);
    let mut init_stores = vec![HashSet::default(); program.functions.len()];
    let mut candidates = Vec::new();
    for (fi, function) in program.functions.iter().enumerate() {
        for (block_ix, block) in function.blocks.iter().enumerate() {
            for (pos, &id) in block.insts.iter().enumerate() {
                let node = &function.nodes[id.0 as usize];
                if !matches!(node.kind, Inst::AllocMap) {
                    continue;
                }
                let Some(&map) = node.results.first() else { continue; };
                let stores = initializing_stores(function, &block.insts[pos + 1..], map);
                init_stores[fi].extend(stores.iter().copied());
                if candidates.len() >= MAX_CANDIDATES
                    || !module_bodies.contains_key(&function.module)
                {
                    continue;
                }
                if let Some(fields) = constant_fields(function, &stores) {
                    candidates.push(Candidate {
                        function: fi,
                        block: block_ix,
                        alloc: id,
                        map,
                        init_stores: stores,
                        fields,
                    });
                }
            }
        }
    }
    if candidates.is_empty() {
        return false;
    }

    let shareable = {
        let sites = candidates.iter().map(|candidate| (candidate.function, candidate.alloc)).collect::<Vec<_>>();
        let mut analysis = Analysis::new(program, init_stores, &sites);
        analysis.propagate();
        let rejected = analysis.rejected_bits();
        candidates
            .iter()
            .enumerate()
            .map(|(bit, _)| rejected & (1u64 << bit) == 0)
            .collect::<Vec<_>>()
    };

    let mut changed = false;
    for (candidate, share) in candidates.iter().zip(shareable) {
        if share {
            let body = module_bodies[&program.functions[candidate.function].module];
            hoist_candidate(program, candidate, body);
            changed = true;
        }
    }
    changed
}

/// Allocation and initialization instructions whose object identities are never
/// observed or mutated anywhere in the closed program. Unlike literal hoisting,
/// memoization can share maps with computed fields, including recursive children.
pub(super) fn memoizable_map_operations(program: &Program) -> HashSet<(FuncId, InstId)> {
    let mut allowed = HashSet::default();
    if !program_is_eligible(program) { return allowed; }
    let mut sites = Vec::new();
    let mut stores_by_site = Vec::new();
    let mut init_stores = vec![HashSet::default(); program.functions.len()];
    for (fi, function) in program.functions.iter().enumerate() {
        for block in &function.blocks {
            for (pos, &id) in block.insts.iter().enumerate() {
                let node = &function.nodes[id.0 as usize];
                if !matches!(node.kind, Inst::AllocMap) { continue; }
                let Some(&map) = node.results.first() else { continue; };
                let stores = initializing_stores(function, &block.insts[pos + 1..], map);
                init_stores[fi].extend(stores.iter().copied());
                if sites.len() == MAX_CANDIDATES { continue; }
                sites.push((fi, id));
                stores_by_site.push(stores);
            }
        }
    }
    if sites.is_empty() { return allowed; }
    let mut analysis = Analysis::new(program, init_stores, &sites);
    analysis.propagate();
    let rejected = analysis.rejected_bits();
    for (bit, (&(fi, alloc), stores)) in sites.iter().zip(&stores_by_site).enumerate() {
        if rejected & (1u64 << bit) != 0 { continue; }
        let function = &program.functions[fi];
        // Computed keys could invoke user-defined hashing/equality. Only literal
        // string keys belong to the allocating-pure subset.
        if !stores.iter().all(|&id| matches!(
            &function.nodes[id.0 as usize].kind,
            Inst::MapSet { key, .. }
                if matches!(resolve_constant(function, *key), Some((Constant::String(_), _)))
        )) { continue; }
        allowed.insert((function.id, alloc));
        allowed.extend(stores.iter().map(|&id| (function.id, id)));
    }
    allowed
}

/// Hidden module slots are initialized by the module body, which isolated
/// workers never run. Workers only exist when some function is concurrent (the
/// VM's `concurrency_capable` check), so such programs are rejected; the
/// worker-callable bindings every synchronous top-level closure receives are
/// then unreachable. A user method bound to a builtin class could also change
/// what a read-only-looking getter does on a Map.
fn program_is_eligible(program: &Program) -> bool {
    for function in &program.functions {
        if function.is_concurrent {
            return false;
        }
        for block in &function.blocks {
            for &id in &block.insts {
                match &function.nodes[id.0 as usize].kind {
                    Inst::Launch { .. } => return false,
                    Inst::BindMethod { class, .. } | Inst::BindForeignMethod { class, .. } => {
                        if !is_created_class(function, *class) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    true
}

fn is_created_class(function: &Function, mut value: ValueId) -> bool {
    for _ in 0..16 {
        match producer(function, value) {
            Some(Inst::CreateClass { .. }) => return true,
            Some(Inst::Move(source)) => value = *source,
            _ => return false,
        }
    }
    false
}

fn unique_module_bodies(program: &Program) -> HashMap<ModuleId, usize> {
    let mut bodies: HashMap<ModuleId, usize> = HashMap::default();
    let mut duplicated: HashSet<ModuleId> = HashSet::default();
    for (fi, function) in program.functions.iter().enumerate() {
        if function.kind == FunctionKind::ModuleBody
            && bodies.insert(function.module, fi).is_some()
        {
            duplicated.insert(function.module);
        }
    }
    for module in duplicated {
        bodies.remove(&module);
    }
    bodies
}

fn producer(function: &Function, value: ValueId) -> Option<&Inst> {
    let ValueDef::InstResult(inst, 0) = function.value(value)?.def else {
        return None;
    };
    Some(&function.node(inst)?.kind)
}

fn resolve_constant(function: &Function, mut value: ValueId) -> Option<(Constant, Type)> {
    for _ in 0..16 {
        match producer(function, value)? {
            Inst::Const(constant) => {
                return Some((constant.clone(), function.value(value)?.ty.clone()));
            }
            Inst::Move(source) => value = *source,
            _ => return None,
        }
    }
    None
}

/// `MapSet`s that fill a fresh literal before the map is observable elsewhere.
fn initializing_stores(function: &Function, rest: &[InstId], map: ValueId) -> Vec<InstId> {
    let mut stores = Vec::new();
    for &id in rest {
        let kind = &function.nodes[id.0 as usize].kind;
        if kind.is_terminator() {
            break;
        }
        if !kind.operands().contains(&map) {
            continue;
        }
        match kind {
            Inst::MapSet { map: target, key, value }
                if *target == map && *key != map && *value != map =>
            {
                stores.push(id);
            }
            _ => break,
        }
    }
    stores
}

fn constant_fields(
    function: &Function,
    stores: &[InstId],
) -> Option<Vec<(Constant, Type, Constant, Type)>> {
    let mut fields = Vec::with_capacity(stores.len());
    for &id in stores {
        let Inst::MapSet { key, value, .. } = &function.nodes[id.0 as usize].kind else {
            return None;
        };
        let (key, key_ty) = resolve_constant(function, *key)?;
        if !matches!(key, Constant::String(_)) {
            return None;
        }
        let (value, value_ty) = resolve_constant(function, *value)?;
        fields.push((key, key_ty, value, value_ty));
    }
    Some(fields)
}

impl<'p> Analysis<'p> {
    fn new(program: &'p Program, init_stores: Vec<HashSet<InstId>>, sites: &[(usize, InstId)]) -> Self {
        let func_index = program
            .functions
            .iter()
            .enumerate()
            .map(|(index, function)| (function.id, index))
            .collect();
        let candidate_bits = sites
            .iter()
            .enumerate()
            .map(|(bit, &site)| (site, 1u64 << bit))
            .collect();
        let taint = program
            .functions
            .iter()
            .map(|function| vec![Taint::default(); function.values.len()])
            .collect();
        Self {
            program,
            func_index,
            init_stores,
            candidate_bits,
            taint,
            returns: vec![Taint::default(); program.functions.len()],
        }
    }

    fn get(&self, fi: usize, value: ValueId) -> Taint {
        self.taint[fi].get(value.0 as usize).copied().unwrap_or_default()
    }

    fn add(&mut self, fi: usize, value: ValueId, taint: Taint) -> bool {
        let Some(slot) = self.taint[fi].get_mut(value.0 as usize) else { return false; };
        add_taint(slot, taint)
    }

    /// Returns the callee index when a direct call is fully analyzable.
    fn direct_callee(&self, conv: CallConv, callee: FuncId, argc: usize) -> Option<usize> {
        let index = *self.func_index.get(&callee)?;
        let target = &self.program.functions[index];
        let entry = target.block(target.entry)?;
        (conv == CallConv::Napalm
            && !target.is_async
            && !target.is_concurrent
            && !matches!(target.kind, FunctionKind::PrimitiveStub | FunctionKind::ForeignStub)
            && entry.params.len() == argc)
            .then_some(index)
    }

    fn pass_arguments(&mut self, fi: usize, callee: usize, args: &[ValueId]) -> bool {
        let target = &self.program.functions[callee];
        let Some(entry) = target.block(target.entry) else { return false; };
        let params = entry.params.clone();
        let mut changed = false;
        for (param, &arg) in params.into_iter().zip(args) {
            let taint = self.get(fi, arg);
            changed |= self.add(callee, param, taint);
        }
        changed
    }

    fn read_only_getter(&self, method: crate::napalm_ir::MethodSymbol) -> bool {
        let Some(signature) = self.program.method(method) else { return false; };
        signature.kind == MethodKind::Getter
            && signature.arity == 0
            && !matches!(self.program.string(signature.name), Some("keys" | "toString") | None)
    }

    fn is_equality(&self, method: crate::napalm_ir::MethodSymbol) -> bool {
        let Some(signature) = self.program.method(method) else { return false; };
        signature.kind == MethodKind::Method
            && signature.arity == 1
            && matches!(self.program.string(signature.name), Some("==" | "!="))
    }

    fn propagate(&mut self) {
        let program = self.program;
        loop {
            let mut changed = false;
            for (fi, function) in program.functions.iter().enumerate() {
                for block in &function.blocks {
                    for &id in &block.insts {
                        let node = &function.nodes[id.0 as usize];
                        let result = node.results.first().copied();
                        match &node.kind {
                            Inst::AllocMap => {
                                if let (Some(&bit), Some(result)) =
                                    (self.candidate_bits.get(&(fi, id)), result)
                                {
                                    changed |= self.add(fi, result, Taint { holds: bit, reach: 0 });
                                }
                            }
                            Inst::MapSet { map, value, .. } if self.init_stores[fi].contains(&id) => {
                                let reach = self.get(fi, *value).all();
                                changed |= self.add(fi, *map, Taint { holds: 0, reach });
                            }
                            Inst::MapGet { map: source, .. } | Inst::LoadProperty { object: source, .. } => {
                                let taint = self.get(fi, *source).fields();
                                if let Some(result) = result {
                                    changed |= self.add(fi, result, taint);
                                }
                            }
                            Inst::Move(source)
                            | Inst::GuardType { value: source, .. }
                            | Inst::GuardClass { value: source, .. }
                            | Inst::GuardNotNull { value: source, .. } => {
                                let taint = self.get(fi, *source);
                                if let Some(result) = result {
                                    changed |= self.add(fi, result, taint);
                                }
                            }
                            Inst::Invoke { receiver, method, args, .. }
                                if args.is_empty() && self.read_only_getter(*method) =>
                            {
                                let taint = self.get(fi, *receiver).fields();
                                if let Some(result) = result {
                                    changed |= self.add(fi, result, taint);
                                }
                            }
                            Inst::CallDirect { conv, callee, args } => {
                                if let Some(target) = self.direct_callee(*conv, *callee, args.len()) {
                                    changed |= self.pass_arguments(fi, target, args);
                                    let taint = self.returns[target];
                                    if let Some(result) = result {
                                        changed |= self.add(fi, result, taint);
                                    }
                                }
                            }
                            Inst::TailCallDirect { conv, callee, args } => {
                                if let Some(target) = self.direct_callee(*conv, *callee, args.len()) {
                                    changed |= self.pass_arguments(fi, target, args);
                                    let taint = self.returns[target];
                                    changed |= add_taint(&mut self.returns[fi], taint);
                                }
                            }
                            Inst::Return(value) => {
                                let taint = self.get(fi, *value);
                                changed |= add_taint(&mut self.returns[fi], taint);
                            }
                            kind => {
                                for edge in kind.edges() {
                                    let Some(target) = function.block(edge.target) else { continue; };
                                    for (&param, &arg) in target.params.iter().zip(&edge.args) {
                                        let taint = self.get(fi, arg);
                                        changed |= self.add(fi, param, taint);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn rejected_bits(&self) -> u64 {
        let program = self.program;
        let mut rejected = 0u64;
        let mut use_counts: Vec<Vec<u32>> = program
            .functions
            .iter()
            .map(|function| vec![0; function.values.len()])
            .collect();

        for (fi, function) in program.functions.iter().enumerate() {
            for block in &function.blocks {
                for &id in &block.insts {
                    let kind = &function.nodes[id.0 as usize].kind;
                    let operands = kind.operands();
                    for &operand in &operands {
                        if let Some(count) = use_counts[fi].get_mut(operand.0 as usize) {
                            *count += 1;
                        }
                    }
                    let bits = |value: ValueId| self.get(fi, value).all();
                    let constant = |value: ValueId| resolve_constant(function, value).is_some();
                    match kind {
                        Inst::Move(_) | Inst::Jump { .. } | Inst::Branch { .. } | Inst::Switch { .. } => {}
                        Inst::Return(value) => {
                            if function.kind == FunctionKind::ModuleBody
                                || function.is_async
                                || function.is_concurrent
                            {
                                rejected |= bits(*value);
                            }
                        }
                        Inst::CallDirect { conv, callee, args }
                        | Inst::TailCallDirect { conv, callee, args } => {
                            if self.direct_callee(*conv, *callee, args.len()).is_none() {
                                rejected |= args.iter().fold(0, |acc, &arg| acc | bits(arg));
                            }
                            if matches!(kind, Inst::TailCallDirect { .. })
                                && (function.kind == FunctionKind::ModuleBody || function.is_async)
                            {
                                if let Some(&target) = self.func_index.get(callee) {
                                    rejected |= self.returns[target].all();
                                }
                            }
                        }
                        Inst::Invoke { receiver, method, args, .. } => {
                            let allowed = if args.is_empty() && self.read_only_getter(*method) {
                                true
                            } else if args.len() == 1 && self.is_equality(*method) {
                                constant(*receiver) || constant(args[0])
                            } else {
                                false
                            };
                            if !allowed {
                                rejected |= operands.iter().fold(0, |acc, &value| acc | bits(value));
                            }
                        }
                        Inst::RefEq { left, right } => {
                            if !constant(*left) && !constant(*right) {
                                rejected |= bits(*left) | bits(*right);
                            }
                        }
                        Inst::MapGet { key, .. } | Inst::LoadProperty { key, .. } => {
                            rejected |= bits(*key);
                        }
                        Inst::MapSet { key, .. } if self.init_stores[fi].contains(&id) => {
                            rejected |= bits(*key);
                        }
                        Inst::MapLen { .. }
                        | Inst::GuardType { .. }
                        | Inst::GuardClass { .. }
                        | Inst::GuardNotNull { .. } => {}
                        Inst::IsInstance { class, .. } => rejected |= bits(*class),
                        _ => {
                            rejected |= operands.iter().fold(0, |acc, &value| acc | bits(value));
                        }
                    }
                }
            }
        }

        let module_functions = discover_module_function_bindings(program);
        for (fi, returned) in self.returns.iter().enumerate() {
            let bits = returned.all();
            if bits != 0 && !self.only_called_directly(fi, &module_functions, &use_counts) {
                rejected |= bits;
            }
        }
        rejected
    }

    /// A function returning shared candidates must not be callable through a
    /// first-class reference, method dispatch, imports or worker bindings.
    fn only_called_directly(
        &self,
        fi: usize,
        module_functions: &HashMap<(ModuleId, ModuleSlot), FuncId>,
        use_counts: &[Vec<u32>],
    ) -> bool {
        let program = self.program;
        let target = &program.functions[fi];
        if target.kind != FunctionKind::Closure
            || target.is_async
            || target.is_concurrent
            || !target.captures.is_empty()
        {
            return false;
        }
        let slots: HashSet<ModuleSlot> = module_functions
            .iter()
            .filter(|&(&(module, _), &function)| module == target.module && function == target.id)
            .map(|(&(_, slot), _)| slot)
            .collect();
        if slots.is_empty() {
            return false;
        }
        let Some(module) = program.module(target.module) else { return false; };
        if module.exports.iter().any(|export| slots.contains(&export.slot)) {
            // Exported callables can also be reached by an embedding host.
            return false;
        }

        for (gi, function) in program.functions.iter().enumerate() {
            for block in &function.blocks {
                for &id in &block.insts {
                    let node = &function.nodes[id.0 as usize];
                    match &node.kind {
                        Inst::MakeClosure { function: made, .. } if *made == target.id => {
                            if function.kind != FunctionKind::ModuleBody
                                || function.module != target.module
                            {
                                return false;
                            }
                            let Some(&closure) = node.results.first() else { return false; };
                            if !closure_only_stored_to(function, closure, target.module, &slots) {
                                return false;
                            }
                        }
                        Inst::LoadModuleVar { module, slot }
                            if *module == target.module && slots.contains(slot) =>
                        {
                            let Some(&result) = node.results.first() else { continue; };
                            if use_counts[gi].get(result.0 as usize).copied().unwrap_or(0) != 0 {
                                return false;
                            }
                        }
                        Inst::BindMethod { function: bound, .. } if *bound == target.id => {
                            return false;
                        }
                        _ => {}
                    }
                }
            }
        }
        true
    }
}

fn add_taint(slot: &mut Taint, taint: Taint) -> bool {
    let next = slot.union(taint);
    if next == *slot {
        return false;
    }
    *slot = next;
    true
}

fn closure_only_stored_to(
    function: &Function,
    closure: ValueId,
    module: ModuleId,
    slots: &HashSet<ModuleSlot>,
) -> bool {
    for block in &function.blocks {
        for &id in &block.insts {
            let kind = &function.nodes[id.0 as usize].kind;
            if !kind.operands().contains(&closure) {
                continue;
            }
            match kind {
                Inst::StoreModuleVar { module: stored, slot, value }
                    if *value == closure && *stored == module && slots.contains(slot) => {}
                _ => return false,
            }
        }
    }
    true
}

fn hoist_candidate(program: &mut Program, candidate: &Candidate, body: usize) {
    let module_id = program.functions[candidate.function].module;
    let module_index = program
        .modules
        .iter()
        .position(|module| module.id == module_id)
        .expect("candidate module has metadata");
    let slot = program.modules[module_index].add_hidden_slot();

    let (map_ty, source) = {
        let function = &program.functions[candidate.function];
        let ty = function.value(candidate.map).map(|value| value.ty.clone()).unwrap_or(Type::Top);
        (ty, function.nodes[candidate.alloc.0 as usize].source.clone())
    };

    {
        let function = &mut program.functions[candidate.function];
        function.nodes[candidate.alloc.0 as usize].kind = Inst::LoadModuleVar { module: module_id, slot };
        let removed: HashSet<InstId> = candidate.init_stores.iter().copied().collect();
        function.blocks[candidate.block].insts.retain(|id| !removed.contains(id));
        repair_effect_chain(function, candidate.block);
    }

    let function = &mut program.functions[body];
    let mut prologue = Vec::with_capacity(candidate.fields.len() * 3 + 2);
    let mut factory = InstFactory::new(function);
    let (alloc, results) = factory.create(Inst::AllocMap, &[map_ty], source.clone());
    let map = results[0];
    let mut stores = Vec::with_capacity(candidate.fields.len());
    for (key, key_ty, value, value_ty) in &candidate.fields {
        let (key_inst, key_value) = factory.create(Inst::Const(key.clone()), &[key_ty.clone()], source.clone());
        let (value_inst, value_value) =
            factory.create(Inst::Const(value.clone()), &[value_ty.clone()], source.clone());
        prologue.push(key_inst);
        prologue.push(value_inst);
        let (store, _) = factory.create(
            Inst::MapSet { map, key: key_value[0], value: value_value[0] },
            &[],
            source.clone(),
        );
        stores.push(store);
    }
    prologue.push(alloc);
    prologue.extend(stores);
    let (publish, _) = factory.create(
        Inst::StoreModuleVar { module: module_id, slot, value: map },
        &[],
        source,
    );
    prologue.push(publish);

    let entry = function
        .blocks
        .iter()
        .position(|block| block.id == function.entry)
        .expect("module body has an entry block");
    function.blocks[entry].insts.splice(0..0, prologue);
    repair_effect_chain(function, entry);
}

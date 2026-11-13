//! Whole-program exploitation of static types.
//!
//! Napalm types are contracts, not hints: parameters, typed bindings, typed
//! fields, typed lists, structural objects and function results are validated
//! where values enter them, and a contract can never be changed afterwards.
//! Every `Value::ty` in WIR is therefore a guarantee. This pass turns those
//! guarantees into code that needs neither guards nor deoptimization:
//!
//! 1. **Class-hierarchy devirtualization.** `Invoke` on a receiver whose static
//!    type is `Instance<C>` is resolved against the complete, statically known
//!    method tables of `C` and every class that may extend it. A unique bytecode
//!    target becomes `CallDirect`; a trivial synthesized property accessor is
//!    replaced by the `LoadField`/`StoreField` it performs. `super` calls and
//!    static calls on exact class values are resolved the same way.
//! 2. **Result types.** Declared result contracts and results inferred from the
//!    returned values type every statically resolved call, so callers keep
//!    numeric/object facts across call boundaries.
//! 3. **Builtin and numeric result types.** Stock `Num` operators over proven
//!    numbers, typed list subscripts (`T | null`) and the count getters each
//!    builtin class registers get precise results.
//! 4. **Validation elimination.** A scalar `BindInterface` whose operand is
//!    already proven to satisfy the contract is a no-op and is removed, as is a
//!    heap-contract binding of a value just bound to the identical contract.
//! 5. **Allocation-enabling inlining.** A small straight-line direct callee that
//!    receives a caller-allocated object is copied into the caller so escape
//!    analysis can scalar-replace the object (see `inline_allocation_calls`).
//!
//! The analysis is deliberately conservative. Any class whose superclass cannot
//! be identified statically, any `BindMethod` whose class cannot be traced, and
//! any foreign/conflicting binding disables devirtualization of the affected
//! method selector instead of being guessed.

use crate::napalm_collections::{HashMap, HashSet};
use crate::napalm_ir::{
    BuiltinId, CallConv, ClassId, ClassSet, Constant, FieldSlot, FuncId, Function, FunctionKind, Inst,
    InstId, MethodKind, MethodSymbol, ModuleId, ModuleSlot, Program, StructuralLayoutId, Type,
    ValueDef, ValueId,
};
use crate::napalm_ir_optimizer::{repair_effect_chain, FunctionOptimization, InstFactory, ProgramOptimization};

use super::{rewrite_known_function_calls, InterpreterOptimizationBundle};

/// Fixed-point bound. Each round can expose new facts (a devirtualized call's
/// typed result may devirtualize the next call in the chain); real programs
/// converge in two or three rounds.
const MAX_ROUNDS: usize = 6;
/// Maximum depth when tracing an SSA value back to its defining class.
const TRACE_DEPTH: u8 = 16;

#[derive(Default)]
pub struct StaticTypeSpecialization;

impl ProgramOptimization for StaticTypeSpecialization {
    fn optimize_program(&mut self, program: &mut Program) -> bool {
        let mut changed_any = false;
        let selectors = SelectorTable::build(program);
        for _ in 0..MAX_ROUNDS {
            // Closure calls through immutable module bindings become CallDirect
            // first, so their results can be typed in this same round.
            changed_any |= rewrite_known_function_calls(program);
            let hierarchy = ClassHierarchy::build(program);
            let accessors = classify_accessors(program);
            let returns = infer_result_types(program);

            let mut touched = Vec::new();
            for index in 0..program.functions.len() {
                let plan = plan_function(
                    program,
                    &program.functions[index],
                    &hierarchy,
                    &accessors,
                    &returns,
                    &selectors,
                );
                if plan.is_empty() {
                    continue;
                }
                apply_plan(&mut program.functions[index], plan);
                touched.push(index);
            }
            // Direct calls that receive a caller-allocated object are inlined so
            // the re-run below can scalar-replace the object.
            for index in inline_allocation_calls(program) {
                if !touched.contains(&index) {
                    touched.push(index);
                }
            }

            if touched.is_empty() {
                break;
            }
            changed_any = true;
            // Re-run the ordinary function pipeline on rewritten functions so the
            // new facts flow through block parameters (TypedObjectSpecialization),
            // inlined field accesses get forwarded/CSE'd and dead code disappears.
            let mut bundle = InterpreterOptimizationBundle::default();
            for index in touched {
                bundle.optimize_function(&mut program.functions[index]);
            }
        }
        changed_any
    }
}

// =============================================================================
// Method selector metadata
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericOp {
    /// `+ - * / % & | ^ << >>` over two numbers.
    Arithmetic,
    /// `< <= > >=` over two numbers.
    Ordering,
    /// `== !=`: a Num receiver compares against any operand and yields Bool.
    Equality,
    /// Unary `-` / `~`.
    Unary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinSelector {
    /// `[]` subscript getter with one index.
    Subscript,
    /// `length` getter (and the `length()` List method): List and String only.
    Length,
    /// `count` getter: List, String, Map-backed values and Set.
    Count,
    /// `size` getter: Map-backed values and Set.
    Size,
}

/// Precomputed classification of the program's interned method selectors.
#[derive(Default)]
struct SelectorTable {
    numeric: HashMap<MethodSymbol, NumericOp>,
    builtin: HashMap<MethodSymbol, BuiltinSelector>,
}

impl SelectorTable {
    fn build(program: &Program) -> Self {
        let mut table = Self::default();
        let mut index = 0u32;
        while let Some(signature) = program.method(MethodSymbol(index)) {
            let symbol = MethodSymbol(index);
            index += 1;
            let Some(name) = program.string(signature.name) else { continue };
            let numeric = match (signature.kind, signature.arity, name) {
                (MethodKind::Method, 1, "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>") => {
                    Some(NumericOp::Arithmetic)
                }
                (MethodKind::Method, 1, "<" | "<=" | ">" | ">=") => Some(NumericOp::Ordering),
                (MethodKind::Method, 1, "==" | "!=") => Some(NumericOp::Equality),
                (MethodKind::Method, 0, "-" | "~") => Some(NumericOp::Unary),
                _ => None,
            };
            if let Some(op) = numeric {
                table.numeric.insert(symbol, op);
            }
            let builtin = match (signature.kind, signature.arity, name) {
                (MethodKind::SubscriptGetter, 1, "[]") => Some(BuiltinSelector::Subscript),
                (MethodKind::Getter, 0, "length") => Some(BuiltinSelector::Length),
                (MethodKind::Getter, 0, "count") => Some(BuiltinSelector::Count),
                (MethodKind::Getter, 0, "size") => Some(BuiltinSelector::Size),
                (MethodKind::Method, 0, "length") => Some(BuiltinSelector::Length),
                _ => None,
            };
            if let Some(selector) = builtin {
                table.builtin.insert(symbol, selector);
            }
        }
        table
    }
}

// =============================================================================
// Class hierarchy
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Binding {
    Bytecode(FuncId),
    /// Foreign implementation or two different bytecode functions bound under
    /// the same selector: never devirtualized.
    Opaque,
}

#[derive(Default)]
struct ClassHierarchy {
    /// Verified static superclass of every created class (`None` = root).
    superclass: HashMap<ClassId, Option<ClassId>>,
    /// Classes whose runtime superclass cannot be identified statically.
    dynamic_super: HashSet<ClassId>,
    children: HashMap<ClassId, Vec<ClassId>>,
    instance_methods: HashMap<(ClassId, MethodSymbol), Binding>,
    static_methods: HashMap<(ClassId, MethodSymbol), Binding>,
    /// Selectors bound on a class value that could not be traced: any class
    /// might implement them, so they are never devirtualized.
    poisoned: HashSet<MethodSymbol>,
    /// Module slots that only ever hold one statically known class object.
    exact_class_slots: HashMap<(ModuleId, ModuleSlot), ClassId>,
}

impl ClassHierarchy {
    fn build(program: &Program) -> Self {
        let mut hierarchy = Self {
            exact_class_slots: discover_exact_class_slots(program),
            ..Self::default()
        };

        // `None` = the superclass could not be verified at some CreateClass.
        let mut seen_super: HashMap<ClassId, Option<Option<ClassId>>> = HashMap::default();
        for function in &program.functions {
            for block in &function.blocks {
                for &id in &block.insts {
                    let node = &function.nodes[id.0 as usize];
                    match &node.kind {
                        Inst::CreateClass { class, superclass } => {
                            let declared = program.class(*class).and_then(|info| info.superclass);
                            let verified = match superclass {
                                None => Some(None),
                                Some(value) => match hierarchy.trace_class(function, *value, 0) {
                                    Some(traced) if Some(traced) == declared => Some(Some(traced)),
                                    _ => None,
                                },
                            };
                            let entry = seen_super.entry(*class).or_insert(verified);
                            if *entry != verified {
                                *entry = None;
                            }
                        }
                        Inst::BindMethod { class, method, function: target, is_static } => {
                            match hierarchy.trace_class(function, *class, 0) {
                                Some(owner) => {
                                    hierarchy.bind(owner, *method, *is_static, Binding::Bytecode(*target))
                                }
                                None => {
                                    hierarchy.poisoned.insert(*method);
                                }
                            }
                        }
                        Inst::BindForeignMethod { class, method, is_static, .. } => {
                            match hierarchy.trace_class(function, *class, 0) {
                                Some(owner) => hierarchy.bind(owner, *method, *is_static, Binding::Opaque),
                                None => {
                                    hierarchy.poisoned.insert(*method);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        for (class, verified) in seen_super {
            match verified {
                Some(superclass) => {
                    hierarchy.superclass.insert(class, superclass);
                    if let Some(superclass) = superclass {
                        hierarchy.children.entry(superclass).or_default().push(class);
                    }
                }
                None => {
                    hierarchy.dynamic_super.insert(class);
                }
            }
        }
        hierarchy
    }

    fn bind(&mut self, owner: ClassId, method: MethodSymbol, is_static: bool, binding: Binding) {
        let table = if is_static { &mut self.static_methods } else { &mut self.instance_methods };
        let entry = table.entry((owner, method)).or_insert(binding);
        if *entry != binding {
            *entry = Binding::Opaque;
        }
    }

    /// Trace an SSA value to the exact class object it holds, if provable.
    fn trace_class(&self, function: &Function, value: ValueId, depth: u8) -> Option<ClassId> {
        if depth >= TRACE_DEPTH {
            return None;
        }
        let ValueDef::InstResult(inst, 0) = function.value(value)?.def else { return None };
        match &function.node(inst)?.kind {
            Inst::CreateClass { class, .. } => Some(*class),
            Inst::Move(source) | Inst::BindInterface { value: source, .. } => {
                self.trace_class(function, *source, depth + 1)
            }
            Inst::LoadModuleVar { module, slot } => self.exact_class_slots.get(&(*module, *slot)).copied(),
            _ => None,
        }
    }

    fn is_known(&self, class: ClassId) -> bool {
        self.superclass.contains_key(&class)
    }

    /// Runtime-equivalent instance method resolution (copy-down inheritance).
    fn resolve_instance(&self, class: ClassId, method: MethodSymbol) -> Option<FuncId> {
        let mut current = class;
        for _ in 0..=self.superclass.len() {
            if let Some(binding) = self.instance_methods.get(&(current, method)) {
                return match binding {
                    Binding::Bytecode(function) => Some(*function),
                    Binding::Opaque => None,
                };
            }
            match self.superclass.get(&current) {
                Some(Some(parent)) => current = *parent,
                // Root without a binding (missing method) or a class with an
                // unidentified superclass: the runtime result is not static.
                Some(None) | None => return None,
            }
        }
        None
    }

    /// `class` itself plus every class whose verified superclass chain reaches it.
    fn subtree(&self, class: ClassId) -> Vec<ClassId> {
        let mut out = vec![class];
        let mut index = 0;
        while index < out.len() {
            if let Some(children) = self.children.get(&out[index]) {
                for &child in children {
                    if !out.contains(&child) {
                        out.push(child);
                    }
                }
            }
            index += 1;
        }
        out
    }

    /// The single bytecode implementation every possible receiver of
    /// `Instance<set>` (classes in `set` or any subclass) dispatches to.
    fn unique_instance_target(&self, set: &ClassSet, method: MethodSymbol) -> Option<FuncId> {
        if set.is_unknown() || self.poisoned.contains(&method) {
            return None;
        }
        let mut target: Option<FuncId> = None;
        for &class in set.as_slice() {
            if !self.is_known(class) {
                return None;
            }
            for member in self.subtree(class) {
                let resolved = self.resolve_instance(member, method)?;
                match target {
                    None => target = Some(resolved),
                    Some(existing) if existing == resolved => {}
                    Some(_) => return None,
                }
            }
        }
        let target = target?;
        // A class with an unidentified superclass may extend any class in
        // `set`. Whatever it inherits is already covered above; only its own
        // (and its static descendants') bindings could introduce new targets.
        for &dynamic in &self.dynamic_super {
            for member in self.subtree(dynamic) {
                if let Some(binding) = self.instance_methods.get(&(member, method)) {
                    if *binding != Binding::Bytecode(target) {
                        return None;
                    }
                }
            }
        }
        Some(target)
    }

    /// `super.method(...)` lookup: static by definition, starts at the
    /// superclass of the lexical home class.
    fn super_target(&self, home: ClassId, method: MethodSymbol, is_static: bool) -> Option<FuncId> {
        if self.poisoned.contains(&method) {
            return None;
        }
        let Some(Some(parent)) = self.superclass.get(&home) else { return None };
        if is_static {
            match self.static_methods.get(&(*parent, method))? {
                Binding::Bytecode(function) => Some(*function),
                Binding::Opaque => None,
            }
        } else {
            self.resolve_instance(*parent, method)
        }
    }

    /// Static method of an exactly known class object. Static methods are not
    /// copied down at runtime, so only the class's own table is consulted.
    fn static_target(&self, class: ClassId, method: MethodSymbol) -> Option<FuncId> {
        if self.poisoned.contains(&method) {
            return None;
        }
        match self.static_methods.get(&(class, method))? {
            Binding::Bytecode(function) => Some(*function),
            Binding::Opaque => None,
        }
    }

    /// Sum of instance fields contributed by every ancestor of `class`, if the
    /// complete chain is static.
    fn inherited_field_count(&self, program: &Program, class: ClassId) -> Option<u32> {
        let mut total = 0u32;
        let mut current = *self.superclass.get(&class)?;
        for _ in 0..=self.superclass.len() {
            let Some(parent) = current else { return Some(total) };
            total += program.class(parent)?.own_instance_field_count as u32;
            current = *self.superclass.get(&parent)?;
        }
        None
    }

    /// FieldSlot operand that addresses `owner`'s local `slot` from a function
    /// compiled against `caller_owner`. Field operands are class-local and the
    /// runtime adds the owner's inherited field count once at class link time.
    /// Functions without a field owner may not contain field accesses at all
    /// (the bytecode verifier rejects them), so they keep the direct call.
    fn caller_slot(
        &self,
        program: &Program,
        caller_owner: Option<ClassId>,
        owner: ClassId,
        slot: FieldSlot,
    ) -> Option<FieldSlot> {
        let caller = caller_owner?;
        if caller == owner {
            return Some(slot);
        }
        let absolute = self.inherited_field_count(program, owner)? + slot.0 as u32;
        let base = self.inherited_field_count(program, caller)?;
        let local = absolute.checked_sub(base)?;
        u16::try_from(local).ok().map(FieldSlot)
    }
}

/// Module slots written only during module initialization, always with the
/// same `CreateClass` result. Reads of such a slot yield exactly that class.
fn discover_exact_class_slots(program: &Program) -> HashMap<(ModuleId, ModuleSlot), ClassId> {
    let mut found: HashMap<(ModuleId, ModuleSlot), ClassId> = HashMap::default();
    let mut ambiguous: HashSet<(ModuleId, ModuleSlot)> = HashSet::default();
    for function in &program.functions {
        for block in &function.blocks {
            for &id in &block.insts {
                let Inst::StoreModuleVar { module, slot, value } = &function.nodes[id.0 as usize].kind else {
                    continue;
                };
                let key = (*module, *slot);
                let created = if function.kind == FunctionKind::ModuleBody {
                    created_class(function, *value, 0)
                } else {
                    None
                };
                match (created, found.get(&key).copied()) {
                    (Some(class), None) => {
                        found.insert(key, class);
                    }
                    (Some(class), Some(previous)) if class == previous => {}
                    _ => {
                        ambiguous.insert(key);
                    }
                }
            }
        }
    }
    for key in ambiguous {
        found.remove(&key);
    }
    found
}

fn created_class(function: &Function, value: ValueId, depth: u8) -> Option<ClassId> {
    if depth >= TRACE_DEPTH {
        return None;
    }
    let ValueDef::InstResult(inst, 0) = function.value(value)?.def else { return None };
    match &function.node(inst)?.kind {
        Inst::CreateClass { class, .. } => Some(*class),
        Inst::Move(source) => created_class(function, *source, depth + 1),
        _ => None,
    }
}

// =============================================================================
// Trivial accessors
// =============================================================================

#[derive(Clone, Debug)]
enum Accessor {
    /// `get x() { return this._x }` compiled to LoadField + Return.
    Getter { owner: ClassId, slot: FieldSlot, ty: Type },
    /// `set x(v) { this._x = v }`, optionally validating `v` first. Synthesized
    /// property setters yield the stored value; source setters yield `null`.
    Setter {
        owner: ClassId,
        slot: FieldSlot,
        check: Option<(Type, Option<StructuralLayoutId>)>,
        returns_value: bool,
    },
}

fn classify_accessors(program: &Program) -> HashMap<FuncId, Accessor> {
    program
        .functions
        .iter()
        .filter_map(|function| classify_accessor(function).map(|accessor| (function.id, accessor)))
        .collect()
}

fn classify_accessor(function: &Function) -> Option<Accessor> {
    if function.kind != FunctionKind::Method || function.is_async || function.is_concurrent {
        return None;
    }
    let owner = function.field_owner?;
    let entry = function.block(function.entry)?;
    let receiver = *entry.params.first()?;
    let insts: Vec<&Inst> = entry
        .insts
        .iter()
        .map(|id| &function.nodes[id.0 as usize].kind)
        .collect();
    let result_of = |index: usize| -> Option<ValueId> {
        function.nodes[entry.insts[index].0 as usize].results.first().copied()
    };

    match (entry.params.len(), insts.as_slice()) {
        (1, [Inst::LoadField { object, slot }, Inst::Return(returned)])
            if *object == receiver && Some(*returned) == result_of(0) =>
        {
            let ty = function.value(*returned)?.ty.clone();
            Some(Accessor::Getter { owner, slot: *slot, ty })
        }
        (2, [Inst::StoreField { object, slot, value }, Inst::Return(returned)])
            if *object == receiver && *value == entry.params[1] && *returned == entry.params[1] =>
        {
            Some(Accessor::Setter { owner, slot: *slot, check: None, returns_value: true })
        }
        (2, [Inst::StoreField { object, slot, value }, Inst::Const(Constant::Null), Inst::Return(returned)])
            if *object == receiver && *value == entry.params[1] && Some(*returned) == result_of(1) =>
        {
            Some(Accessor::Setter { owner, slot: *slot, check: None, returns_value: false })
        }
        (
            2,
            [Inst::BindInterface { value: checked, expected, layout }, Inst::StoreField { object, slot, value }, Inst::Return(returned)],
        ) if *checked == entry.params[1]
            && *object == receiver
            && Some(*value) == result_of(0)
            && (*returned == entry.params[1] || Some(*returned) == result_of(0)) =>
        {
            Some(Accessor::Setter {
                owner,
                slot: *slot,
                check: Some((expected.clone(), *layout)),
                returns_value: true,
            })
        }
        (
            2,
            [Inst::BindInterface { value: checked, expected, layout }, Inst::StoreField { object, slot, value }, Inst::Const(Constant::Null), Inst::Return(returned)],
        ) if *checked == entry.params[1]
            && *object == receiver
            && Some(*value) == result_of(0)
            && Some(*returned) == result_of(2) =>
        {
            Some(Accessor::Setter {
                owner,
                slot: *slot,
                check: Some((expected.clone(), *layout)),
                returns_value: false,
            })
        }
        _ => None,
    }
}

// =============================================================================
// Result types
// =============================================================================

/// Guaranteed result type of every function, from its declared contract or from
/// the static types of the values it returns. Async/concurrent/module bodies,
/// constructors and functions with tail calls are left unknown.
fn infer_result_types(program: &Program) -> HashMap<FuncId, Type> {
    let mut results = HashMap::default();
    for function in &program.functions {
        if function.is_async
            || function.is_concurrent
            || matches!(function.kind, FunctionKind::ModuleBody | FunctionKind::Constructor)
        {
            continue;
        }
        if !matches!(function.signature.result, Type::Top) {
            results.insert(function.id, function.signature.result.clone());
            continue;
        }
        let mut joined: Option<Type> = None;
        let mut known = true;
        for block in &function.blocks {
            let Some(&last) = block.insts.last() else { continue };
            match &function.nodes[last.0 as usize].kind {
                Inst::Return(value) => {
                    let Some(info) = function.value(*value) else {
                        known = false;
                        break;
                    };
                    // Raw internal representations never cross the bytecode
                    // call ABI as tagged values; do not publish them.
                    if info.ty.is_internal_representation() || matches!(info.ty, Type::Cell(_)) {
                        known = false;
                        break;
                    }
                    joined = Some(match joined {
                        None => info.ty.clone(),
                        Some(previous) => previous.flow_join(&info.ty),
                    });
                }
                Inst::TailInvoke { .. }
                | Inst::TailInvokeSuper { .. }
                | Inst::TailCallDirect { .. }
                | Inst::TailCallClosure { .. } => {
                    known = false;
                    break;
                }
                _ => {}
            }
        }
        if !known {
            continue;
        }
        if let Some(ty) = joined.filter(|ty| !matches!(ty, Type::Top | Type::Bottom)) {
            results.insert(function.id, ty);
        }
    }
    results
}

/// Narrow `current` to `candidate` when both are guarantees and the candidate
/// carries strictly more information.
fn refine(current: &Type, candidate: &Type) -> Option<Type> {
    if matches!(candidate, Type::Top | Type::Bottom) || current == candidate {
        return None;
    }
    if matches!(current, Type::Top) || current.accepts(candidate) {
        Some(candidate.clone())
    } else {
        None
    }
}

fn requires_runtime_contract(ty: &Type) -> bool {
    match ty {
        Type::Interface(_) | Type::Record(_) | Type::List(Some(_)) | Type::GenericMap(_, _) | Type::Set(Some(_)) => true,
        Type::Builtin(
            BuiltinId::Event | BuiltinId::Emitter | BuiltinId::Channel | BuiltinId::Receiver | BuiltinId::ConcurrentMap | BuiltinId::ConcurrentSet | BuiltinId::Generator,
            Some(_),
        ) => true,
        Type::Union(types) => types.iter().any(requires_runtime_contract),
        _ => false,
    }
}

fn builtin_result(
    selector: BuiltinSelector,
    receiver: &Type,
    args: &[ValueId],
    type_of: &impl Fn(ValueId) -> Type,
) -> Option<Type> {
    match selector {
        BuiltinSelector::Subscript => {
            let Type::List(Some(element)) = receiver else { return None };
            // A Range index slices; only a numeric index yields one element.
            // Out-of-range (and fractional) indices yield `null`, so the
            // guaranteed result is `T | null`, never plain `T`.
            matches!(type_of(*args.first()?), Type::Num)
                .then(|| element.flow_join(&Type::Null))
        }
        // Only the stock count getters registered for each builtin class: on a
        // Map-backed value `length` is an ordinary property read, and `size`
        // does not exist on List/String.
        BuiltinSelector::Length => match receiver {
            Type::List(_) | Type::String => Some(Type::Num),
            _ => None,
        },
        BuiltinSelector::Count => match receiver {
            Type::List(_)
            | Type::String
            | Type::Map
            | Type::GenericMap(_, _)
            | Type::Interface(_)
            | Type::Record(_)
            | Type::Set(_) => Some(Type::Num),
            _ => None,
        },
        BuiltinSelector::Size => match receiver {
            Type::Map | Type::GenericMap(_, _) | Type::Interface(_) | Type::Record(_) | Type::Set(_) => {
                Some(Type::Num)
            }
            _ => None,
        },
    }
}

fn numeric_result(
    op: NumericOp,
    receiver: &Type,
    args: &[ValueId],
    type_of: &impl Fn(ValueId) -> Type,
) -> Option<Type> {
    if !matches!(receiver, Type::Num) {
        return None;
    }
    let rhs_is_num = || args.first().is_some_and(|arg| matches!(type_of(*arg), Type::Num));
    match op {
        NumericOp::Unary => Some(Type::Num),
        NumericOp::Equality => Some(Type::Bool),
        NumericOp::Arithmetic => rhs_is_num().then_some(Type::Num),
        NumericOp::Ordering => rhs_is_num().then_some(Type::Bool),
    }
}

// =============================================================================
// Planning
// =============================================================================

enum Rewrite {
    /// Replace the node's instruction in place (same result count).
    Replace(Inst),
    /// Inline a trivial setter: optionally validate `value`, store it, and turn
    /// the call itself into the setter's result (the stored value or `null`).
    InlineSetter {
        object: ValueId,
        slot: FieldSlot,
        value: ValueId,
        check: Option<(Type, Option<StructuralLayoutId>)>,
        returns_value: bool,
    },
}

#[derive(Default)]
struct FunctionPlan {
    rewrites: Vec<(InstId, Rewrite)>,
    retypes: Vec<(ValueId, Type)>,
}

impl FunctionPlan {
    fn is_empty(&self) -> bool {
        self.rewrites.is_empty() && self.retypes.is_empty()
    }
}

struct Planner<'a> {
    program: &'a Program,
    function: &'a Function,
    hierarchy: &'a ClassHierarchy,
    accessors: &'a HashMap<FuncId, Accessor>,
    returns: &'a HashMap<FuncId, Type>,
    selectors: &'a SelectorTable,
    /// Types refined earlier in this sweep, visible to later instructions.
    types: HashMap<ValueId, Type>,
    plan: FunctionPlan,
}

fn plan_function(
    program: &Program,
    function: &Function,
    hierarchy: &ClassHierarchy,
    accessors: &HashMap<FuncId, Accessor>,
    returns: &HashMap<FuncId, Type>,
    selectors: &SelectorTable,
) -> FunctionPlan {
    let mut planner = Planner {
        program,
        function,
        hierarchy,
        accessors,
        returns,
        selectors,
        types: HashMap::default(),
        plan: FunctionPlan::default(),
    };
    // Blocks are lowered roughly in dominance order; a value whose operand is
    // refined later in the sweep is picked up by the next fixed-point round.
    for block in &function.blocks {
        for &id in &block.insts {
            planner.visit(id);
        }
    }
    planner.plan
}

impl Planner<'_> {
    fn type_of(&self, value: ValueId) -> Type {
        self.types
            .get(&value)
            .cloned()
            .or_else(|| self.function.value(value).map(|info| info.ty.clone()))
            .unwrap_or(Type::Top)
    }

    fn narrow(&mut self, value: Option<ValueId>, candidate: Option<Type>) {
        let (Some(value), Some(candidate)) = (value, candidate) else { return };
        if let Some(narrowed) = refine(&self.type_of(value), &candidate) {
            self.types.insert(value, narrowed.clone());
            self.plan.retypes.push((value, narrowed));
        }
    }

    fn receiver_args(receiver: ValueId, args: &[ValueId]) -> Vec<ValueId> {
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(receiver);
        call_args.extend(args.iter().copied());
        call_args
    }

    fn visit(&mut self, id: InstId) {
        let function = self.function;
        let node = &function.nodes[id.0 as usize];
        let result = node.results.first().copied();
        match &node.kind {
            Inst::Invoke { receiver, method, args, .. } => {
                let receiver_type = self.type_of(*receiver);
                if let Some((rewrite, ty)) = self.devirtualize(&receiver_type, *receiver, *method, args) {
                    self.plan.rewrites.push((id, rewrite));
                    self.narrow(result, ty);
                    return;
                }
                let type_of = |value: ValueId| self.type_of(value);
                let typed = self
                    .selectors
                    .numeric
                    .get(method)
                    .and_then(|op| numeric_result(*op, &receiver_type, args, &type_of))
                    .or_else(|| {
                        self.selectors
                            .builtin
                            .get(method)
                            .and_then(|selector| builtin_result(*selector, &receiver_type, args, &type_of))
                    });
                self.narrow(result, typed);
            }
            Inst::InvokeSuper { home_class, receiver, method, args, .. } => {
                let Some(target) = self.super_target(*home_class, *receiver, *method, args.len()) else { return };
                self.plan.rewrites.push((id, Rewrite::Replace(Inst::CallDirect {
                    conv: CallConv::Napalm,
                    callee: target,
                    args: Self::receiver_args(*receiver, args),
                })));
                self.narrow(result, self.returns.get(&target).cloned());
            }
            Inst::TailInvoke { receiver, method, args, .. } => {
                let receiver_type = self.type_of(*receiver);
                let Some(target) = self.invoke_target(&receiver_type, *receiver, *method) else { return };
                if !self.call_shape_matches(target, args.len() + 1) {
                    return;
                }
                self.plan.rewrites.push((id, Rewrite::Replace(Inst::TailCallDirect {
                    conv: CallConv::Napalm,
                    callee: target,
                    args: Self::receiver_args(*receiver, args),
                })));
            }
            Inst::TailInvokeSuper { home_class, receiver, method, args, .. } => {
                let Some(target) = self.super_target(*home_class, *receiver, *method, args.len()) else { return };
                self.plan.rewrites.push((id, Rewrite::Replace(Inst::TailCallDirect {
                    conv: CallConv::Napalm,
                    callee: target,
                    args: Self::receiver_args(*receiver, args),
                })));
            }
            Inst::CallDirect { conv: CallConv::Napalm, callee, args } => {
                // The lowerer already binds accessors of exactly typed receivers
                // directly; those calls can collapse into the field access too.
                if let Some((&receiver, rest)) = args.split_first() {
                    if matches!(self.type_of(receiver), Type::Instance(_)) {
                        if let Some((rewrite, ty)) = self.inline_accessor(*callee, receiver, rest) {
                            self.plan.rewrites.push((id, rewrite));
                            self.narrow(result, ty);
                            return;
                        }
                    }
                }
                self.narrow(result, self.returns.get(callee).cloned());
            }
            Inst::CallDirect { callee, .. } => {
                self.narrow(result, self.returns.get(callee).cloned());
            }
            Inst::BindInterface { value, expected, .. } => {
                // Scalar contracts only validate; when the operand is already
                // proven to satisfy them the check is a no-op. Heap contracts
                // are persistent writes and must stay, except when the operand
                // comes straight out of a binding of the identical contract:
                // rebinding an installed contract returns immediately at runtime.
                if requires_runtime_contract(expected) {
                    if bound_by_identical_contract(self.function, *value, expected) {
                        self.plan.rewrites.push((id, Rewrite::Replace(Inst::Move(*value))));
                    }
                    return;
                }
                let actual = self.type_of(*value);
                if matches!(actual, Type::Top | Type::Bottom) || actual.is_internal_representation() {
                    return;
                }
                if expected.accepts(&actual) {
                    self.plan.rewrites.push((id, Rewrite::Replace(Inst::Move(*value))));
                    self.narrow(result, Some(actual));
                }
            }
            Inst::Move(source) => {
                let source_type = self.type_of(*source);
                self.narrow(result, Some(source_type));
            }
            _ => {}
        }
    }

    /// The unique bytecode target of an `Invoke` given its receiver's static type.
    fn invoke_target(&self, receiver_type: &Type, receiver: ValueId, method: MethodSymbol) -> Option<FuncId> {
        match receiver_type {
            Type::Instance(set) => self.hierarchy.unique_instance_target(set, method),
            // Static calls are only devirtualized on provably exact class
            // objects: `this` inside a static method may be a subclass reached
            // through `super`.
            // A module binding of a class reads as `Top`, yet `trace_class` can
            // still prove the exact class object (single CreateClass store):
            // `new C(...)` / `C.staticMethod(...)` resolve statically too.
            Type::Class(_) | Type::Top => {
                let class = self.hierarchy.trace_class(self.function, receiver, 0)?;
                self.hierarchy.static_target(class, method)
            }
            _ => None,
        }
    }

    fn super_target(&self, home: ClassId, receiver: ValueId, method: MethodSymbol, arity: usize) -> Option<FuncId> {
        let is_static = match self.type_of(receiver) {
            Type::Instance(_) => false,
            Type::Class(_) => true,
            _ => return None,
        };
        let target = self.hierarchy.super_target(home, method, is_static)?;
        self.call_shape_matches(target, arity + 1).then_some(target)
    }

    /// The callee must accept exactly the receiver-prefixed argument list.
    fn call_shape_matches(&self, target: FuncId, arg_count: usize) -> bool {
        self.program.function(target).is_some_and(|callee| {
            !callee.is_concurrent
                && callee.signature.params.len() == arg_count
                && matches!(
                    callee.kind,
                    FunctionKind::Method | FunctionKind::StaticMethod | FunctionKind::Constructor
                )
        })
    }

    fn devirtualize(
        &self,
        receiver_type: &Type,
        receiver: ValueId,
        method: MethodSymbol,
        args: &[ValueId],
    ) -> Option<(Rewrite, Option<Type>)> {
        let target = self.invoke_target(receiver_type, receiver, method)?;
        if !self.call_shape_matches(target, args.len() + 1) {
            return None;
        }

        // Only instance receivers can use an instance-field accessor directly.
        if matches!(receiver_type, Type::Instance(_)) {
            if let Some(inlined) = self.inline_accessor(target, receiver, args) {
                return Some(inlined);
            }
        }

        Some((
            Rewrite::Replace(Inst::CallDirect {
                conv: CallConv::Napalm,
                callee: target,
                args: Self::receiver_args(receiver, args),
            }),
            self.returns.get(&target).cloned(),
        ))
    }

    /// Field access equivalent to calling the trivial accessor `target` on an
    /// instance `receiver` (`args` excludes the receiver).
    fn inline_accessor(
        &self,
        target: FuncId,
        receiver: ValueId,
        args: &[ValueId],
    ) -> Option<(Rewrite, Option<Type>)> {
        let caller_owner = self.function.field_owner;
        match self.accessors.get(&target)? {
            Accessor::Getter { owner, slot, ty } if args.is_empty() => {
                let slot = self.hierarchy.caller_slot(self.program, caller_owner, *owner, *slot)?;
                Some((Rewrite::Replace(Inst::LoadField { object: receiver, slot }), Some(ty.clone())))
            }
            Accessor::Setter { owner, slot, check, returns_value } if args.len() == 1 => {
                let slot = self.hierarchy.caller_slot(self.program, caller_owner, *owner, *slot)?;
                let value = args[0];
                let ty = match (returns_value, check) {
                    (false, _) => Type::Null,
                    (true, Some((expected, _))) => expected.clone(),
                    (true, None) => self.type_of(value),
                };
                Some((
                    Rewrite::InlineSetter {
                        object: receiver,
                        slot,
                        value,
                        check: check.clone(),
                        returns_value: *returns_value,
                    },
                    Some(ty),
                ))
            }
            _ => None,
        }
    }
}

// =============================================================================
// Application
// =============================================================================

fn apply_plan(function: &mut Function, plan: FunctionPlan) {
    let mut changed_blocks: HashSet<usize> = HashSet::default();
    let mut position_of: HashMap<InstId, (usize, usize)> = HashMap::default();
    for (block_ix, block) in function.blocks.iter().enumerate() {
        for (position, id) in block.insts.iter().enumerate() {
            position_of.insert(*id, (block_ix, position));
        }
    }

    // Insertions shift positions; apply them per block from the back. Several
    // insertions before the same node keep their relative `order`.
    let mut insertions: Vec<(usize, usize, usize, InstId)> = Vec::new();
    for (id, rewrite) in plan.rewrites {
        let Some(&(block_ix, position)) = position_of.get(&id) else { continue };
        changed_blocks.insert(block_ix);
        match rewrite {
            Rewrite::Replace(kind) => {
                function.nodes[id.0 as usize].kind = kind;
            }
            Rewrite::InlineSetter { object, slot, value, check, returns_value } => {
                let source = function.nodes[id.0 as usize].source;
                let mut factory = InstFactory::new(function);
                let mut before = Vec::with_capacity(2);
                let stored = match check {
                    Some((expected, layout)) => {
                        let (bind, results) = factory.create(
                            Inst::BindInterface { value, expected: expected.clone(), layout },
                            &[expected],
                            source,
                        );
                        before.push(bind);
                        results[0]
                    }
                    None => value,
                };
                let (store, _) = factory.create(Inst::StoreField { object, slot, value: stored }, &[], source);
                before.push(store);
                // The call's own result becomes the setter's result.
                function.nodes[id.0 as usize].kind = if returns_value {
                    Inst::Move(stored)
                } else {
                    Inst::Const(Constant::Null)
                };
                for (order, inst) in before.into_iter().enumerate() {
                    insertions.push((block_ix, position, order, inst));
                }
            }
        }
    }
    insertions.sort_by(|left, right| (right.0, right.1, right.2).cmp(&(left.0, left.1, left.2)));
    for (block_ix, position, _, id) in insertions {
        function.blocks[block_ix].insts.insert(position, id);
    }

    for (value, ty) in plan.retypes {
        if let Some(info) = function.values.get_mut(value.0 as usize) {
            if info.id == value {
                info.ty = ty;
            }
        }
    }

    for block_ix in changed_blocks {
        repair_effect_chain(function, block_ix);
    }
}

// =============================================================================
// Allocation-enabling inlining
// =============================================================================

/// Upper bound on the instructions of an inlined callee body.
const MAX_INLINE_INSTS: usize = 24;

/// A straight-line callee body that can be copied into a caller verbatim.
struct InlineBody {
    params: Vec<ValueId>,
    /// (instruction, result types, source) in program order, Return excluded.
    insts: Vec<(Inst, smallvec::SmallVec<[ValueId; 1]>, Vec<Type>, Option<crate::napalm_ir::SourceSpan>)>,
    returned: ValueId,
}

/// Whether a callee instruction may be copied into another function. Only
/// context-free operations qualify: no field/module/capture state of the callee's
/// own frame, no deopt metadata, no control flow, fibers or tail calls. Errors
/// raised by the copies (e.g. a failed contract) propagate exactly like errors
/// of the call they replace, through the caller's own exception regions.
fn inlinable_inst(kind: &Inst) -> bool {
    matches!(
        kind,
        Inst::Const(_)
            | Inst::Move(_)
            | Inst::BindInterface { .. }
            | Inst::Invoke { .. }
            | Inst::CallDirect { conv: CallConv::Napalm, .. }
            | Inst::LoadProperty { .. }
            | Inst::StoreProperty { .. }
            | Inst::AllocMap
            | Inst::AllocInstance { .. }
            | Inst::MapGet { .. }
            | Inst::MapSet { .. }
            | Inst::MapLen { .. }
            | Inst::AllocList { .. }
            | Inst::ListLen { .. }
            | Inst::ListGet { .. }
            | Inst::ListAppend { .. }
            | Inst::StringLen { .. }
            | Inst::StringConcat { .. }
            | Inst::IntUnary { .. }
            | Inst::IntBin { .. }
            | Inst::F64Unary { .. }
            | Inst::F64Bin { .. }
            | Inst::CmpInt { .. }
            | Inst::CmpF64 { .. }
            | Inst::RefEq { .. }
            | Inst::NumToInt32 { .. }
            | Inst::NumToF64 { .. }
            | Inst::Int32ToNum { .. }
            | Inst::F64ToNum { .. }
    )
}

fn inline_body(function: &Function) -> Option<InlineBody> {
    if function.is_async
        || function.is_concurrent
        || !function.captures.is_empty()
        || !function.exception_regions.is_empty()
        // Memo tables and `computed` validation live at the call boundary.
        || function.memoization.enabled()
        || !matches!(function.kind, FunctionKind::Closure | FunctionKind::Method | FunctionKind::StaticMethod)
    {
        return None;
    }
    let entry = function.blocks.get(function.entry.0 as usize)?;
    let (&last, body) = entry.insts.split_last()?;
    let Inst::Return(returned) = function.nodes.get(last.0 as usize)?.kind else { return None };
    if body.len() > MAX_INLINE_INSTS {
        return None;
    }
    // Every operand must be a parameter or an earlier result of the body.
    let mut defined: HashSet<ValueId> = entry.params.iter().copied().collect();
    let mut insts = Vec::with_capacity(body.len());
    for &id in body {
        let node = function.nodes.get(id.0 as usize)?;
        if !inlinable_inst(&node.kind) || node.kind.operands().iter().any(|value| !defined.contains(value)) {
            return None;
        }
        if matches!(node.kind, Inst::CallDirect { callee, .. } if callee == function.id) {
            return None;
        }
        let types = node
            .results
            .iter()
            .map(|value| function.values.get(value.0 as usize).map(|value| value.ty.clone()))
            .collect::<Option<Vec<_>>>()?;
        defined.extend(node.results.iter().copied());
        insts.push((node.kind.clone(), node.results.clone(), types, node.source));
    }
    defined.contains(&returned).then(|| InlineBody { params: entry.params.clone(), insts, returned })
}

/// Whether `value` is (a copy or binding of) an allocation made by `function`.
/// Passing such an object to a small callee is what keeps scalar replacement
/// from removing it; inlining the callee lets escape analysis see its uses.
fn is_local_allocation(function: &Function, mut value: ValueId) -> bool {
    for _ in 0..TRACE_DEPTH {
        let Some(info) = function.values.get(value.0 as usize) else { return false };
        let ValueDef::InstResult(id, 0) = info.def else { return false };
        match &function.nodes[id.0 as usize].kind {
            Inst::AllocMap | Inst::AllocList { .. } => return true,
            Inst::Move(source) | Inst::BindInterface { value: source, .. } => value = *source,
            _ => return false,
        }
    }
    false
}

/// A synthesized `new` factory: `AllocInstance; [CallDirect ctor(inst, ..)];
/// Return inst`. Inlining it puts the allocation and the constructor call next
/// to each other in the caller, where the bytecode compiler fuses them into one
/// `NewClassInstance` (known class, layout and constructor) and the JIT tiers
/// see the exact class of the new object.
fn is_constructor_factory(body: &InlineBody) -> bool {
    let Some((Inst::AllocInstance { .. }, alloc_results, ..)) = body.insts.first() else { return false };
    let [instance] = alloc_results.as_slice() else { return false };
    if body.returned != *instance { return false; }
    match &body.insts[1..] {
        [] => true,
        [(Inst::CallDirect { conv: CallConv::Napalm, args, .. }, ..)] => {
            args.first() == Some(instance) && !args[1..].contains(instance)
        }
        _ => false,
    }
}

/// Inline small straight-line `CallDirect` targets that receive an object
/// allocated by the caller, and synthesized constructor factories. Static types made these calls direct; inlining them
/// lets the ordinary escape analysis scalar-replace the object (e.g. a
/// structural `{ x, y }` literal handed to `dist(a: Point, b: Point)`), which
/// removes the allocation, its contract binding and every property access.
/// Returns the indices of the rewritten callers.
fn inline_allocation_calls(program: &mut Program) -> Vec<usize> {
    inline_calls(program, false)
}

/// AOT also inlines scalar helpers. Preserve the context/exception restrictions
/// of allocation inlining and refuse recursive call-graph cycles/code explosion.
pub(super) fn inline_aot_calls(program: &mut Program) -> Vec<usize> {
    inline_calls(program, true)
}

fn inline_calls(program: &mut Program, aot: bool) -> Vec<usize> {
    fn reaches(program: &Program, from: FuncId, target: FuncId, seen: &mut HashSet<FuncId>) -> bool {
        if from == target { return true; }
        if !seen.insert(from) { return false; }
        let Some(function) = program.function(from) else { return true };
        function.blocks.iter().flat_map(|b| &b.insts).any(|id| {
            match &function.nodes[id.0 as usize].kind {
                Inst::CallDirect { callee, .. } | Inst::TailCallDirect { callee, .. } => reaches(program, *callee, target, seen),
                _ => false,
            }
        })
    }
    let bodies: HashMap<FuncId, InlineBody> = program
        .functions
        .iter()
        .filter_map(|function| inline_body(function).map(|body| (function.id, body)))
        .collect();
    if bodies.is_empty() {
        return Vec::new();
    }
    let mut touched = Vec::new();
    for index in 0..program.functions.len() {
        let caller_id = program.functions[index].id;
        let mut sites: Vec<(usize, InstId, FuncId)> = Vec::new();
        {
            let caller = &program.functions[index];
            let mut remaining = 512usize.saturating_sub(caller.blocks.iter().map(|b| b.insts.len()).sum());
            for (block_ix, block) in caller.blocks.iter().enumerate() {
                for &id in &block.insts {
                    let Inst::CallDirect { conv: CallConv::Napalm, callee, args } = &caller.nodes[id.0 as usize].kind
                    else {
                        continue;
                    };
                    let Some(body) = bodies.get(callee) else { continue };
                    if *callee == caller_id
                        || body.params.len() != args.len()
                        || caller.nodes[id.0 as usize].results.len() != 1
                        || !(aot || is_constructor_factory(body)
                            || args.iter().any(|arg| is_local_allocation(caller, *arg)))
                        || (aot && (body.insts.len() > remaining
                            || reaches(program, *callee, caller_id, &mut HashSet::default())
                            // A recursive callee would unroll once per HIR round.
                            || body.insts.iter().any(|(kind, ..)| matches!(kind,
                                Inst::CallDirect { callee: next, .. }
                                    if reaches(program, *next, *callee, &mut HashSet::default())))))
                    {
                        continue;
                    }
                    sites.push((block_ix, id, *callee));
                    remaining = remaining.saturating_sub(body.insts.len());
                }
            }
        }
        if sites.is_empty() {
            continue;
        }
        let mut changed_blocks: HashSet<usize> = HashSet::default();
        for (block_ix, call, callee) in sites {
            let body = &bodies[&callee];
            let mut fresh_sites = Vec::new();
            for (kind, ..) in &body.insts {
                if matches!(kind, Inst::Invoke { .. }) {
                    fresh_sites.push(program.fresh_site());
                }
            }
            let caller = &mut program.functions[index];
            let Inst::CallDirect { args, .. } = caller.nodes[call.0 as usize].kind.clone() else { continue };
            let source = caller.nodes[call.0 as usize].source;
            let mut map: HashMap<ValueId, ValueId> = body.params.iter().copied().zip(args).collect();
            let mut created = Vec::with_capacity(body.insts.len());
            let mut fresh_sites = fresh_sites.into_iter();
            for (kind, results, types, inst_source) in &body.insts {
                let mut kind = kind.clone();
                kind.for_each_operand_mut(|value| *value = map[value]);
                if let Inst::Invoke { site, .. } = &mut kind {
                    *site = fresh_sites.next().expect("one fresh site per Invoke");
                }
                let (id, new_results) = InstFactory::new(caller).create(kind, types, inst_source.or(source));
                for (old, new) in results.iter().zip(new_results) {
                    map.insert(*old, new);
                }
                created.push(id);
            }
            // The call node keeps its result value (and therefore all its uses).
            caller.nodes[call.0 as usize].kind = Inst::Move(map[&body.returned]);
            let block = &mut caller.blocks[block_ix];
            let position = block.insts.iter().position(|id| *id == call).expect("call site in its block");
            block.insts.splice(position..position, created);
            changed_blocks.insert(block_ix);
        }
        let caller = &mut program.functions[index];
        for block_ix in changed_blocks {
            repair_effect_chain(caller, block_ix);
        }
        touched.push(index);
    }
    touched
}

/// Whether `value` is (a copy of) the result of `BindInterface` with exactly
/// `expected`. Persistent contracts cannot be removed, so such a value already
/// carries the installed contract and binding it again is a runtime no-op.
fn bound_by_identical_contract(function: &Function, mut value: ValueId, expected: &Type) -> bool {
    for _ in 0..TRACE_DEPTH {
        let Some(info) = function.values.get(value.0 as usize) else { return false };
        let ValueDef::InstResult(id, 0) = info.def else { return false };
        match &function.nodes[id.0 as usize].kind {
            Inst::BindInterface { expected: installed, .. } if installed == expected => return true,
            Inst::Move(source) => value = *source,
            _ => return false,
        }
    }
    false
}

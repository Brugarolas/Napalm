//! Trace-tree regions: one LLVM function for a hot trace (the *head*) and every
//! trace it reaches through stitched native links.
//!
//! Separately compiled traces cross a link by materializing the child's entry
//! state in the frame, returning `TRACE_RESULT_LINK` and letting the native
//! dispatcher indirect-call the child, which reloads every entry slot, re-guards
//! it and reloads its VM bindings. Inside a region a link to another member is
//! a direct branch to that member's `m{k}_start` block instead:
//!
//! ```text
//!   entry --> m0_start (head) ... exit e ---- linked to member 1 ----+
//!               ^                                                    |
//!               +---- m1_start ... closing exit (linked to head) <---+
//! ```
//!
//! The frame writes of the link are still emitted (any later real side exit
//! relies on them), but they are plain stores that the child's entry loads
//! immediately follow, so GVN forwards them and the values stay in registers.
//!
//! Type-guarded entry reads are not re-guarded either: a link passes the
//! child the exact value its guard would compute from the written word (the
//! unboxed operand of a box, a reconstructible constant, or the parent's own
//! guarded entry value for a slot it never wrote), see `TypedEntry`. The head
//! guards on the function-entry edge only, in its `m0_entry` prologue.
//! `NAPALM_LLVM_REGION_TYPED=0` disables this (A/B measurements).
//! Nested loops become real LLVM loop nests: the head is the only entry, so
//! the outer cycle is reducible and its header is `m0_start`.
//!
//! The dispatcher protocol is preserved at every internal link: the link
//! budget (`dispatch_links_remaining`) is spent exactly as the dispatcher
//! spends it, and a pending tier-up yields with the child's state materialized,
//! so a recompiled (grown) region still gets installed. Returns from any member
//! name that member in `next_trace`, which is how the manager attributes exits.
//!
//! Each member is lowered by its own `LlvmLowering` (its value numbering, heap
//! caches and proofs are trace-local), but the module-level state (SSA and
//! label counters, TBAA leaves, PGO metadata) is handed from member to member.
//! Two facts that are only valid for a single invocation of a single trace are
//! disabled region-wide: identity noalias scopes (members are re-entered with
//! fresh entry values) and `!invariant.load` unless no member crosses a
//! runtime/GC epoch.

use super::*;
use std::cell::RefCell;
use rclite::Rc;

/// One member of a region job: data cloned by the manager.
pub(in crate::napalm_tracing_jit) struct RegionMemberJob {
    pub(in crate::napalm_tracing_jit) trace: Trace,
    pub(in crate::napalm_tracing_jit) exits: Vec<CompiledExitPlan>,
    pub(in crate::napalm_tracing_jit) binding_slots: VmBindingSlots,
    pub(in crate::napalm_tracing_jit) profile: Option<TraceProfile>,
}

/// Region context of the member currently being lowered.
pub(super) struct RegionMember {
    pub(super) trace: TraceId,
    /// Label/stack-slot prefix of this member (`m{k}_`).
    pub(super) prefix: String,
    /// Start block of every member, by trace id.
    starts: HashMap<TraceId, String>,
    /// This member's child-link table (entry-block value).
    pub(super) links: V,
    /// This member's VM-binding table; None for the head, which is entered
    /// with its own table in `TraceActivation::vm_bindings`.
    pub(super) bindings: Option<V>,
    /// `TraceActivation::tier_up_pending` (entry-block value).
    tier_up: V,
    /// Typed entry slots of every member, by trace id (see `TypedEntry`).
    typed_entries: Rc<HashMap<TraceId, Vec<TypedEntry>>>,
    /// Deferred frame slots (see `deferred_frame_slots`): slot -> alloca.
    deferred: Rc<Vec<(FrameSlot, String)>>,
    /// With deferred slots, every return branches to one `region_exit` block
    /// that flushes them once: `(predecessor block, result code)`.
    returns: Rc<RefCell<Vec<(String, u32)>>>,
}

/// Frame slots whose architectural value lives in an entry-block alloca for
/// the whole region invocation instead of in the frame: every General slot
/// written by an internal link. All reads and writes of such a slot go to the
/// alloca (entry reads, LoadFrameSlot, link writes, exit writes, commits) and
/// every `ret` flushes all of them to the frame, so the frame is exact again
/// whenever anything outside this function can observe it. In between,
/// nothing else can: deferral requires that no member crosses a runtime/GC
/// boundary (no helper reads or writes the frame, no collector scans it), and
/// General slots are words of the fiber's register stack, disjoint from the
/// other banks. A link therefore stores nothing to memory on the hot path.
/// `NAPALM_LLVM_REGION_DEFER=0` disables this (A/B measurements).
fn deferred_frame_slots(members: &[RegionMemberJob]) -> Vec<(FrameSlot, String)> {
    if std::env::var("NAPALM_LLVM_REGION_DEFER").ok().as_deref() == Some("0")
        || members.iter().any(|member| trace_has_heap_epoch_boundary(&member.trace))
    {
        return Vec::new();
    }
    let ids: HashSet<TraceId> = members.iter().map(|member| member.trace.id).collect();
    let mut slots: Vec<(FrameSlot, String)> = Vec::new();
    for member in members {
        for plan in &member.exits {
            let internal = plan.native_linkable && matches!(plan.target, ExitTarget::Trace(child) if ids.contains(&child));
            if !internal {
                continue;
            }
            for write in &plan.writes {
                if let FrameSlot::General(reg) = write.slot {
                    if !slots.iter().any(|(slot, _)| *slot == write.slot) {
                        slots.push((write.slot, format!("%frame_r{}", reg.0)));
                    }
                }
            }
        }
    }
    slots
}

impl RegionMember {
    pub(super) fn deferred_slots(&self) -> Rc<Vec<(FrameSlot, String)>> {
        self.deferred.clone()
    }

    /// Record a return from `block` with `code`; the caller branches to
    /// `region_exit`. None when nothing is deferred (plain `ret`).
    pub(super) fn record_return(&self, block: &str, code: u32) -> Option<&'static str> {
        if self.deferred.is_empty() {
            return None;
        }
        self.returns.borrow_mut().push((block.to_owned(), code));
        Some("region_exit")
    }

    pub(super) fn deferred_slot(&self, slot: FrameSlot) -> Option<&str> {
        self.deferred.iter().find(|(deferred, _)| *deferred == slot).map(|(_, name)| name.as_str())
    }
}

/// An entry slot that a member reads through a type guard (`GuardedEntry` or a
/// promoted loop phi's entry guard), passed unboxed by internal links in the
/// entry-block alloca `value` (mem2reg turns it into a phi).
///
/// Non-head members are entered only through internal links. A link that
/// can pass the value stores it and sets the `known` alloca, otherwise clears
/// it; every edge writes `known`, so a stale `true` is never observed, and
/// when all edges pass, `known` folds to true and the guard disappears.
///
/// The head is also entered by the function entry. A merged `known` flag
/// there would put control flow on the outer cycle's header, so the head only
/// takes slots that *every* internal link can pass (`always_known`): the
/// `m0_entry` prologue performs the guard on the function-entry edge alone.
#[derive(Clone)]
struct TypedEntry {
    slot: FrameSlot,
    rep: TraceRep,
    exit: SideExitId,
    known: String,
    value: String,
    always_known: bool,
}

impl RegionMember {
    pub(super) fn start_of(&self, trace: TraceId) -> Option<String> {
        self.starts.get(&trace).cloned()
    }
}

/// `(slot, rep, exit)` of every type-guarded entry read of `trace`.
fn typed_entry_slots(trace: &Trace) -> Vec<(FrameSlot, TraceRep, SideExitId)> {
    let entry_slot = |value: TraceValueId| {
        let value = trace.canonical_value(value);
        match trace.value(value).map(|info| &info.origin) {
            Some(TraceValueOrigin::EntrySlot { frame, slot }) if frame.0 == 0 => Some(*slot),
            _ => None,
        }
    };
    let guarded = trace.values.iter().filter_map(|info| match info.origin {
        TraceValueOrigin::GuardedEntry { value, exit } => entry_slot(value).map(|slot| (slot, info.rep, exit)),
        _ => None,
    });
    let phis = trace.loop_phis.iter().filter_map(|phi| {
        let exit = phi.entry_exit?;
        entry_slot(phi.initial).map(|slot| (slot, phi.rep, exit))
    });
    let mut slots = Vec::new();
    for entry in guarded.chain(phis) {
        if matches!(entry.1, TraceRep::F64 | TraceRep::Int32 | TraceRep::UInt32) && !slots.contains(&entry) {
            slots.push(entry);
        }
    }
    slots
}

/// What a link (`plan` of `trace`) can pass for a child entry read as `rep`
/// from `slot`: exactly the value the child's guard would produce from the
/// written word, when the guard is statically certain to pass. Only words
/// written as Tagged qualify (a raw F64 word with a NaN payload may fail).
enum PassSource {
    /// A trace value of the given representation (a box is looked through:
    /// BoxF64/BoxInt32/BoxUInt32 lower exactly like the exit materialization).
    /// `boxed`: the child sees the value through a box, which canonicalizes
    /// NaN; an untouched slot is seen as the very same raw word.
    Value { value: TraceValueId, rep: TraceRep, boxed: bool },
    Constant(V),
}

fn unboxed_operand(trace: &Trace, value: TraceValueId) -> TraceValueId {
    let value = trace.canonical_value(value);
    let Some(TraceValueOrigin::InstResult { node }) = trace.value(value).map(|info| &info.origin) else {
        return value;
    };
    match trace.nodes.get(*node).map(|node| &node.kind) {
        Some(TraceInst::BoxF64 { value: inner })
        | Some(TraceInst::BoxInt32 { value: inner })
        | Some(TraceInst::BoxUInt32 { value: inner }) => trace.canonical_value(*inner),
        _ => value,
    }
}

fn pass_source(trace: &Trace, plan: &CompiledExitPlan, slot: FrameSlot, rep: TraceRep) -> Option<PassSource> {
    let Some(write) = plan.writes.iter().rev().find(|write| write.slot == slot) else {
        return untouched_pass_source(trace, slot, rep);
    };
    if write.target_rep != TraceRep::Tagged {
        return None;
    }
    match write.source {
        ExitWriteSource::Dynamic(index) => {
            let dynamic = plan.dynamic_values.get(index as usize)?;
            let value = unboxed_operand(trace, dynamic.value);
            let source = trace.value(value)?.rep;
            let compatible = matches!(
                (source, rep),
                (TraceRep::F64 | TraceRep::Int32 | TraceRep::UInt32, TraceRep::F64)
                    | (TraceRep::Int32, TraceRep::Int32)
                    | (TraceRep::UInt32, TraceRep::UInt32)
            );
            compatible.then_some(PassSource::Value { value, rep: source, boxed: true })
        }
        ExitWriteSource::Constant(constant) => {
            let number = match constant {
                TraceConstant::Int32(value) => value as f64,
                TraceConstant::F64Bits(bits) => {
                    let value = f64::from_bits(bits);
                    if value.is_nan() { f64::from_bits(CANONICAL_NAN) } else { value }
                }
                TraceConstant::NumBits(bits) => word_num(RegisterWord(bits))?,
                _ => return None,
            };
            match rep {
                TraceRep::F64 => Some(PassSource::Constant(V::f64(number))),
                TraceRep::Int32
                    if number.to_bits() != (-0.0f64).to_bits() && number == (number as i32) as f64 =>
                {
                    Some(PassSource::Constant(V::i32(number as i32)))
                }
                TraceRep::UInt32 if number == (number as u32) as f64 => {
                    Some(PassSource::Constant(V::i32(number as u32 as i32)))
                }
                _ => None,
            }
        }
    }
}

/// A slot the link does not write still holds the parent's entry word, unless
/// a runtime helper rewrote the frame meanwhile. If the parent already guarded
/// that word, its unboxed entry value is exactly what the child's guard would
/// produce, for compatible representations. (A UInt32 guard accepts -0.0, so
/// it does not determine the F64 value.) GuardedEntry values are all lowered
/// in the parent's start block, so they dominate every exit.
fn untouched_pass_source(trace: &Trace, slot: FrameSlot, rep: TraceRep) -> Option<PassSource> {
    if trace_has_heap_epoch_boundary(trace) {
        return None;
    }
    trace.values.iter().find_map(|info| {
        let TraceValueOrigin::GuardedEntry { value: source, .. } = info.origin else { return None; };
        let source = trace.canonical_value(source);
        let reads_slot = matches!(
            trace.value(source).map(|source| &source.origin),
            Some(TraceValueOrigin::EntrySlot { frame, slot: read }) if frame.0 == 0 && *read == slot
        );
        let compatible = matches!(
            (info.rep, rep),
            (TraceRep::F64 | TraceRep::Int32, TraceRep::F64)
                | (TraceRep::Int32, TraceRep::Int32)
                | (TraceRep::UInt32, TraceRep::UInt32)
        );
        // The canonical id is what the parent's lowering caches.
        (reads_slot && compatible && trace.canonical_value(info.id) == info.id)
            .then_some(PassSource::Value { value: info.id, rep: info.rep, boxed: false })
    })
}

/// Typed entries of every member (see `TypedEntry`).
fn region_typed_entries(members: &[RegionMemberJob]) -> HashMap<TraceId, Vec<TypedEntry>> {
    let mode = std::env::var("NAPALM_LLVM_REGION_TYPED").ok();
    if mode.as_deref() == Some("0") {
        return members.iter().map(|member| (member.trace.id, Vec::new())).collect();
    }
    let head = members[0].trace.id;
    // Every internal link into the head.
    let head_links: Vec<(&Trace, &CompiledExitPlan)> = members
        .iter()
        .flat_map(|member| member.exits.iter().map(move |plan| (&member.trace, plan)))
        .filter(|(_, plan)| plan.native_linkable && matches!(plan.target, ExitTarget::Trace(child) if child == head))
        .collect();
    members
        .iter()
        .enumerate()
        .map(|(index, member)| {
            let entries = typed_entry_slots(&member.trace)
                .into_iter()
                .filter(|&(slot, rep, exit)| {
                    index > 0
                        || (member.exits.get(exit.0 as usize).is_some_and(|plan| plan.writes.is_empty())
                            && head_links
                                .iter()
                                .all(|(trace, plan)| pass_source(trace, plan, slot, rep).is_some()))
                })
                .enumerate()
                .map(|(n, (slot, rep, exit))| TypedEntry {
                    slot,
                    rep,
                    exit,
                    known: format!("%m{index}_known{n}"),
                    value: format!("%m{index}_typed{n}"),
                    always_known: index == 0,
                })
                .collect::<Vec<_>>();
            (member.trace.id, entries)
        })
        .collect()
}

pub(super) fn lower_region(
    members: &[RegionMemberJob],
    native_budget: u32,
    symbol: &str,
    host: &HostInfo,
) -> Result<String, String> {
    let head = members.first().ok_or_else(|| "empty trace-tree region".to_owned())?;
    let starts: HashMap<TraceId, String> = members
        .iter()
        .enumerate()
        .map(|(index, member)| (member.trace.id, format!("m{index}_start")))
        .collect();
    if starts.len() != members.len() {
        return Err("trace-tree region lists a trace twice".to_owned());
    }
    let invariant_loads = members.iter().all(|member| !trace_has_heap_epoch_boundary(&member.trace));
    let typed_entries: Rc<HashMap<TraceId, Vec<TypedEntry>>> = Rc::new(region_typed_entries(members));
    let deferred = Rc::new(deferred_frame_slots(members));
    let returns: Rc<RefCell<Vec<(String, u32)>>> = Rc::default();

    let mut previous: Option<LlvmLowering> = None;
    // Entry-block pointers shared by all members.
    let mut shared: Option<(V, V, V)> = None;
    for (index, member) in members.iter().enumerate() {
        let profile = member.profile.as_ref().filter(|profile| profile.entries > 0);
        let mut state = LlvmLowering::new(
            &member.trace,
            &member.exits,
            member.binding_slots.clone(),
            native_budget,
            profile,
        );
        if let Some(previous) = previous.as_mut() {
            state.adopt_module_state(previous);
        }
        state.invariant_loads = invariant_loads;
        state.speculative_noalias = false;

        let (link_tables, binding_tables, tier_up) = match &shared {
            Some(shared) => shared.clone(),
            None => {
                let link_tables = state.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, trace_link_tables));
                let binding_tables = state.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, trace_vm_bindings));
                let tier_up = state.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, tier_up_pending));
                let values = (link_tables, binding_tables, tier_up);
                shared = Some(values.clone());
                // Typed-entry allocas of all members (the head's prologue
                // fills its own on the function-entry edge).
                for member in members {
                    for typed in &typed_entries[&member.trace.id] {
                        let ty = Ty::of_rep(typed.rep);
                        if !typed.always_known {
                            let _ = writeln!(state.entry, "  {} = alloca i1, align 1", typed.known);
                        }
                        let _ = writeln!(state.entry, "  {} = alloca {}, align {}", typed.value, ty.ir(), ty.align());
                    }
                }
                // Deferred frame slots start as the frame's current words.
                // No runtime boundary in the region: the bank is invariant.
                if !deferred.is_empty() {
                    let general = state.entry_activation_load(Ty::Ptr, offset_of!(TraceActivation, general));
                    for (slot, name) in deferred.iter() {
                        let FrameSlot::General(reg) = slot else { continue; };
                        let offset = reg.0 as usize * std::mem::size_of::<RegisterWord>();
                        let ptr = state.def_entry(
                            Ty::Ptr,
                            &format!("getelementptr inbounds i8, ptr {}, i64 {offset}", general.op),
                        );
                        let word = state.def_entry(
                            Ty::I64,
                            &format!("load i64, ptr {}, align 8, !tbaa !{}", ptr.op, Mem::Reg.tag()),
                        );
                        let _ = writeln!(state.entry, "  {name} = alloca i64, align 8");
                        let _ = writeln!(state.entry, "  store i64 {}, ptr {name}, align 8", word.op);
                    }
                }
                values
            }
        };
        let (links, bindings) = if index == 0 {
            (LlvmLowering::links(), None)
        } else {
            (
                state.dense_table_entry(&link_tables, member.trace.id),
                Some(state.dense_table_entry(&binding_tables, member.trace.id)),
            )
        };
        state.region = Some(RegionMember {
            trace: member.trace.id,
            prefix: format!("m{index}_"),
            starts: starts.clone(),
            links,
            bindings,
            tier_up,
            typed_entries: typed_entries.clone(),
            deferred: deferred.clone(),
            returns: returns.clone(),
        });
        if index == 0 {
            state.emit_head_prologue("m0_start")?;
        }
        state.lower_function(native_budget)?;
        previous = Some(state);
    }

    let mut state = previous.ok_or_else(|| "empty trace-tree region".to_owned())?;
    // The single return of a region with deferred frame slots: flush them
    // once, then return the code of whichever exit branched here.
    let returns = returns.borrow().clone();
    if !returns.is_empty() {
        state.begin("region_exit");
        let incoming = returns
            .iter()
            .map(|(block, code)| format!("[ {code}, %{block} ]"))
            .collect::<Vec<_>>()
            .join(", ");
        let code = state.def(Ty::I32, &format!("phi i32 {incoming}"));
        state.flush_deferred_frame();
        state.line(&format!("ret i32 {}", code.op));
    }
    // The module is described by its head: entry count and profile summary.
    state.trace = &head.trace;
    state.profile = head.profile.as_ref().filter(|profile| profile.entries > 0);
    state.entry_branch_target = "m0_entry".to_owned();
    Ok(state.finish_module(symbol, host))
}

impl<'p> LlvmLowering<'p> {
    /// Continue the module of `previous`: emitted IR, SSA/label numbering and
    /// every metadata id allocator. Trace-local caches start empty.
    fn adopt_module_state(&mut self, previous: &mut LlvmLowering<'p>) {
        self.entry = std::mem::take(&mut previous.entry);
        self.body = std::mem::take(&mut previous.body);
        self.tmp = previous.tmp;
        self.labels = previous.labels;
        // One physical slot keeps one TBAA leaf in the whole function, so
        // accesses to it from different members still alias.
        self.shaped_slot_tbaa = std::mem::take(&mut previous.shaped_slot_tbaa);
        self.instance_slot_tbaa = std::mem::take(&mut previous.instance_slot_tbaa);
        self.next_tbaa = previous.next_tbaa;
        self.identity_proofs = std::mem::take(&mut previous.identity_proofs);
        self.identity_metadata_lists = std::mem::take(&mut previous.identity_metadata_lists);
        // A fresh epoch: no proof of an earlier member can look active here.
        self.identity_epoch = previous.identity_epoch.wrapping_add(1);
        self.prof_metadata = std::mem::take(&mut previous.prof_metadata);
        self.prof_weight_ids = std::mem::take(&mut previous.prof_weight_ids);
    }

    /// `table[trace]` of a dense TraceId-indexed pointer table installed in the
    /// activation (stable for one dispatcher invocation).
    fn dense_table_entry(&mut self, table: &V, trace: TraceId) -> V {
        let slot = self.def_entry(
            Ty::Ptr,
            &format!("getelementptr inbounds i64, ptr {}, i64 {}", table.op, trace.0),
        );
        self.def_entry(
            Ty::Ptr,
            &format!(
                "load ptr, ptr {}, align 8, !tbaa !{}, !invariant.load !{MD_EMPTY}",
                slot.op,
                Mem::Act.tag()
            ),
        )
    }

    /// For each typed entry of `child`: store the value its guard would
    /// produce from the word this link just wrote (`pass_source`), or clear
    /// `known`. An `always_known` (head) entry is passable by construction.
    fn pass_typed_entries(&mut self, plan: &CompiledExitPlan, child: TraceId) -> Result<(), String> {
        let Some(entries) = self
            .region
            .as_ref()
            .and_then(|region| region.typed_entries.get(&child).cloned())
        else {
            return Ok(());
        };
        for typed in &entries {
            let passed = match pass_source(self.trace, plan, typed.slot, typed.rep) {
                // Already lowered (cached): it dominates this exit.
                Some(PassSource::Value { value, rep: source, boxed }) => {
                    let value = self.lower_value(value)?;
                    Some(match (source, typed.rep) {
                        (TraceRep::F64, TraceRep::F64) if !boxed => value,
                        (TraceRep::F64, TraceRep::F64) => {
                            // Boxing canonicalizes NaN; so does the unboxed pass.
                            let is_nan = self.fcmp("uno", &value, &value);
                            self.select(&is_nan, &V::f64_bits(CANONICAL_NAN), &value)
                        }
                        (TraceRep::Int32, TraceRep::F64) => self.cast("sitofp", &value, Ty::F64),
                        (TraceRep::UInt32, TraceRep::F64) => self.cast("uitofp", &value, Ty::F64),
                        _ => self.coerce(&value, Ty::I32),
                    })
                }
                Some(PassSource::Constant(value)) => Some(value),
                None => None,
            };
            match passed {
                Some(value) => {
                    let ty = Ty::of_rep(typed.rep);
                    self.line(&format!("store {} {}, ptr {}, align {}", ty.ir(), value.op, typed.value, ty.align()));
                    if !typed.always_known {
                        self.line(&format!("store i1 true, ptr {}, align 1", typed.known));
                    }
                }
                None if typed.always_known => {
                    return Err(self.error(format!("link cannot pass head entry {:?}", typed.slot)));
                }
                None => self.line(&format!("store i1 false, ptr {}, align 1", typed.known)),
            }
        }
        Ok(())
    }

    /// The head's `m0_entry` prologue, on the function-entry edge only: guard
    /// and unbox every `always_known` entry straight from the frame. Those
    /// guards' exits write nothing and resume at the anchor, so performing
    /// them before the head's other entry checks preserves semantics.
    pub(super) fn emit_head_prologue(&mut self, start: &str) -> Result<(), String> {
        let entries = self
            .region
            .as_ref()
            .and_then(|region| region.typed_entries.get(&region.trace).cloned())
            .unwrap_or_default();
        self.begin("m0_entry");
        for typed in entries.iter().filter(|typed| typed.always_known) {
            // Not cached: this block does not dominate the head's body.
            let word = self.load_frame_slot(typed.slot, TraceRep::Tagged);
            let value = match typed.rep {
                TraceRep::F64 => self.guard_numeric(&word, typed.exit)?,
                TraceRep::Int32 => self.guard_int32_numeric(&word, typed.exit)?,
                TraceRep::UInt32 => self.guard_uint32_numeric(&word, typed.exit)?,
                other => return Err(self.error(format!("unsupported typed region entry {other:?}"))),
            };
            let ty = Ty::of_rep(typed.rep);
            self.line(&format!("store {} {}, ptr {}, align {}", ty.ir(), value.op, typed.value, ty.align()));
        }
        self.br(start);
        Ok(())
    }
    /// Entry read of a type-guarded slot inside a region: the unboxed value an
    /// internal link passed, or the ordinary guard when the edge (the function
    /// entry, or a link that only had a boxed word) passed nothing.
    pub(super) fn region_typed_entry(
        &mut self,
        source: TraceValueId,
        rep: TraceRep,
        exit: SideExitId,
    ) -> Result<Option<V>, String> {
        let Some(region) = self.region.as_ref() else { return Ok(None); };
        let source = self.trace.canonical_value(source);
        let slot = match self.trace.value(source).map(|info| &info.origin) {
            Some(TraceValueOrigin::EntrySlot { frame, slot }) if frame.0 == 0 => *slot,
            _ => return Ok(None),
        };
        let Some(typed) = region
            .typed_entries
            .get(&region.trace)
            .and_then(|entries| {
                entries.iter().find(|typed| typed.slot == slot && typed.rep == rep && typed.exit == exit)
            })
            .cloned()
        else {
            return Ok(None);
        };
        let ty = Ty::of_rep(rep);
        if typed.always_known {
            // Every edge into the head stored it: the prologue or a link.
            return Ok(Some(self.def(ty, &format!("load {}, ptr {}, align {}", ty.ir(), typed.value, ty.align()))));
        }
        // The boxed word stays loaded unconditionally: other users (exit
        // snapshots) may need it, and it must dominate them.
        let tagged = self.lower_value(source)?;
        let known = self.def(Ty::I1, &format!("load i1, ptr {}, align 1", typed.known));
        let passed = self.def(ty, &format!("load {}, ptr {}, align {}", ty.ir(), typed.value, ty.align()));
        let before = self.current.clone();
        let check = self.new_block();
        let have = self.new_block();
        self.cond_br(&known, &have, &check, None);
        self.begin(&check);
        let guarded = match rep {
            TraceRep::F64 => self.guard_numeric(&tagged, exit)?,
            TraceRep::Int32 => self.guard_int32_numeric(&tagged, exit)?,
            TraceRep::UInt32 => self.guard_uint32_numeric(&tagged, exit)?,
            other => return Err(self.error(format!("unsupported typed region entry {other:?}"))),
        };
        let checked = self.current.clone();
        self.br(&have);
        self.begin(&have);
        let merged = self.def(
            ty,
            &format!("phi {} [ {}, %{before} ], [ {}, %{checked} ]", ty.ir(), passed.op, guarded.op),
        );
        Ok(Some(merged))
    }

    /// A native link to another region member: materialize the child's entry
    /// state, honor the dispatcher's tier-up and link-budget protocol, then
    /// branch to the member instead of returning to the dispatcher.
    pub(super) fn materialize_region_link(
        &mut self,
        plan: &CompiledExitPlan,
        child: TraceId,
        start: &str,
    ) -> Result<(), String> {
        self.store_link_writes(plan)?;
        self.pass_typed_entries(plan, child)?;
        let tier_up = self
            .region
            .as_ref()
            .map(|region| region.tier_up.clone())
            .ok_or_else(|| self.error("region link outside a region"))?;
        let activation = Self::activation();
        let remaining_offset = offset_of!(TraceActivation, dispatch_links_remaining) as i64;

        let fuel = self.new_block();
        let spend = self.new_block();
        let enter = self.new_block();
        let yield_to_dispatcher = self.new_block();

        // Installed by another thread: an atomic (monotonic) load, which LLVM
        // neither hoists out of the region's loops nor treats as a clobber.
        let pending = self.def(Ty::I32, &format!("load atomic i32, ptr {} monotonic, align 4", tier_up.op));
        let upgrade = self.icmp("ne", &pending, &V::i32(0));
        self.cond_br(&upgrade, &yield_to_dispatcher, &fuel, Some(MD_COLD_TRUE));

        self.begin(&fuel);
        let remaining = self.load_at(Ty::I32, &activation, remaining_offset, Mem::Act);
        let unlimited = self.icmp("eq", &remaining, &V::i32(-1));
        self.cond_br(&unlimited, &enter, &spend, None);

        // The link that would exhaust the budget is not spent here: returning
        // TRACE_RESULT_LINK makes the dispatcher spend it (1 -> 0) and yield.
        // Spending it here as well would wrap 0 to u32::MAX (unlimited).
        self.begin(&spend);
        let exhausted = self.icmp("ule", &remaining, &V::i32(1));
        let decrement = self.new_block();
        self.cond_br(&exhausted, &yield_to_dispatcher, &decrement, Some(MD_COLD_TRUE));
        self.begin(&decrement);
        let left = self.bin("add", &remaining, &V::i32(-1));
        self.store_at(&left, &activation, remaining_offset, Mem::Act);
        self.br(&enter);

        self.begin(&enter);
        self.br(start);

        // Exactly what the dispatcher sees after a materialized link.
        self.begin(&yield_to_dispatcher);
        self.store_completed_iterations();
        self.store_at(&V::i32(child.0 as i32), &activation, offset_of!(TraceActivation, next_trace) as i64, Mem::Act);
        self.ret(TRACE_RESULT_LINK);
        Ok(())
    }
}

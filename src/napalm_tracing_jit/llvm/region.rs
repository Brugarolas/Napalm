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

/// One member of a region job: data cloned by the manager.
pub(in crate::napalm_tracing_jit) struct RegionMemberJob {
    pub(in crate::napalm_tracing_jit) trace: Trace,
    pub(in crate::napalm_tracing_jit) exits: Vec<CompiledExitPlan>,
    pub(in crate::napalm_tracing_jit) binding_slots: HashMap<VmBindingKey, u32>,
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
}

impl RegionMember {
    pub(super) fn start_of(&self, trace: TraceId) -> Option<String> {
        self.starts.get(&trace).cloned()
    }
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
        });
        state.lower_function(native_budget)?;
        previous = Some(state);
    }

    let mut state = previous.ok_or_else(|| "empty trace-tree region".to_owned())?;
    // The module is described by its head: entry count and profile summary.
    state.trace = &head.trace;
    state.profile = head.profile.as_ref().filter(|profile| profile.entries > 0);
    state.entry_branch_target = "m0_start".to_owned();
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

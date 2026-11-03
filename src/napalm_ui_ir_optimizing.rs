//! Optimization passes for Napalm UI IR.
//!
//! The default pipeline is deliberately aggressive about static UI structure but
//! conservative about semantics owned by WIR. A UI pass never executes arbitrary
//! Napalm code; compile-time binding values must arrive through
//! `UiBinding::constant` after WIR/lowering analysis.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::napalm_collections::{HashMap, HashSet};
use crate::napalm_ui_ir::*;
use crate::napalm_ui_ir_optimizer::{Optimize, UiOptimization};

const DEFAULT_INLINE_NODE_LIMIT: usize = 48;

pub fn optimization_passes() -> Optimize {
    Optimize::new()
        .with_pass(CanonicalizeSelectors)
        .with_pass(ResolveStaticTokens)
        .with_pass(DependencyDeduplication)
        .with_pass(ConstantBindingFold)
        .with_pass(BindingCSE)
        .with_pass(StaticConditionFold)
        .with_pass(DeadBranchElimination)
        .with_pass(FlattenFragments)
        .with_pass(MergeAdjacentText)
        .with_pass(KeyedLoopCanonicalization)
        .with_pass(ComponentSpecialization)
        .with_pass(ComponentInlining::default())
        .with_pass(LayoutConstantPropagation)
        .with_pass(BindingSchedulingClassification)
        .with_pass(PaintOnlyInvalidationAnalysis)
        .with_pass(DeadStyleRuleElimination)
        .with_pass(StyleCascadePrecompute)
        .with_pass(StaticStyleInterning)
        .with_pass(StaticSubtreeHoisting)
        .with_pass(ResourceDeduplication)
        .with_pass(CanvasDisplayListFolding)
        .with_pass(AnimationCanonicalization)
        .with_pass(DeadNodeElimination)
        // Inlining/dead-node cleanup can expose adjacent fragments/text again.
        .with_pass(FlattenFragments)
        .with_pass(MergeAdjacentText)
        .with_pass(DeadNodeElimination)
        .with_pass(ComponentSpecialization)
        .with_pass(StaticStyleInterning)
}

pub fn conservative_optimization_passes() -> Optimize {
    Optimize::new()
        .with_pass(CanonicalizeSelectors)
        .with_pass(ResolveStaticTokens)
        .with_pass(DependencyDeduplication)
        .with_pass(ConstantBindingFold)
        .with_pass(StaticConditionFold)
        .with_pass(DeadBranchElimination)
        .with_pass(FlattenFragments)
        .with_pass(LayoutConstantPropagation)
        .with_pass(BindingSchedulingClassification)
        .with_pass(PaintOnlyInvalidationAnalysis)
        .with_pass(DeadStyleRuleElimination)
        .with_pass(StyleCascadePrecompute)
        .with_pass(ResourceDeduplication)
        .with_pass(CanvasDisplayListFolding)
        .with_pass(AnimationCanonicalization)
        .with_pass(DeadNodeElimination)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BindingUseClass {
    Immediate,
    Structural,
    Layout,
    Paint,
}

#[derive(Clone, Copy, Debug, Default)]
struct BindingUseSummary {
    immediate: bool,
    structural: bool,
    layout: bool,
    paint: bool,
}

impl BindingUseSummary {
    fn record(&mut self, class: BindingUseClass) {
        match class {
            BindingUseClass::Immediate => self.immediate = true,
            BindingUseClass::Structural => self.structural = true,
            BindingUseClass::Layout => self.layout = true,
            BindingUseClass::Paint => self.paint = true,
        }
    }
}

#[derive(Default)]
struct StableHasher(u64);

impl StableHasher {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }
}

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}

fn stable_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = StableHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn hash_f32(hasher: &mut StableHasher, value: f32) {
    hasher.write_u32(value.to_bits());
}

fn hash_f64(hasher: &mut StableHasher, value: f64) {
    hasher.write_u64(value.to_bits());
}

fn hash_color(hasher: &mut StableHasher, value: UiColor) {
    hash_f32(hasher, value.r);
    hash_f32(hasher, value.g);
    hash_f32(hasher, value.b);
    hash_f32(hasher, value.a);
}

fn hash_color_ref(hasher: &mut StableHasher, value: &UiColorRef) {
    match value {
        UiColorRef::Literal(color) => {
            hasher.write_u8(0);
            hash_color(hasher, *color);
        }
        UiColorRef::Theme(color) => {
            hasher.write_u8(1);
            color.hash(hasher);
        }
    }
}

fn hash_runtime_value(hasher: &mut StableHasher, value: &UiRuntimeValue) {
    match value {
        UiRuntimeValue::Null => hasher.write_u8(0),
        UiRuntimeValue::Bool(value) => {
            hasher.write_u8(1);
            hasher.write_u8(*value as u8);
        }
        UiRuntimeValue::Number(value) => {
            hasher.write_u8(2);
            hash_f64(hasher, *value);
        }
        UiRuntimeValue::String(value) => {
            hasher.write_u8(3);
            value.hash(hasher);
        }
        UiRuntimeValue::Opaque(value) => {
            hasher.write_u8(4);
            hasher.write_u64(*value);
        }
    }
}

fn hash_constant(hasher: &mut StableHasher, value: &UiConstantValue) {
    match value {
        UiConstantValue::Null => hasher.write_u8(0),
        UiConstantValue::Bool(value) => {
            hasher.write_u8(1);
            hasher.write_u8(*value as u8);
        }
        UiConstantValue::Number(value) => {
            hasher.write_u8(2);
            hash_f64(hasher, *value);
        }
        UiConstantValue::String(value) => {
            hasher.write_u8(3);
            value.hash(hasher);
        }
        UiConstantValue::Color(value) => {
            hasher.write_u8(4);
            hash_color_ref(hasher, value);
        }
        UiConstantValue::Background(value) => {
            hasher.write_u8(5);
            hash_background(hasher, value);
        }
        UiConstantValue::Length(value) => {
            hasher.write_u8(6);
            hash_length(hasher, *value);
        }
        UiConstantValue::Padding(value) => {
            hasher.write_u8(7);
            hash_padding(hasher, *value);
        }
        UiConstantValue::Border(value) => {
            hasher.write_u8(8);
            hash_border(hasher, *value);
        }
        UiConstantValue::Shadow(value) => {
            hasher.write_u8(9);
            hash_shadow(hasher, *value);
        }
        UiConstantValue::Radius(value) => {
            hasher.write_u8(10);
            hash_radius(hasher, *value);
        }
        UiConstantValue::Transform(value) => {
            hasher.write_u8(11);
            hash_transform(hasher, *value);
        }
        UiConstantValue::Font(value) => {
            hasher.write_u8(12);
            value.hash(hasher);
        }
        UiConstantValue::Resource(value) => {
            hasher.write_u8(13);
            value.hash(hasher);
        }
        UiConstantValue::Theme(value) => {
            hasher.write_u8(14);
            value.hash(hasher);
        }
        UiConstantValue::Node(value) => {
            hasher.write_u8(15);
            value.hash(hasher);
        }
        UiConstantValue::Runtime(value) => {
            hasher.write_u8(16);
            hash_runtime_value(hasher, value);
        }
    }
}

fn hash_length(hasher: &mut StableHasher, value: UiLength) {
    match value {
        UiLength::Fill => hasher.write_u8(0),
        UiLength::FillPortion(value) => {
            hasher.write_u8(1);
            hasher.write_u16(value);
        }
        UiLength::Shrink => hasher.write_u8(2),
        UiLength::Pixels(value) => {
            hasher.write_u8(3);
            hash_f32(hasher, value);
        }
        UiLength::Percent(value) => {
            hasher.write_u8(4);
            hash_f32(hasher, value);
        }
        UiLength::MinContent => hasher.write_u8(5),
        UiLength::MaxContent => hasher.write_u8(6),
    }
}

fn hash_padding(hasher: &mut StableHasher, value: UiPadding) {
    hash_f32(hasher, value.top);
    hash_f32(hasher, value.right);
    hash_f32(hasher, value.bottom);
    hash_f32(hasher, value.left);
}

fn hash_radius(hasher: &mut StableHasher, value: UiRadius) {
    hash_f32(hasher, value.top_left);
    hash_f32(hasher, value.top_right);
    hash_f32(hasher, value.bottom_right);
    hash_f32(hasher, value.bottom_left);
}

fn hash_border(hasher: &mut StableHasher, value: UiBorder) {
    hash_color_ref(hasher, &value.color);
    hash_f32(hasher, value.width);
    hash_radius(hasher, value.radius);
}

fn hash_shadow(hasher: &mut StableHasher, value: UiShadow) {
    hash_color_ref(hasher, &value.color);
    hash_f32(hasher, value.offset.x);
    hash_f32(hasher, value.offset.y);
    hash_f32(hasher, value.blur_radius);
}

fn hash_transform(hasher: &mut StableHasher, value: UiTransform) {
    hash_f32(hasher, value.translate.x);
    hash_f32(hasher, value.translate.y);
    hash_f32(hasher, value.scale.x);
    hash_f32(hasher, value.scale.y);
    hash_f32(hasher, value.rotation_radians);
    hash_f32(hasher, value.origin.x);
    hash_f32(hasher, value.origin.y);
}

fn hash_background(hasher: &mut StableHasher, value: &UiBackground) {
    match value {
        UiBackground::Color(color) => {
            hasher.write_u8(0);
            hash_color_ref(hasher, color);
        }
        UiBackground::LinearGradient(gradient) => {
            hasher.write_u8(1);
            hash_f32(hasher, gradient.angle_radians);
            gradient.interpolation.hash(hasher);
            hasher.write_usize(gradient.stops.len());
            for stop in &gradient.stops {
                hash_f32(hasher, stop.offset);
                hash_color_ref(hasher, &stop.color);
            }
        }
        UiBackground::RadialGradient(gradient) => {
            hasher.write_u8(2);
            hash_f32(hasher, gradient.center.x);
            hash_f32(hasher, gradient.center.y);
            hash_f32(hasher, gradient.radius);
            gradient.interpolation.hash(hasher);
            hasher.write_usize(gradient.stops.len());
            for stop in &gradient.stops {
                hash_f32(hasher, stop.offset);
                hash_color_ref(hasher, &stop.color);
            }
        }
    }
}

fn binding_fingerprint(binding: &UiBinding) -> u64 {
    let mut hasher = StableHasher::new();
    binding.function.hash(&mut hasher);
    binding.result.hash(&mut hasher);
    binding.dependencies.hash(&mut hasher);
    binding.equality.hash(&mut hasher);
    // Scheduling and invalidation flags are derived from the union of call sites
    // later in the pipeline. Keeping them out of the key lets CSE merge one pure
    // expression used by both layout and paint; the classification passes then
    // conservatively promote the canonical binding to the strongest requirement.
    hasher.write_u8(binding.flags.pure as u8);
    hasher.write_u8(binding.flags.memoizable as u8);
    if let Some(constant) = &binding.constant {
        hasher.write_u8(1);
        hash_constant(&mut hasher, constant);
    } else {
        hasher.write_u8(0);
    }
    hasher.finish()
}

fn binding_equivalent(left: &UiBinding, right: &UiBinding) -> bool {
    left.function == right.function
        && left.result == right.result
        && left.dependencies == right.dependencies
        && left.equality == right.equality
        && left.flags.pure == right.flags.pure
        && left.flags.memoizable == right.flags.memoizable
        && left.constant == right.constant
}

// =============================================================================
// Pass declarations
// =============================================================================

#[derive(Default)]
pub struct CanonicalizeSelectors;
#[derive(Default)]
pub struct ResolveStaticTokens;
#[derive(Default)]
pub struct ConstantBindingFold;
#[derive(Default)]
pub struct StaticConditionFold;
#[derive(Default)]
pub struct DeadBranchElimination;
#[derive(Default)]
pub struct FlattenFragments;
#[derive(Default)]
pub struct MergeAdjacentText;
#[derive(Default)]
pub struct StyleCascadePrecompute;
#[derive(Default)]
pub struct DeadStyleRuleElimination;
#[derive(Default)]
pub struct StaticStyleInterning;
#[derive(Default)]
pub struct DependencyDeduplication;
#[derive(Default)]
pub struct BindingCSE;
#[derive(Default)]
pub struct BindingSchedulingClassification;
#[derive(Default)]
pub struct StaticSubtreeHoisting;
#[derive(Default)]
pub struct KeyedLoopCanonicalization;
#[derive(Default)]
pub struct ComponentSpecialization;

pub struct ComponentInlining {
    pub max_nodes: usize,
}

impl Default for ComponentInlining {
    fn default() -> Self {
        Self { max_nodes: DEFAULT_INLINE_NODE_LIMIT }
    }
}

#[derive(Default)]
pub struct LayoutConstantPropagation;
#[derive(Default)]
pub struct PaintOnlyInvalidationAnalysis;
#[derive(Default)]
pub struct ResourceDeduplication;
#[derive(Default)]
pub struct CanvasDisplayListFolding;
#[derive(Default)]
pub struct AnimationCanonicalization;
#[derive(Default)]
pub struct DeadNodeElimination;

// =============================================================================
// Selector/media canonicalization
// =============================================================================

fn pseudo_rank(value: UiPseudoClass) -> u8 {
    match value {
        UiPseudoClass::Active => 0,
        UiPseudoClass::Hover => 1,
        UiPseudoClass::Pressed => 2,
        UiPseudoClass::Disabled => 3,
        UiPseudoClass::Focus => 4,
        UiPseudoClass::FocusHover => 5,
        UiPseudoClass::Checked => 6,
        UiPseudoClass::Unchecked => 7,
        UiPseudoClass::Selected => 8,
        UiPseudoClass::Unselected => 9,
        UiPseudoClass::Dragged => 10,
        UiPseudoClass::Open => 11,
        UiPseudoClass::Closed => 12,
        UiPseudoClass::Invalid => 13,
        UiPseudoClass::PlaceholderShown => 14,
        UiPseudoClass::FirstChild => 15,
        UiPseudoClass::LastChild => 16,
        UiPseudoClass::Odd => 17,
        UiPseudoClass::Even => 18,
    }
}

fn property_rank(value: UiAnimatableProperty) -> u8 {
    match value {
        UiAnimatableProperty::Opacity => 0,
        UiAnimatableProperty::Translate => 1,
        UiAnimatableProperty::Scale => 2,
        UiAnimatableProperty::Rotation => 3,
        UiAnimatableProperty::Width => 4,
        UiAnimatableProperty::Height => 5,
        UiAnimatableProperty::Padding => 6,
        UiAnimatableProperty::BackgroundColor => 7,
        UiAnimatableProperty::ForegroundColor => 8,
        UiAnimatableProperty::BorderColor => 9,
        UiAnimatableProperty::BorderWidth => 10,
        UiAnimatableProperty::BorderRadius => 11,
        UiAnimatableProperty::Shadow => 12,
        UiAnimatableProperty::TextColor => 13,
        UiAnimatableProperty::FontSize => 14,
        UiAnimatableProperty::SliderValue => 15,
        UiAnimatableProperty::ScrollX => 16,
        UiAnimatableProperty::ScrollY => 17,
        UiAnimatableProperty::ImageScale => 18,
        UiAnimatableProperty::ImageOpacity => 19,
    }
}

fn canonicalize_selector(selector: &mut UiSelector) -> bool {
    let before = format!("{selector:?}");

    if let Some(first) = selector.chain.first_mut() {
        first.combinator = None;
    }
    for part in selector.chain.iter_mut().skip(1) {
        if part.combinator.is_none() {
            part.combinator = Some(UiSelectorCombinator::Descendant);
        }
    }

    for part in &mut selector.chain {
        part.atom.classes.sort_unstable_by_key(|id| id.0);
        part.atom.classes.dedup();
        part.atom.states.sort_unstable_by_key(|state| pseudo_rank(*state));
        part.atom.states.dedup();
        if part.atom.widget == Some(UiWidgetTag::Any) {
            part.atom.widget = None;
        }
    }

    // Consecutive universal descendant selectors carry no information.
    let mut compact = Vec::with_capacity(selector.chain.len());
    for part in selector.chain.drain(..) {
        let empty_atom = part.atom.widget.is_none()
            && part.atom.name.is_none()
            && part.atom.classes.is_empty()
            && part.atom.states.is_empty()
            && part.atom.part.is_none();
        let redundant = empty_atom
            && part.combinator == Some(UiSelectorCombinator::Descendant)
            && compact.last().is_some_and(|previous: &UiSelectorPart| {
                previous.atom.widget.is_none()
                    && previous.atom.name.is_none()
                    && previous.atom.classes.is_empty()
                    && previous.atom.states.is_empty()
                    && previous.atom.part.is_none()
            });
        if !redundant {
            compact.push(part);
        }
    }
    selector.chain = compact;

    before != format!("{selector:?}")
}

fn selector_specificity(selector: &UiSelector) -> u32 {
    let mut ids = 0u32;
    let mut classes = 0u32;
    let mut elements = 0u32;

    for part in &selector.chain {
        ids += u32::from(part.atom.name.is_some());
        classes += part.atom.classes.len() as u32;
        classes += part.atom.states.len() as u32;
        elements += u32::from(part.atom.widget.is_some());
        elements += u32::from(part.atom.part.is_some());
    }

    // Packed CSS-like specificity, saturating each bucket to keep comparison O(1).
    (ids.min(0x3ff) << 20) | (classes.min(0x3ff) << 10) | elements.min(0x3ff)
}

fn simplify_media(condition: &mut UiMediaCondition) -> bool {
    let before = format!("{condition:?}");
    let old = std::mem::replace(condition, UiMediaCondition::True);
    *condition = simplify_media_owned(old);
    before != format!("{condition:?}")
}

fn simplify_media_owned(condition: UiMediaCondition) -> UiMediaCondition {
    match condition {
        UiMediaCondition::Not(inner) => match simplify_media_owned(*inner) {
            UiMediaCondition::Not(nested) => *nested,
            other => UiMediaCondition::Not(Box::new(other)),
        },
        UiMediaCondition::All(items) => {
            let mut out = Vec::new();
            for item in items {
                match simplify_media_owned(item) {
                    UiMediaCondition::True => {}
                    UiMediaCondition::All(nested) => out.extend(nested),
                    other => out.push(other),
                }
            }
            match out.len() {
                0 => UiMediaCondition::True,
                1 => out.pop().unwrap(),
                _ => UiMediaCondition::All(out),
            }
        }
        UiMediaCondition::Any(items) => {
            let mut out = Vec::new();
            for item in items {
                match simplify_media_owned(item) {
                    UiMediaCondition::True => return UiMediaCondition::True,
                    UiMediaCondition::Any(nested) => out.extend(nested),
                    other => out.push(other),
                }
            }
            match out.len() {
                0 => UiMediaCondition::Any(Vec::new()),
                1 => out.pop().unwrap(),
                _ => UiMediaCondition::Any(out),
            }
        }
        other => other,
    }
}

impl UiOptimization for CanonicalizeSelectors {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for sheet in &mut program.style_sheets {
            for (order, rule) in sheet.rules.iter_mut().enumerate() {
                changed |= canonicalize_selector(&mut rule.selector);
                let specificity = selector_specificity(&rule.selector);
                if rule.specificity != specificity {
                    rule.specificity = specificity;
                    changed = true;
                }
                if rule.order != order as u32 {
                    rule.order = order as u32;
                    changed = true;
                }
                if let Some(media) = &mut rule.media {
                    changed |= simplify_media(media);
                    if matches!(media, UiMediaCondition::True) {
                        rule.media = None;
                        changed = true;
                    }
                }
                let original_transitions = std::mem::take(&mut rule.transitions);
                let before_transitions = original_transitions.clone();
                let mut canonical_transitions: Vec<UiTransition> = Vec::with_capacity(original_transitions.len());
                for transition in original_transitions {
                    if let Some(existing) = canonical_transitions
                        .iter_mut()
                        .find(|candidate| candidate.property == transition.property)
                    {
                        *existing = transition;
                    } else {
                        canonical_transitions.push(transition);
                    }
                }
                canonical_transitions.sort_by_key(|transition| property_rank(transition.property));
                changed |= before_transitions != canonical_transitions;
                rule.transitions = canonical_transitions;
            }
        }
        for animation in &mut program.animations {
            for track in &mut animation.tracks {
                if let UiAnimationTarget::Selector(selector) = &mut track.target {
                    changed |= canonicalize_selector(selector);
                }
            }
        }
        changed
    }
}

// =============================================================================
// Static style token resolution
// =============================================================================

fn resolve_token_literal(
    tokens: &[UiStyleToken],
    id: UiStyleTokenId,
    visiting: &mut HashSet<UiStyleTokenId>,
    memo: &mut HashMap<UiStyleTokenId, UiStyleLiteral>,
) -> Option<UiStyleLiteral> {
    if let Some(value) = memo.get(&id) {
        return Some(value.clone());
    }
    if !visiting.insert(id) {
        return None;
    }
    let token = tokens.get(id.0 as usize)?;
    let value = match &token.value {
        UiStyleLiteral::Token(next) => resolve_token_literal(tokens, *next, visiting, memo)?,
        other => other.clone(),
    };
    visiting.remove(&id);
    memo.insert(id, value.clone());
    Some(value)
}

impl UiOptimization for ResolveStaticTokens {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let snapshot = program.style_tokens.clone();
        let mut memo = HashMap::default();
        let mut changed = false;
        for index in 0..program.style_tokens.len() {
            let id = UiStyleTokenId(index as u32);
            let mut visiting = HashSet::default();
            if let Some(resolved) = resolve_token_literal(&snapshot, id, &mut visiting, &mut memo) {
                if program.style_tokens[index].value != resolved {
                    program.style_tokens[index].value = resolved;
                    changed = true;
                }
            }
        }
        changed
    }
}

// =============================================================================
// Dynamic-reference traversal shared by CSE/scheduling/token remapping
// =============================================================================

fn visit_value_ref<T>(
    value: &mut UiValue<T>,
    class: BindingUseClass,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    match value {
        UiValue::Static(_) => {}
        UiValue::Binding(id) => on_binding(id, class),
        UiValue::Token(id) => on_token(id),
    }
}

fn visit_opt_value_ref<T>(
    value: &mut Option<UiValue<T>>,
    class: BindingUseClass,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    if let Some(value) = value {
        visit_value_ref(value, class, on_binding, on_token);
    }
}

fn visit_optional_style_value_ref<T>(
    value: &mut Option<UiOptionalStyleValue<T>>,
    class: BindingUseClass,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    if let Some(UiOptionalStyleValue::Value(value)) = value {
        visit_value_ref(value, class, on_binding, on_token);
    }
}

fn visit_style_patch_dynamic_refs_mut(
    patch: &mut UiStylePatch,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    let common = &mut patch.common;
    visit_optional_style_value_ref(&mut common.background, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.foreground, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.border, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.shadow, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.radius, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.opacity, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.cursor, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut common.snap, BindingUseClass::Paint, on_binding, on_token);

    let text = &mut patch.text;
    visit_optional_style_value_ref(&mut text.color, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut text.font, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut text.size, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut text.weight, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut text.line_height, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut text.letter_spacing, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut text.underline, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut text.strikethrough, BindingUseClass::Paint, on_binding, on_token);

    let Some(widget) = &mut patch.widget else { return; };
    match widget {
        UiWidgetStylePatch::Application(style) => {
            visit_opt_value_ref(&mut style.background_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Container(style) => {
            visit_optional_style_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_optional_style_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.shadow, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.snap, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Float(style) => {
            visit_opt_value_ref(&mut style.shadow, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.shadow_border_radius, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Button(style) => {
            visit_optional_style_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.shadow, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.snap, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Checkbox(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.icon_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_optional_style_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Radio(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.dot_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border_width, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border_color, BindingUseClass::Paint, on_binding, on_token);
            visit_optional_style_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Toggler(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.background_border_width, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.background_border_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.foreground, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.foreground_border_width, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.foreground_border_color, BindingUseClass::Paint, on_binding, on_token);
            visit_optional_style_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_optional_style_value_ref(&mut style.border_radius, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.padding_ratio, BindingUseClass::Layout, on_binding, on_token);
        }
        UiWidgetStylePatch::Text(style) => {
            visit_optional_style_value_ref(&mut style.color, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::TextInput(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.icon, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.placeholder, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.value, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.selection, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::TextEditor(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.placeholder, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.value, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.selection, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::PickList(style) => {
            visit_opt_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.placeholder_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.handle_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Menu(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.selected_text_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.selected_background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.shadow, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Slider(style) => {
            if let Some(rail) = &mut style.rail {
                visit_value_ref(&mut rail.filled, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut rail.unfilled, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut rail.width, BindingUseClass::Layout, on_binding, on_token);
                visit_value_ref(&mut rail.border, BindingUseClass::Paint, on_binding, on_token);
            }
            if let Some(handle) = &mut style.handle {
                visit_value_ref(&mut handle.shape, BindingUseClass::Layout, on_binding, on_token);
                visit_value_ref(&mut handle.background, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut handle.border_width, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut handle.border_color, BindingUseClass::Paint, on_binding, on_token);
            }
        }
        UiWidgetStylePatch::Scrollable(style) => {
            if let Some(container) = &mut style.container {
                visit_optional_style_value_ref(&mut container.text_color, BindingUseClass::Paint, on_binding, on_token);
                visit_optional_style_value_ref(&mut container.background, BindingUseClass::Paint, on_binding, on_token);
                visit_opt_value_ref(&mut container.border, BindingUseClass::Paint, on_binding, on_token);
                visit_opt_value_ref(&mut container.shadow, BindingUseClass::Paint, on_binding, on_token);
                visit_opt_value_ref(&mut container.snap, BindingUseClass::Paint, on_binding, on_token);
            }
            for rail in [&mut style.vertical_rail, &mut style.horizontal_rail] {
                if let Some(rail) = rail {
                    visit_opt_value_ref(&mut rail.background, BindingUseClass::Paint, on_binding, on_token);
                    visit_value_ref(&mut rail.border, BindingUseClass::Paint, on_binding, on_token);
                    visit_value_ref(&mut rail.scroller.background, BindingUseClass::Paint, on_binding, on_token);
                    visit_value_ref(&mut rail.scroller.border, BindingUseClass::Paint, on_binding, on_token);
                }
            }
            visit_optional_style_value_ref(&mut style.gap, BindingUseClass::Paint, on_binding, on_token);
            if let Some(auto) = &mut style.auto_scroll {
                visit_value_ref(&mut auto.background, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut auto.border, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut auto.shadow, BindingUseClass::Paint, on_binding, on_token);
            }
        }
        UiWidgetStylePatch::ProgressBar(style) => {
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.bar, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.border, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Rule(style) => {
            visit_opt_value_ref(&mut style.color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.radius, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.fill_mode, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.snap, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Svg(style) => {
            visit_optional_style_value_ref(&mut style.color, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::PaneGrid(style) => {
            if let Some(region) = &mut style.hovered_region {
                visit_value_ref(&mut region.background, BindingUseClass::Paint, on_binding, on_token);
                visit_value_ref(&mut region.border, BindingUseClass::Paint, on_binding, on_token);
            }
            for line in [&mut style.picked_split, &mut style.hovered_split] {
                if let Some(line) = line {
                    visit_value_ref(&mut line.color, BindingUseClass::Paint, on_binding, on_token);
                    visit_value_ref(&mut line.width, BindingUseClass::Paint, on_binding, on_token);
                }
            }
        }
        UiWidgetStylePatch::Table(style) => {
            visit_opt_value_ref(&mut style.separator_x, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.separator_y, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::QrCode(style) => {
            visit_opt_value_ref(&mut style.cell, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.background, BindingUseClass::Paint, on_binding, on_token);
        }
        UiWidgetStylePatch::Markdown(style) => {
            visit_opt_value_ref(&mut style.font, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut style.inline_code_background, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.inline_code_padding, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut style.inline_code_color, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut style.inline_code_font, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut style.code_block_font, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut style.link_color, BindingUseClass::Paint, on_binding, on_token);
        }
    }
}

fn visit_text_format_dynamic_refs_mut(
    format: &mut UiTextFormat,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    visit_value_ref(&mut format.width, BindingUseClass::Layout, on_binding, on_token);
    visit_value_ref(&mut format.height, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut format.size, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut format.font, BindingUseClass::Layout, on_binding, on_token);
}

fn visit_layout_dynamic_refs_mut(
    layout: &mut UiLayout,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    visit_value_ref(&mut layout.width, BindingUseClass::Layout, on_binding, on_token);
    visit_value_ref(&mut layout.height, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut layout.min_width, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut layout.min_height, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut layout.max_width, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut layout.max_height, BindingUseClass::Layout, on_binding, on_token);
    visit_value_ref(&mut layout.padding, BindingUseClass::Layout, on_binding, on_token);
    visit_value_ref(&mut layout.margin, BindingUseClass::Layout, on_binding, on_token);
    visit_opt_value_ref(&mut layout.aspect_ratio, BindingUseClass::Layout, on_binding, on_token);
}

fn visit_accessibility_dynamic_refs_mut(
    accessibility: &mut UiAccessibility,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    visit_opt_value_ref(&mut accessibility.name, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.description, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.value, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.placeholder, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.checked, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.selected, BindingUseClass::Paint, on_binding, on_token);
    visit_opt_value_ref(&mut accessibility.expanded, BindingUseClass::Paint, on_binding, on_token);
}

fn visit_node_kind_dynamic_refs_mut(
    kind: &mut UiNodeKind,
    on_binding: &mut impl FnMut(&mut UiBindingId, BindingUseClass),
    on_token: &mut impl FnMut(&mut UiStyleTokenId),
) {
    match kind {
        UiNodeKind::Fragment
        | UiNodeKind::Stack(_)
        | UiNodeKind::Opaque
        | UiNodeKind::Canvas(_)
        | UiNodeKind::Slot(_)
        | UiNodeKind::Portal(_)
        | UiNodeKind::Custom(_) => {}
        UiNodeKind::Container(widget) => {
            visit_opt_value_ref(&mut widget.max_width, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.max_height, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Row(widget) | UiNodeKind::Column(widget) => {
            visit_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.padding, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Grid(widget) => {
            visit_value_ref(&mut widget.columns, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.row_spacing, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.column_spacing, BindingUseClass::Layout, on_binding, on_token);
            match &mut widget.sizing {
                UiGridSizing::AspectRatio(value) => visit_value_ref(value, BindingUseClass::Layout, on_binding, on_token),
                UiGridSizing::EvenlyDistribute(value) => visit_value_ref(value, BindingUseClass::Layout, on_binding, on_token),
                UiGridSizing::Tracks(_) => {}
            }
        }
        UiNodeKind::Space(widget) => {
            visit_value_ref(&mut widget.width, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.height, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Rule(widget) => {
            visit_value_ref(&mut widget.thickness, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Text(widget) => {
            visit_value_ref(&mut widget.content, BindingUseClass::Layout, on_binding, on_token);
            visit_text_format_dynamic_refs_mut(&mut widget.format, on_binding, on_token);
        }
        UiNodeKind::RichText(widget) => {
            for span in &mut widget.spans {
                visit_value_ref(&mut span.text, BindingUseClass::Layout, on_binding, on_token);
                visit_opt_value_ref(&mut span.font, BindingUseClass::Layout, on_binding, on_token);
                visit_opt_value_ref(&mut span.size, BindingUseClass::Layout, on_binding, on_token);
                visit_opt_value_ref(&mut span.color, BindingUseClass::Paint, on_binding, on_token);
                visit_opt_value_ref(&mut span.background, BindingUseClass::Paint, on_binding, on_token);
                visit_opt_value_ref(&mut span.link, BindingUseClass::Immediate, on_binding, on_token);
            }
            visit_text_format_dynamic_refs_mut(&mut widget.format, on_binding, on_token);
        }
        UiNodeKind::Markdown(widget) => {
            visit_value_ref(&mut widget.source, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.text_size, BindingUseClass::Layout, on_binding, on_token);
            for value in [
                &mut widget.settings.h1_size,
                &mut widget.settings.h2_size,
                &mut widget.settings.h3_size,
                &mut widget.settings.h4_size,
                &mut widget.settings.h5_size,
                &mut widget.settings.h6_size,
                &mut widget.settings.code_size,
                &mut widget.settings.spacing,
            ] {
                visit_opt_value_ref(value, BindingUseClass::Layout, on_binding, on_token);
            }
        }
        UiNodeKind::Button(_) => {}
        UiNodeKind::Checkbox(widget) => {
            visit_value_ref(&mut widget.checked, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut widget.label, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.size, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Radio(widget) => {
            visit_value_ref(&mut widget.value, BindingUseClass::Immediate, on_binding, on_token);
            visit_value_ref(&mut widget.selected, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.label, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.size, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
            visit_text_format_dynamic_refs_mut(&mut widget.text, on_binding, on_token);
        }
        UiNodeKind::Toggler(widget) => {
            visit_value_ref(&mut widget.checked, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut widget.label, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.size, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
            visit_text_format_dynamic_refs_mut(&mut widget.text, on_binding, on_token);
        }
        UiNodeKind::TextInput(widget) => {
            visit_value_ref(&mut widget.value, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.placeholder, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.padding, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.size, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.font, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::TextEditor(widget) => {
            visit_value_ref(&mut widget.content, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.placeholder, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.padding, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.height, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.font, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.size, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Slider(widget) | UiNodeKind::VerticalSlider(widget) => {
            visit_value_ref(&mut widget.min, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.max, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.value, BindingUseClass::Paint, on_binding, on_token);
            visit_opt_value_ref(&mut widget.step, BindingUseClass::Immediate, on_binding, on_token);
            visit_opt_value_ref(&mut widget.shift_step, BindingUseClass::Immediate, on_binding, on_token);
        }
        UiNodeKind::PickList(widget) => {
            visit_value_ref(&mut widget.options, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.selected, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.placeholder, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.text_size, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.padding, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.menu_height, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::ComboBox(widget) => {
            visit_value_ref(&mut widget.options, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.value, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.placeholder, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.selected, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.menu_height, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::ProgressBar(widget) => {
            visit_value_ref(&mut widget.min, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.max, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.value, BindingUseClass::Paint, on_binding, on_token);
        }
        UiNodeKind::Scrollable(widget) => {
            visit_opt_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
            for bar in [&mut widget.horizontal, &mut widget.vertical] {
                if let Some(bar) = bar {
                    visit_value_ref(&mut bar.visible, BindingUseClass::Layout, on_binding, on_token);
                    visit_value_ref(&mut bar.width, BindingUseClass::Layout, on_binding, on_token);
                    visit_value_ref(&mut bar.margin, BindingUseClass::Layout, on_binding, on_token);
                    visit_value_ref(&mut bar.scroller_width, BindingUseClass::Layout, on_binding, on_token);
                    visit_opt_value_ref(&mut bar.spacing, BindingUseClass::Layout, on_binding, on_token);
                }
            }
        }
        UiNodeKind::PaneGrid(widget) => {
            visit_value_ref(&mut widget.spacing, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.resize_leeway, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Table(widget) => {
            visit_value_ref(&mut widget.rows, BindingUseClass::Structural, on_binding, on_token);
        }
        UiNodeKind::Tooltip(widget) => {
            visit_value_ref(&mut widget.gap, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.padding, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.delay_ms, BindingUseClass::Immediate, on_binding, on_token);
        }
        UiNodeKind::MouseArea(_) | UiNodeKind::Hover(_) => {}
        UiNodeKind::Float(widget) => {
            visit_value_ref(&mut widget.scale, BindingUseClass::Paint, on_binding, on_token);
            if let UiFloatTranslation::Static(value) = &mut widget.translation {
                visit_value_ref(value, BindingUseClass::Paint, on_binding, on_token);
            }
        }
        UiNodeKind::Pin(widget) => {
            visit_opt_value_ref(&mut widget.x, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.y, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Responsive(_) | UiNodeKind::Sensor(_) => {}
        UiNodeKind::Themer(widget) => {
            visit_value_ref(&mut widget.theme, BindingUseClass::Structural, on_binding, on_token);
        }
        UiNodeKind::Lazy(widget) => on_binding(&mut widget.dependency, BindingUseClass::Structural),
        UiNodeKind::Image(widget) => {
            visit_value_ref(&mut widget.source, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.rotation, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.opacity, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.scale, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.border_radius, BindingUseClass::Paint, on_binding, on_token);
        }
        UiNodeKind::Svg(widget) => {
            visit_value_ref(&mut widget.source, BindingUseClass::Layout, on_binding, on_token);
            visit_value_ref(&mut widget.rotation, BindingUseClass::Paint, on_binding, on_token);
            visit_value_ref(&mut widget.opacity, BindingUseClass::Paint, on_binding, on_token);
        }
        UiNodeKind::Shader(widget) => {
            for uniform in &mut widget.uniforms {
                on_binding(&mut uniform.value, BindingUseClass::Paint);
            }
        }
        UiNodeKind::QrCode(widget) => {
            visit_value_ref(&mut widget.data, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.cell_size, BindingUseClass::Layout, on_binding, on_token);
            visit_opt_value_ref(&mut widget.quiet_zone, BindingUseClass::Layout, on_binding, on_token);
        }
        UiNodeKind::Component(widget) => {
            for (_, binding) in &mut widget.props {
                on_binding(binding, BindingUseClass::Structural);
            }
        }
        UiNodeKind::If(widget) => on_binding(&mut widget.condition, BindingUseClass::Structural),
        UiNodeKind::Switch(widget) => on_binding(&mut widget.scrutinee, BindingUseClass::Structural),
        UiNodeKind::ForEach(widget) => on_binding(&mut widget.collection, BindingUseClass::Structural),
    }
}

fn visit_component_dynamic_refs_mut(
    component: &mut UiComponent,
    mut on_binding: impl FnMut(&mut UiBindingId, BindingUseClass),
    mut on_token: impl FnMut(&mut UiStyleTokenId),
) {
    if let Some(theme) = &mut component.theme {
        visit_value_ref(theme, BindingUseClass::Structural, &mut on_binding, &mut on_token);
    }

    for property in &mut component.properties {
        if let Some(binding) = &mut property.default {
            on_binding(binding, BindingUseClass::Immediate);
        }
    }
    for state in &mut component.states {
        on_binding(&mut state.initial, BindingUseClass::Immediate);
    }

    for window in &mut component.windows {
        visit_value_ref(&mut window.title, BindingUseClass::Layout, &mut on_binding, &mut on_token);
        if let Some(theme) = &mut window.theme {
            visit_value_ref(theme, BindingUseClass::Structural, &mut on_binding, &mut on_token);
        }
        let settings = &mut window.settings;
        visit_value_ref(&mut settings.size, BindingUseClass::Layout, &mut on_binding, &mut on_token);
        visit_opt_value_ref(&mut settings.min_size, BindingUseClass::Layout, &mut on_binding, &mut on_token);
        visit_opt_value_ref(&mut settings.max_size, BindingUseClass::Layout, &mut on_binding, &mut on_token);
        if let UiWindowPosition::Specific(position) = &mut settings.position {
            visit_value_ref(position, BindingUseClass::Layout, &mut on_binding, &mut on_token);
        }
        for value in [
            &mut settings.visible,
            &mut settings.maximized,
            &mut settings.fullscreen,
            &mut settings.resizable,
            &mut settings.closeable,
            &mut settings.minimizable,
            &mut settings.decorations,
            &mut settings.transparent,
            &mut settings.blur,
        ] {
            visit_value_ref(value, BindingUseClass::Structural, &mut on_binding, &mut on_token);
        }
    }

    for node in &mut component.nodes {
        if let Some(key) = &mut node.key {
            visit_value_ref(key, BindingUseClass::Structural, &mut on_binding, &mut on_token);
        }
        visit_layout_dynamic_refs_mut(&mut node.layout, &mut on_binding, &mut on_token);
        visit_style_patch_dynamic_refs_mut(&mut node.style.inline, &mut on_binding, &mut on_token);
        visit_value_ref(&mut node.visible, BindingUseClass::Structural, &mut on_binding, &mut on_token);
        visit_value_ref(&mut node.enabled, BindingUseClass::Paint, &mut on_binding, &mut on_token);
        visit_value_ref(&mut node.opacity, BindingUseClass::Paint, &mut on_binding, &mut on_token);
        visit_value_ref(&mut node.transform, BindingUseClass::Paint, &mut on_binding, &mut on_token);
        match &mut node.clip {
            UiClip::Rounded(radius) => visit_value_ref(radius, BindingUseClass::Paint, &mut on_binding, &mut on_token),
            UiClip::None | UiClip::Bounds | UiClip::Path(_) => {}
        }
        visit_opt_value_ref(&mut node.cursor, BindingUseClass::Paint, &mut on_binding, &mut on_token);
        if let Some(accessibility) = &mut node.accessibility {
            visit_accessibility_dynamic_refs_mut(accessibility, &mut on_binding, &mut on_token);
        }
        visit_node_kind_dynamic_refs_mut(&mut node.kind, &mut on_binding, &mut on_token);
    }
}

fn visit_program_token_refs_mut(program: &mut UiProgram, mut f: impl FnMut(&mut UiStyleTokenId)) {
    for component in &mut program.components {
        visit_component_dynamic_refs_mut(component, |_, _| {}, |token| f(token));
    }
    for sheet in &mut program.style_sheets {
        for rule in &mut sheet.rules {
            visit_style_patch_dynamic_refs_mut(&mut rule.declarations, &mut |_, _| {}, &mut |token| f(token));
        }
    }
    for style in &mut program.static_styles {
        visit_style_patch_dynamic_refs_mut(&mut style.patch, &mut |_, _| {}, &mut |token| f(token));
    }
    for token in &mut program.style_tokens {
        if let UiStyleLiteral::Token(id) = &mut token.value {
            f(id);
        }
    }
}

// =============================================================================
// Binding canonicalization and CSE
// =============================================================================

fn dependency_sort_key(dependency: &UiDependency) -> (u8, u64) {
    match dependency {
        UiDependency::Property(id) => (0, u64::from(id.0)),
        UiDependency::State(id) => (1, u64::from(id.0)),
        UiDependency::Computed(id) => (2, u64::from(id.0)),
        UiDependency::Local(id) => (3, u64::from(id.0)),
        UiDependency::StyleToken(id) => (4, u64::from(id.0)),
        UiDependency::Theme => (5, 0),
        UiDependency::WindowSize(id) => (6, u64::from(id.0)),
        UiDependency::WindowScaleFactor(id) => (7, u64::from(id.0)),
        UiDependency::Environment(key) => (8, *key as u64),
        UiDependency::Resource(id) => (9, u64::from(id.0)),
    }
}

fn canonicalize_dependencies(dependencies: &mut Vec<UiDependency>) -> bool {
    let old = dependencies.clone();
    dependencies.sort_unstable_by_key(dependency_sort_key);
    dependencies.dedup();
    *dependencies != old
}

impl UiOptimization for DependencyDeduplication {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            for binding in &mut component.bindings {
                changed |= canonicalize_dependencies(&mut binding.dependencies);
            }
            for computed in &mut component.computed {
                changed |= canonicalize_dependencies(&mut computed.dependencies);
            }
        }
        changed
    }
}

impl UiOptimization for ConstantBindingFold {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            for binding in &mut component.bindings {
                if binding.constant.is_none() {
                    continue;
                }
                if !binding.dependencies.is_empty() {
                    binding.dependencies.clear();
                    changed = true;
                }
                if !binding.flags.pure {
                    binding.flags.pure = true;
                    changed = true;
                }
                if !binding.flags.memoizable {
                    binding.flags.memoizable = true;
                    changed = true;
                }
                if binding.equality != UiEqualityPolicy::Value {
                    binding.equality = UiEqualityPolicy::Value;
                    changed = true;
                }
            }
        }
        changed
    }
}

fn visit_owned_style_sheet_binding_refs_mut(
    program: &mut UiProgram,
    owner: UiComponentId,
    mut f: impl FnMut(&mut UiBindingId, BindingUseClass),
) {
    for sheet in &mut program.style_sheets {
        if sheet.owner != Some(owner) {
            continue;
        }
        for rule in &mut sheet.rules {
            let mut visit_binding = |binding: &mut UiBindingId, class: BindingUseClass| {
                f(binding, class);
            };
            let mut ignore_token = |_token: &mut UiStyleTokenId| {};
            visit_style_patch_dynamic_refs_mut(
                &mut rule.declarations,
                &mut visit_binding,
                &mut ignore_token,
            );
        }
    }
}

fn remap_owned_style_sheet_binding_refs(
    program: &mut UiProgram,
    owner: UiComponentId,
    map: &[UiBindingId],
) {
    visit_owned_style_sheet_binding_refs_mut(program, owner, |id, _| {
        if let Some(mapped) = map.get(id.0 as usize) {
            *id = *mapped;
        }
    });
}

fn remap_component_binding_refs(component: &mut UiComponent, map: &[UiBindingId]) {
    visit_component_dynamic_refs_mut(
        component,
        |id, _| {
            if let Some(mapped) = map.get(id.0 as usize) {
                *id = *mapped;
            }
        },
        |_| {},
    );
}

fn compact_bindings(program: &mut UiProgram, component_index: usize) -> bool {
    let component_id = program.components[component_index].id;
    let binding_len = program.components[component_index].bindings.len();
    if binding_len == 0 {
        return false;
    }

    let mut used = vec![false; binding_len];
    {
        let component = &mut program.components[component_index];
        visit_component_dynamic_refs_mut(
            component,
            |id, _| {
                if let Some(slot) = used.get_mut(id.0 as usize) {
                    *slot = true;
                }
            },
            |_| {},
        );
    }
    visit_owned_style_sheet_binding_refs_mut(program, component_id, |id, _| {
        if let Some(slot) = used.get_mut(id.0 as usize) {
            *slot = true;
        }
    });
    for animation in &program.animations {
        if let UiAnimationTrigger::Binding { component, binding } = &animation.trigger {
            if *component == component_id {
                if let Some(slot) = used.get_mut(binding.0 as usize) {
                    *slot = true;
                }
            }
        }
    }

    if used.iter().all(|used| *used) {
        return false;
    }

    let mut map = vec![UiBindingId(0); binding_len];
    let mut new_bindings = Vec::with_capacity(used.iter().filter(|used| **used).count());
    for (old_index, mut binding) in std::mem::take(&mut program.components[component_index].bindings)
        .into_iter()
        .enumerate()
    {
        if used[old_index] {
            let new_id = UiBindingId(new_bindings.len() as u32);
            map[old_index] = new_id;
            binding.id = new_id;
            new_bindings.push(binding);
        }
    }

    // Dead binding IDs should have no remaining references. Mapping them to zero is
    // harmless and keeps the remapper total for malformed custom-pass input.
    program.components[component_index].bindings = new_bindings;
    remap_component_binding_refs(&mut program.components[component_index], &map);
    remap_owned_style_sheet_binding_refs(program, component_id, &map);
    for animation in &mut program.animations {
        if let UiAnimationTrigger::Binding { component, binding } = &mut animation.trigger {
            if *component == component_id {
                if let Some(mapped) = map.get(binding.0 as usize) {
                    *binding = *mapped;
                }
            }
        }
    }
    true
}

impl UiOptimization for BindingCSE {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;

        for component_index in 0..program.components.len() {
            let binding_len = program.components[component_index].bindings.len();
            if binding_len < 2 {
                continue;
            }

            let mut buckets: HashMap<u64, Vec<UiBindingId>> = HashMap::default();
            let mut aliases: Vec<UiBindingId> = (0..binding_len)
                .map(|index| UiBindingId(index as u32))
                .collect();

            {
                let component = &program.components[component_index];
                for binding in &component.bindings {
                    if !binding.flags.pure {
                        continue;
                    }
                    let fingerprint = binding_fingerprint(binding);
                    let bucket = buckets.entry(fingerprint).or_default();
                    let canonical = bucket.iter().copied().find(|candidate| {
                        binding_equivalent(
                            &component.bindings[candidate.0 as usize],
                            binding,
                        )
                    });
                    if let Some(canonical) = canonical {
                        aliases[binding.id.0 as usize] = canonical;
                    } else {
                        bucket.push(binding.id);
                    }
                }
            }

            if aliases.iter().enumerate().all(|(index, id)| id.0 as usize == index) {
                continue;
            }

            let component_id = program.components[component_index].id;
            remap_component_binding_refs(&mut program.components[component_index], &aliases);
            remap_owned_style_sheet_binding_refs(program, component_id, &aliases);
            for animation in &mut program.animations {
                if let UiAnimationTrigger::Binding { component, binding } = &mut animation.trigger {
                    if *component == component_id {
                        if let Some(mapped) = aliases.get(binding.0 as usize) {
                            *binding = *mapped;
                        }
                    }
                }
            }
            changed = true;
            changed |= compact_bindings(program, component_index);
        }

        changed
    }
}

// =============================================================================
// Structural control-flow simplification
// =============================================================================

fn runtime_value_eq(left: &UiRuntimeValue, right: &UiRuntimeValue) -> bool {
    match (left, right) {
        (UiRuntimeValue::Null, UiRuntimeValue::Null) => true,
        (UiRuntimeValue::Bool(a), UiRuntimeValue::Bool(b)) => a == b,
        (UiRuntimeValue::Number(a), UiRuntimeValue::Number(b)) => a.to_bits() == b.to_bits(),
        (UiRuntimeValue::String(a), UiRuntimeValue::String(b)) => a == b,
        (UiRuntimeValue::Opaque(a), UiRuntimeValue::Opaque(b)) => a == b,
        _ => false,
    }
}

impl UiOptimization for StaticConditionFold {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            let constants: Vec<Option<UiConstantValue>> = component
                .bindings
                .iter()
                .map(|binding| binding.constant.clone())
                .collect();

            for node in &mut component.nodes {
                let replacement: Option<Vec<UiNodeId>> = match &node.kind {
                    UiNodeKind::If(branch) => constants
                        .get(branch.condition.0 as usize)
                        .and_then(|constant| constant.as_ref())
                        .and_then(UiConstantValue::as_bool)
                        .map(|condition| {
                            if condition {
                                vec![branch.then_node]
                            } else {
                                branch.else_node.into_iter().collect()
                            }
                        }),
                    UiNodeKind::Switch(switch) => constants
                        .get(switch.scrutinee.0 as usize)
                        .and_then(|constant| constant.as_ref())
                        .and_then(UiConstantValue::as_runtime)
                        .map(|constant| {
                            switch
                                .cases
                                .iter()
                                .find(|case| runtime_value_eq(&case.pattern, &constant))
                                .map(|case| case.node)
                                .or(switch.fallback)
                                .into_iter()
                                .collect()
                        }),
                    _ => None,
                };
                if let Some(children) = replacement {
                    node.kind = UiNodeKind::Fragment;
                    node.children = children;
                    changed = true;
                }
            }
        }
        changed
    }
}

fn media_truth(condition: &UiMediaCondition) -> Option<bool> {
    match condition {
        UiMediaCondition::True => Some(true),
        UiMediaCondition::Not(inner) => media_truth(inner).map(|value| !value),
        UiMediaCondition::All(items) => {
            let mut unknown = false;
            for item in items {
                match media_truth(item) {
                    Some(false) => return Some(false),
                    Some(true) => {}
                    None => unknown = true,
                }
            }
            if unknown { None } else { Some(true) }
        }
        UiMediaCondition::Any(items) => {
            let mut unknown = false;
            for item in items {
                match media_truth(item) {
                    Some(true) => return Some(true),
                    Some(false) => {}
                    None => unknown = true,
                }
            }
            if unknown { None } else { Some(false) }
        }
        _ => None,
    }
}

impl UiOptimization for DeadBranchElimination {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            for node in &mut component.nodes {
                let mut replacement: Option<Vec<UiNodeId>> = None;
                match &mut node.kind {
                    UiNodeKind::If(branch) => {
                        if branch.else_node == Some(branch.then_node) {
                            replacement = Some(vec![branch.then_node]);
                        }
                    }
                    UiNodeKind::Switch(switch) => {
                        let old_len = switch.cases.len();
                        let mut unique = Vec::<UiSwitchCase>::with_capacity(old_len);
                        for case in switch.cases.drain(..) {
                            if unique.iter().any(|existing| runtime_value_eq(&existing.pattern, &case.pattern)) {
                                changed = true;
                                continue;
                            }
                            unique.push(case);
                        }
                        switch.cases = unique;
                        changed |= old_len != switch.cases.len();

                        let all_same = switch
                            .cases
                            .first()
                            .map(|first| {
                                switch.cases.iter().all(|case| case.node == first.node)
                                    && switch.fallback.is_none_or(|fallback| fallback == first.node)
                            })
                            .unwrap_or(false);
                        if all_same {
                            replacement = Some(vec![switch.cases[0].node]);
                        } else if switch.cases.is_empty() {
                            replacement = Some(switch.fallback.into_iter().collect());
                        }
                    }
                    UiNodeKind::Responsive(responsive) => {
                        let old_len = responsive.cases.len();
                        responsive.cases.retain(|case| media_truth(&case.condition) != Some(false));
                        changed |= old_len != responsive.cases.len();
                        if let Some(index) = responsive
                            .cases
                            .iter()
                            .position(|case| media_truth(&case.condition) == Some(true))
                        {
                            if index == 0 {
                                replacement = Some(vec![responsive.cases[0].root]);
                            } else if responsive.cases.len() > index + 1 {
                                // A statically true case dominates only the cases after it;
                                // earlier unknown media predicates must retain priority.
                                responsive.cases.truncate(index + 1);
                                changed = true;
                            }
                        } else if responsive.cases.is_empty() {
                            replacement = Some(vec![responsive.fallback]);
                        }
                    }
                    _ => {}
                }
                if let Some(children) = replacement {
                    node.kind = UiNodeKind::Fragment;
                    node.children = children;
                    changed = true;
                }
            }
        }
        changed
    }
}

fn is_static_true(value: &UiValue<bool>) -> bool {
    matches!(value, UiValue::Static(true))
}

fn is_static_one(value: &UiValue<f32>) -> bool {
    matches!(value, UiValue::Static(value) if value.to_bits() == 1.0f32.to_bits())
}

fn is_identity_transform(value: &UiValue<UiTransform>) -> bool {
    matches!(value, UiValue::Static(transform)
        if transform.translate.x == 0.0
            && transform.translate.y == 0.0
            && transform.scale.x == 1.0
            && transform.scale.y == 1.0
            && transform.rotation_radians == 0.0
            && transform.origin.x == 0.5
            && transform.origin.y == 0.5)
}

fn is_zero_padding(value: &UiValue<UiPadding>) -> bool {
    matches!(value, UiValue::Static(padding) if *padding == UiPadding::ZERO)
}

fn is_default_layout(layout: &UiLayout) -> bool {
    matches!(&layout.width, UiValue::Static(UiLength::Shrink))
        && matches!(&layout.height, UiValue::Static(UiLength::Shrink))
        && layout.min_width.is_none()
        && layout.min_height.is_none()
        && layout.max_width.is_none()
        && layout.max_height.is_none()
        && is_zero_padding(&layout.padding)
        && is_zero_padding(&layout.margin)
        && layout.align_self_x.is_none()
        && layout.align_self_y.is_none()
        && layout.aspect_ratio.is_none()
        && layout.z_index == 0
}

fn style_patch_is_empty(patch: &UiStylePatch) -> bool {
    patch.common.background.is_none()
        && patch.common.foreground.is_none()
        && patch.common.border.is_none()
        && patch.common.shadow.is_none()
        && patch.common.radius.is_none()
        && patch.common.opacity.is_none()
        && patch.common.cursor.is_none()
        && patch.common.snap.is_none()
        && patch.text.color.is_none()
        && patch.text.font.is_none()
        && patch.text.size.is_none()
        && patch.text.weight.is_none()
        && patch.text.line_height.is_none()
        && patch.text.letter_spacing.is_none()
        && patch.text.underline.is_none()
        && patch.text.strikethrough.is_none()
        && patch.widget.is_none()
}

fn transparent_fragment(node: &UiNode) -> bool {
    matches!(&node.kind, UiNodeKind::Fragment)
        && node.key.is_none()
        && node.name.is_none()
        && node.classes.is_empty()
        && is_default_layout(&node.layout)
        && style_patch_is_empty(&node.style.inline)
        && node.style.resolved_static.is_none()
        && node.style.static_transitions.is_empty()
        && node.style.dynamic_rules.is_empty()
        && is_static_true(&node.visible)
        && is_static_true(&node.enabled)
        && is_static_one(&node.opacity)
        && is_identity_transform(&node.transform)
        && matches!(&node.clip, UiClip::None)
        && node.cursor.is_none()
        && node.handlers.is_empty()
        && node.accessibility.is_none()
        && node.cache == UiCachePolicy::Automatic
}

impl UiOptimization for FlattenFragments {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            let mut rounds = 0;
            loop {
                rounds += 1;
                let snapshot = component.nodes.clone();
                let mut round_changed = false;
                for node in &mut component.nodes {
                    let mut flattened = Vec::with_capacity(node.children.len());
                    for child in node.children.drain(..) {
                        let Some(fragment) = snapshot.get(child.0 as usize) else {
                            flattened.push(child);
                            continue;
                        };
                        if transparent_fragment(fragment) {
                            flattened.extend(fragment.children.iter().copied());
                            round_changed = true;
                        } else {
                            flattened.push(child);
                        }
                    }
                    node.children = flattened;
                }
                changed |= round_changed;
                if !round_changed || rounds >= 8 {
                    break;
                }
            }
        }
        changed
    }
}

fn text_wrapper_is_mergeable(node: &UiNode) -> bool {
    node.children.is_empty()
        && node.key.is_none()
        && node.name.is_none()
        && node.classes.is_empty()
        && is_default_layout(&node.layout)
        && style_patch_is_empty(&node.style.inline)
        && node.style.resolved_static.is_none()
        && node.style.static_transitions.is_empty()
        && node.style.dynamic_rules.is_empty()
        && is_static_true(&node.visible)
        && is_static_true(&node.enabled)
        && is_static_one(&node.opacity)
        && is_identity_transform(&node.transform)
        && matches!(&node.clip, UiClip::None)
        && node.cursor.is_none()
        && node.handlers.is_empty()
        && node.accessibility.is_none()
        && node.cache == UiCachePolicy::Automatic
}

fn text_node_to_spans(node: &UiNode) -> Option<(UiTextFormat, Vec<UiTextSpan>)> {
    if !text_wrapper_is_mergeable(node) {
        return None;
    }
    match &node.kind {
        UiNodeKind::Text(text) => {
            let UiValue::Static(content) = &text.content else { return None; };
            Some((
                text.format.clone(),
                vec![UiTextSpan {
                    text: UiValue::Static(*content),
                    font: None,
                    size: None,
                    color: None,
                    background: None,
                    underline: false,
                    strikethrough: false,
                    link: None,
                }],
            ))
        }
        UiNodeKind::RichText(text) if text.on_link.is_none() => Some((text.format.clone(), text.spans.clone())),
        _ => None,
    }
}

impl UiOptimization for MergeAdjacentText {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            let mut parent_index = 0usize;
            while parent_index < component.nodes.len() {
                let children = component.nodes[parent_index].children.clone();
                if children.len() < 2 {
                    parent_index += 1;
                    continue;
                }
                let mut out = Vec::with_capacity(children.len());
                let mut index = 0;
                while index < children.len() {
                    let first_id = children[index];
                    let Some((format, mut spans)) = component
                        .nodes
                        .get(first_id.0 as usize)
                        .and_then(text_node_to_spans)
                    else {
                        out.push(first_id);
                        index += 1;
                        continue;
                    };

                    let mut end = index + 1;
                    while end < children.len() {
                        let next_id = children[end];
                        let Some((next_format, next_spans)) = component
                            .nodes
                            .get(next_id.0 as usize)
                            .and_then(text_node_to_spans)
                        else { break; };
                        if next_format != format {
                            break;
                        }
                        spans.extend(next_spans);
                        end += 1;
                    }

                    if end > index + 1 {
                        component.nodes[first_id.0 as usize].kind = UiNodeKind::RichText(UiRichTextNode {
                            spans,
                            format,
                            on_link: None,
                        });
                        for dead in &children[index + 1..end] {
                            let dead_node = &mut component.nodes[dead.0 as usize];
                            dead_node.kind = UiNodeKind::Fragment;
                            dead_node.children.clear();
                        }
                        changed = true;
                    }
                    out.push(first_id);
                    index = end;
                }
                component.nodes[parent_index].children = out;
                parent_index += 1;
            }
        }
        changed
    }
}

// =============================================================================
// Style cascade analysis, dead rules and static style interning
// =============================================================================

fn widget_tag(kind: &UiNodeKind) -> UiWidgetTag {
    match kind {
        UiNodeKind::Fragment => UiWidgetTag::Any,
        UiNodeKind::Container(_) => UiWidgetTag::Container,
        UiNodeKind::Row(_) => UiWidgetTag::Row,
        UiNodeKind::Column(_) => UiWidgetTag::Column,
        UiNodeKind::Grid(_) => UiWidgetTag::Grid,
        UiNodeKind::Stack(_) => UiWidgetTag::Stack,
        UiNodeKind::Space(_) => UiWidgetTag::Space,
        UiNodeKind::Rule(_) => UiWidgetTag::Rule,
        UiNodeKind::Text(_) => UiWidgetTag::Text,
        UiNodeKind::RichText(_) => UiWidgetTag::RichText,
        UiNodeKind::Markdown(_) => UiWidgetTag::Markdown,
        UiNodeKind::Button(_) => UiWidgetTag::Button,
        UiNodeKind::Checkbox(_) => UiWidgetTag::Checkbox,
        UiNodeKind::Radio(_) => UiWidgetTag::Radio,
        UiNodeKind::Toggler(_) => UiWidgetTag::Toggler,
        UiNodeKind::TextInput(_) => UiWidgetTag::TextInput,
        UiNodeKind::TextEditor(_) => UiWidgetTag::TextEditor,
        UiNodeKind::Slider(_) => UiWidgetTag::Slider,
        UiNodeKind::VerticalSlider(_) => UiWidgetTag::VerticalSlider,
        UiNodeKind::PickList(_) => UiWidgetTag::PickList,
        UiNodeKind::ComboBox(_) => UiWidgetTag::ComboBox,
        UiNodeKind::ProgressBar(_) => UiWidgetTag::ProgressBar,
        UiNodeKind::Scrollable(_) => UiWidgetTag::Scrollable,
        UiNodeKind::PaneGrid(_) => UiWidgetTag::PaneGrid,
        UiNodeKind::Table(_) => UiWidgetTag::Table,
        UiNodeKind::Tooltip(_) => UiWidgetTag::Tooltip,
        UiNodeKind::MouseArea(_) => UiWidgetTag::MouseArea,
        UiNodeKind::Hover(_) => UiWidgetTag::Hover,
        UiNodeKind::Float(_) => UiWidgetTag::Float,
        UiNodeKind::Pin(_) => UiWidgetTag::Pin,
        UiNodeKind::Responsive(_) => UiWidgetTag::Responsive,
        UiNodeKind::Sensor(_) => UiWidgetTag::Sensor,
        UiNodeKind::Themer(_) => UiWidgetTag::Themer,
        UiNodeKind::Opaque => UiWidgetTag::Opaque,
        UiNodeKind::Lazy(_) => UiWidgetTag::Lazy,
        UiNodeKind::Image(_) => UiWidgetTag::Image,
        UiNodeKind::Svg(_) => UiWidgetTag::Svg,
        UiNodeKind::Canvas(_) => UiWidgetTag::Canvas,
        UiNodeKind::Shader(_) => UiWidgetTag::Shader,
        UiNodeKind::QrCode(_) => UiWidgetTag::QrCode,
        UiNodeKind::Component(_) => UiWidgetTag::Component,
        UiNodeKind::Slot(_) | UiNodeKind::If(_) | UiNodeKind::Switch(_) | UiNodeKind::ForEach(_) | UiNodeKind::Portal(_) => UiWidgetTag::Any,
        UiNodeKind::Custom(_) => UiWidgetTag::Custom,
    }
}

fn is_dynamic_pseudo(state: UiPseudoClass) -> bool {
    !matches!(
        state,
        UiPseudoClass::FirstChild | UiPseudoClass::LastChild | UiPseudoClass::Odd | UiPseudoClass::Even
    )
}

fn selector_has_dynamic_state(selector: &UiSelector) -> bool {
    selector.chain.iter().any(|part| part.atom.states.iter().copied().any(is_dynamic_pseudo))
}

fn selector_targets_widget_part(selector: &UiSelector) -> bool {
    selector.chain.iter().any(|part| part.atom.part.is_some())
}

fn structural_child_ids(node: &UiNode, out: &mut Vec<UiNodeId>) {
    out.extend(node.children.iter().copied());
    match &node.kind {
        UiNodeKind::PaneGrid(widget) => {
            for pane in &widget.panes {
                out.push(pane.content);
                if let Some(title) = pane.title_bar {
                    out.push(title);
                }
            }
        }
        UiNodeKind::Table(widget) => {
            out.push(widget.row_template);
            if let Some(header) = widget.header {
                out.push(header);
            }
            if let Some(footer) = widget.footer {
                out.push(footer);
            }
        }
        UiNodeKind::Tooltip(widget) => out.push(widget.tooltip),
        UiNodeKind::Hover(widget) => out.push(widget.overlay),
        UiNodeKind::Responsive(widget) => {
            out.extend(widget.cases.iter().map(|case| case.root));
            out.push(widget.fallback);
        }
        UiNodeKind::Lazy(widget) => out.push(widget.subtree),
        UiNodeKind::Component(widget) => out.extend(widget.slots.iter().map(|(_, node)| *node)),
        UiNodeKind::If(widget) => {
            out.push(widget.then_node);
            if let Some(else_node) = widget.else_node {
                out.push(else_node);
            }
        }
        UiNodeKind::Switch(widget) => {
            out.extend(widget.cases.iter().map(|case| case.node));
            if let Some(fallback) = widget.fallback {
                out.push(fallback);
            }
        }
        UiNodeKind::ForEach(widget) => {
            out.push(widget.template);
            if let Some(empty) = widget.empty {
                out.push(empty);
            }
        }
        UiNodeKind::Portal(widget) => out.push(widget.content),
        _ => {}
    }
}

fn build_parent_map(component: &UiComponent) -> Vec<Option<UiNodeId>> {
    let mut parents = vec![None; component.nodes.len()];
    let mut children = Vec::new();
    for node in &component.nodes {
        children.clear();
        structural_child_ids(node, &mut children);
        for &child in &children {
            if let Some(parent) = parents.get_mut(child.0 as usize) {
                if parent.is_none() {
                    *parent = Some(node.id);
                }
            }
        }
    }
    parents
}

fn sibling_ids(component: &UiComponent, parent: UiNodeId) -> Option<Vec<UiNodeId>> {
    let parent = component.node(parent)?;
    let mut children = Vec::new();
    structural_child_ids(parent, &mut children);
    Some(children)
}

fn sibling_position(
    component: &UiComponent,
    parents: &[Option<UiNodeId>],
    node: UiNodeId,
) -> Option<(usize, usize)> {
    let parent = parents.get(node.0 as usize).copied().flatten()?;
    let children = sibling_ids(component, parent)?;
    let position = children.iter().position(|candidate| *candidate == node)?;
    Some((position, children.len()))
}

fn atom_structurally_matches(
    component: &UiComponent,
    parents: &[Option<UiNodeId>],
    node_id: UiNodeId,
    atom: &UiSelectorAtom,
) -> bool {
    let Some(node) = component.node(node_id) else { return false; };
    if let Some(tag) = atom.widget {
        if tag != UiWidgetTag::Any && tag != widget_tag(&node.kind) {
            return false;
        }
    }
    if atom.name.is_some() && atom.name != node.name {
        return false;
    }
    if atom.classes.iter().any(|class| !node.classes.contains(class)) {
        return false;
    }
    for state in &atom.states {
        match state {
            UiPseudoClass::FirstChild => {
                let Some((position, _)) = sibling_position(component, parents, node_id) else {
                    return false;
                };
                if position != 0 {
                    return false;
                }
            }
            UiPseudoClass::LastChild => {
                let Some((position, len)) = sibling_position(component, parents, node_id) else {
                    return false;
                };
                if position + 1 != len {
                    return false;
                }
            }
            UiPseudoClass::Odd => {
                let Some((position, _)) = sibling_position(component, parents, node_id) else {
                    return false;
                };
                if (position + 1) % 2 == 0 {
                    return false;
                }
            }
            UiPseudoClass::Even => {
                let Some((position, _)) = sibling_position(component, parents, node_id) else {
                    return false;
                };
                if (position + 1) % 2 != 0 {
                    return false;
                }
            }
            dynamic if is_dynamic_pseudo(*dynamic) => {}
            _ => {}
        }
    }
    true
}

fn previous_sibling(
    component: &UiComponent,
    parents: &[Option<UiNodeId>],
    node: UiNodeId,
) -> Option<UiNodeId> {
    let parent = parents.get(node.0 as usize).copied().flatten()?;
    let children = sibling_ids(component, parent)?;
    let position = children.iter().position(|candidate| *candidate == node)?;
    position.checked_sub(1).map(|index| children[index])
}

fn selector_matches_from(
    component: &UiComponent,
    parents: &[Option<UiNodeId>],
    selector: &UiSelector,
    part_index: usize,
    node: UiNodeId,
) -> bool {
    if !atom_structurally_matches(component, parents, node, &selector.chain[part_index].atom) {
        return false;
    }
    if part_index == 0 {
        return true;
    }

    match selector.chain[part_index]
        .combinator
        .unwrap_or(UiSelectorCombinator::Descendant)
    {
        UiSelectorCombinator::Child => parents
            .get(node.0 as usize)
            .copied()
            .flatten()
            .is_some_and(|parent| selector_matches_from(component, parents, selector, part_index - 1, parent)),
        UiSelectorCombinator::Descendant => {
            let mut current = parents.get(node.0 as usize).copied().flatten();
            while let Some(ancestor) = current {
                if selector_matches_from(component, parents, selector, part_index - 1, ancestor) {
                    return true;
                }
                current = parents.get(ancestor.0 as usize).copied().flatten();
            }
            false
        }
        UiSelectorCombinator::AdjacentSibling => previous_sibling(component, parents, node)
            .is_some_and(|sibling| selector_matches_from(component, parents, selector, part_index - 1, sibling)),
        UiSelectorCombinator::GeneralSibling => {
            let Some(parent) = parents.get(node.0 as usize).copied().flatten() else { return false; };
            let Some(children) = sibling_ids(component, parent) else { return false; };
            let Some(position) = children.iter().position(|candidate| *candidate == node) else { return false; };
            children[..position]
                .iter()
                .copied()
                .rev()
                .any(|sibling| selector_matches_from(component, parents, selector, part_index - 1, sibling))
        }
    }
}

fn selector_structurally_matches(
    component: &UiComponent,
    parents: &[Option<UiNodeId>],
    node: UiNodeId,
    selector: &UiSelector,
) -> bool {
    if selector.chain.is_empty() {
        return false;
    }
    selector_matches_from(component, parents, selector, selector.chain.len() - 1, node)
}

fn cascade_origin_rank(origin: UiStyleOrigin) -> u8 {
    match origin {
        UiStyleOrigin::UserAgent => 0,
        UiStyleOrigin::Theme => 1,
        UiStyleOrigin::Component => 2,
        UiStyleOrigin::Author => 3,
        UiStyleOrigin::Inline => 4,
    }
}

fn cascade_compare(left: &UiStyleRule, right: &UiStyleRule) -> Ordering {
    left.important
        .cmp(&right.important)
        .then_with(|| cascade_origin_rank(left.origin).cmp(&cascade_origin_rank(right.origin)))
        .then_with(|| left.layer.cmp(&right.layer))
        .then_with(|| left.specificity.cmp(&right.specificity))
        .then_with(|| left.order.cmp(&right.order))
}

fn patch_has_binding(patch: &UiStylePatch) -> bool {
    let mut patch = patch.clone();
    let mut found = false;
    visit_style_patch_dynamic_refs_mut(&mut patch, &mut |_, _| found = true, &mut |_| {});
    found
}

fn assign_option<T: Clone>(target: &mut Option<T>, source: &Option<T>) {
    if let Some(value) = source {
        *target = Some(value.clone());
    }
}

fn merge_static_transitions(target: &mut Vec<UiTransition>, source: &[UiTransition]) {
    for transition in source {
        if let Some(existing) = target
            .iter_mut()
            .find(|candidate| candidate.property == transition.property)
        {
            *existing = transition.clone();
        } else {
            target.push(transition.clone());
        }
    }
}

fn merge_container_style(target: &mut UiContainerStylePatch, source: &UiContainerStylePatch) {
    assign_option(&mut target.text_color, &source.text_color);
    assign_option(&mut target.background, &source.background);
    assign_option(&mut target.border, &source.border);
    assign_option(&mut target.shadow, &source.shadow);
    assign_option(&mut target.snap, &source.snap);
}

fn merge_style_patch(target: &mut UiStylePatch, source: &UiStylePatch) {
    let common = &mut target.common;
    assign_option(&mut common.background, &source.common.background);
    assign_option(&mut common.foreground, &source.common.foreground);
    assign_option(&mut common.border, &source.common.border);
    assign_option(&mut common.shadow, &source.common.shadow);
    assign_option(&mut common.radius, &source.common.radius);
    assign_option(&mut common.opacity, &source.common.opacity);
    assign_option(&mut common.cursor, &source.common.cursor);
    assign_option(&mut common.snap, &source.common.snap);

    let text = &mut target.text;
    assign_option(&mut text.color, &source.text.color);
    assign_option(&mut text.font, &source.text.font);
    assign_option(&mut text.size, &source.text.size);
    assign_option(&mut text.weight, &source.text.weight);
    assign_option(&mut text.line_height, &source.text.line_height);
    assign_option(&mut text.letter_spacing, &source.text.letter_spacing);
    assign_option(&mut text.underline, &source.text.underline);
    assign_option(&mut text.strikethrough, &source.text.strikethrough);

    let Some(source_widget) = &source.widget else { return; };
    match (&mut target.widget, source_widget) {
        (Some(UiWidgetStylePatch::Application(target)), UiWidgetStylePatch::Application(source)) => {
            assign_option(&mut target.background_color, &source.background_color);
            assign_option(&mut target.text_color, &source.text_color);
        }
        (Some(UiWidgetStylePatch::Container(target)), UiWidgetStylePatch::Container(source)) => {
            merge_container_style(target, source);
        }
        (Some(UiWidgetStylePatch::Float(target)), UiWidgetStylePatch::Float(source)) => {
            assign_option(&mut target.shadow, &source.shadow);
            assign_option(&mut target.shadow_border_radius, &source.shadow_border_radius);
        }
        (Some(UiWidgetStylePatch::Button(target)), UiWidgetStylePatch::Button(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.text_color, &source.text_color);
            assign_option(&mut target.border, &source.border);
            assign_option(&mut target.shadow, &source.shadow);
            assign_option(&mut target.snap, &source.snap);
        }
        (Some(UiWidgetStylePatch::Checkbox(target)), UiWidgetStylePatch::Checkbox(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.icon_color, &source.icon_color);
            assign_option(&mut target.border, &source.border);
            assign_option(&mut target.text_color, &source.text_color);
        }
        (Some(UiWidgetStylePatch::Radio(target)), UiWidgetStylePatch::Radio(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.dot_color, &source.dot_color);
            assign_option(&mut target.border_width, &source.border_width);
            assign_option(&mut target.border_color, &source.border_color);
            assign_option(&mut target.text_color, &source.text_color);
        }
        (Some(UiWidgetStylePatch::Toggler(target)), UiWidgetStylePatch::Toggler(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.background_border_width, &source.background_border_width);
            assign_option(&mut target.background_border_color, &source.background_border_color);
            assign_option(&mut target.foreground, &source.foreground);
            assign_option(&mut target.foreground_border_width, &source.foreground_border_width);
            assign_option(&mut target.foreground_border_color, &source.foreground_border_color);
            assign_option(&mut target.text_color, &source.text_color);
            assign_option(&mut target.border_radius, &source.border_radius);
            assign_option(&mut target.padding_ratio, &source.padding_ratio);
        }
        (Some(UiWidgetStylePatch::Text(target)), UiWidgetStylePatch::Text(source)) => {
            assign_option(&mut target.color, &source.color);
        }
        (Some(UiWidgetStylePatch::TextInput(target)), UiWidgetStylePatch::TextInput(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.border, &source.border);
            assign_option(&mut target.icon, &source.icon);
            assign_option(&mut target.placeholder, &source.placeholder);
            assign_option(&mut target.value, &source.value);
            assign_option(&mut target.selection, &source.selection);
        }
        (Some(UiWidgetStylePatch::TextEditor(target)), UiWidgetStylePatch::TextEditor(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.border, &source.border);
            assign_option(&mut target.placeholder, &source.placeholder);
            assign_option(&mut target.value, &source.value);
            assign_option(&mut target.selection, &source.selection);
        }
        (Some(UiWidgetStylePatch::PickList(target)), UiWidgetStylePatch::PickList(source)) => {
            assign_option(&mut target.text_color, &source.text_color);
            assign_option(&mut target.placeholder_color, &source.placeholder_color);
            assign_option(&mut target.handle_color, &source.handle_color);
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.border, &source.border);
        }
        (Some(UiWidgetStylePatch::Menu(target)), UiWidgetStylePatch::Menu(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.border, &source.border);
            assign_option(&mut target.text_color, &source.text_color);
            assign_option(&mut target.selected_text_color, &source.selected_text_color);
            assign_option(&mut target.selected_background, &source.selected_background);
            assign_option(&mut target.shadow, &source.shadow);
        }
        (Some(UiWidgetStylePatch::Slider(target)), UiWidgetStylePatch::Slider(source)) => {
            assign_option(&mut target.rail, &source.rail);
            assign_option(&mut target.handle, &source.handle);
        }
        (Some(UiWidgetStylePatch::Scrollable(target)), UiWidgetStylePatch::Scrollable(source)) => {
            match (&mut target.container, &source.container) {
                (Some(target), Some(source)) => merge_container_style(target, source),
                (None, Some(source)) => target.container = Some(source.clone()),
                _ => {}
            }
            assign_option(&mut target.vertical_rail, &source.vertical_rail);
            assign_option(&mut target.horizontal_rail, &source.horizontal_rail);
            assign_option(&mut target.gap, &source.gap);
            assign_option(&mut target.auto_scroll, &source.auto_scroll);
        }
        (Some(UiWidgetStylePatch::ProgressBar(target)), UiWidgetStylePatch::ProgressBar(source)) => {
            assign_option(&mut target.background, &source.background);
            assign_option(&mut target.bar, &source.bar);
            assign_option(&mut target.border, &source.border);
        }
        (Some(UiWidgetStylePatch::Rule(target)), UiWidgetStylePatch::Rule(source)) => {
            assign_option(&mut target.color, &source.color);
            assign_option(&mut target.radius, &source.radius);
            assign_option(&mut target.fill_mode, &source.fill_mode);
            assign_option(&mut target.snap, &source.snap);
        }
        (Some(UiWidgetStylePatch::Svg(target)), UiWidgetStylePatch::Svg(source)) => {
            assign_option(&mut target.color, &source.color);
        }
        (Some(UiWidgetStylePatch::PaneGrid(target)), UiWidgetStylePatch::PaneGrid(source)) => {
            assign_option(&mut target.hovered_region, &source.hovered_region);
            assign_option(&mut target.picked_split, &source.picked_split);
            assign_option(&mut target.hovered_split, &source.hovered_split);
        }
        (Some(UiWidgetStylePatch::Table(target)), UiWidgetStylePatch::Table(source)) => {
            assign_option(&mut target.separator_x, &source.separator_x);
            assign_option(&mut target.separator_y, &source.separator_y);
        }
        (Some(UiWidgetStylePatch::QrCode(target)), UiWidgetStylePatch::QrCode(source)) => {
            assign_option(&mut target.cell, &source.cell);
            assign_option(&mut target.background, &source.background);
        }
        (Some(UiWidgetStylePatch::Markdown(target)), UiWidgetStylePatch::Markdown(source)) => {
            assign_option(&mut target.font, &source.font);
            assign_option(&mut target.inline_code_background, &source.inline_code_background);
            assign_option(&mut target.inline_code_padding, &source.inline_code_padding);
            assign_option(&mut target.inline_code_color, &source.inline_code_color);
            assign_option(&mut target.inline_code_font, &source.inline_code_font);
            assign_option(&mut target.code_block_font, &source.code_block_font);
            assign_option(&mut target.link_color, &source.link_color);
        }
        (target, source) => *target = Some(source.clone()),
    }
}

fn component_style_sheets(program: &UiProgram, component: &UiComponent) -> Vec<UiStyleSheetId> {
    let mut out = component.style_sheets.clone();
    if let Some(UiValue::Static(theme)) = &component.theme {
        if let Some(theme) = program.theme(*theme) {
            out.extend(theme.style_sheets.iter().copied());
        }
    }
    let mut seen = HashSet::default();
    out.retain(|id| seen.insert(*id));
    out
}

impl UiOptimization for DeadStyleRuleElimination {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let components = program.components.clone();
        let parent_maps: Vec<Vec<Option<UiNodeId>>> = components.iter().map(build_parent_map).collect();
        let had_cached_style_metadata = !program.static_styles.is_empty()
            || program.components.iter().any(|component| {
                component.nodes.iter().any(|node| {
                    node.style.resolved_static.is_some()
                        || !node.style.static_transitions.is_empty()
                        || !node.style.dynamic_rules.is_empty()
                })
            });
        let mut changed = had_cached_style_metadata;

        for component in &mut program.components {
            for node in &mut component.nodes {
                node.style.dynamic_rules.clear();
                node.style.static_transitions.clear();
                node.style.resolved_static = None;
            }
        }
        program.static_styles.clear();

        for sheet in &mut program.style_sheets {
            let old_len = sheet.rules.len();
            sheet.rules.retain(|rule| {
                components.iter().enumerate().any(|(component_index, component)| {
                    component.nodes.iter().any(|node| {
                        selector_structurally_matches(
                            component,
                            &parent_maps[component_index],
                            node.id,
                            &rule.selector,
                        )
                    })
                })
            });
            for (index, rule) in sheet.rules.iter_mut().enumerate() {
                rule.id = UiStyleRuleId(index as u32);
                rule.order = index as u32;
            }
            changed |= old_len != sheet.rules.len();
        }
        changed
    }
}

impl UiOptimization for StyleCascadePrecompute {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        program.static_styles.clear();
        for component in &mut program.components {
            for node in &mut component.nodes {
                node.style.resolved_static = None;
                node.style.static_transitions.clear();
                node.style.dynamic_rules.clear();
            }
        }

        let components_snapshot = program.components.clone();
        let style_sheets_snapshot = program.style_sheets.clone();
        let themes_snapshot = program.themes.clone();
        let mut changed = false;

        for component_index in 0..program.components.len() {
            let snapshot = &components_snapshot[component_index];
            let parents = build_parent_map(snapshot);
            let mut sheet_ids = snapshot.style_sheets.clone();
            if let Some(UiValue::Static(theme_id)) = &snapshot.theme {
                if let Some(theme) = themes_snapshot.get(theme_id.0 as usize) {
                    sheet_ids.extend(theme.style_sheets.iter().copied());
                }
            }
            let mut seen = HashSet::default();
            sheet_ids.retain(|id| seen.insert(*id));

            for node_index in 0..snapshot.nodes.len() {
                let node_id = UiNodeId(node_index as u32);
                let mut matching: Vec<(UiStyleRuleRef, UiStyleRule)> = Vec::new();
                for sheet_id in &sheet_ids {
                    let Some(sheet) = style_sheets_snapshot.get(sheet_id.0 as usize) else { continue; };
                    for rule in &sheet.rules {
                        if selector_structurally_matches(snapshot, &parents, node_id, &rule.selector) {
                            matching.push((UiStyleRuleRef { sheet: *sheet_id, rule: rule.id }, rule.clone()));
                        }
                    }
                }
                matching.sort_by(|(_, left), (_, right)| cascade_compare(left, right));

                let mut static_patch = UiStylePatch::default();
                let mut static_transitions = Vec::new();
                let mut dynamic = Vec::new();
                for (reference, rule) in matching {
                    let static_rule = rule.media.is_none()
                        && !selector_has_dynamic_state(&rule.selector)
                        && !selector_targets_widget_part(&rule.selector)
                        && !patch_has_binding(&rule.declarations);
                    if static_rule {
                        merge_style_patch(&mut static_patch, &rule.declarations);
                        merge_static_transitions(&mut static_transitions, &rule.transitions);
                    } else {
                        dynamic.push(reference);
                    }
                }

                let resolved_static = if !style_patch_is_empty(&static_patch) {
                    let id = UiStaticStyleId(program.static_styles.len() as u32);
                    program.static_styles.push(UiStaticStyle { id, patch: static_patch });
                    Some(id)
                } else {
                    None
                };
                let node = &mut program.components[component_index].nodes[node_index];
                node.style.dynamic_rules = dynamic;
                node.style.static_transitions = static_transitions;
                node.style.resolved_static = resolved_static;
                changed = true;
            }
        }
        changed
    }
}

fn static_style_fingerprint(style: &UiStaticStyle) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write(format!("{:?}", style.patch).as_bytes());
    hasher.finish()
}

impl UiOptimization for StaticStyleInterning {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        if program.static_styles.is_empty() {
            return false;
        }

        let old_len = program.static_styles.len();
        let mut referenced = vec![false; old_len];
        for component in &program.components {
            for node in &component.nodes {
                if let Some(style) = node.style.resolved_static {
                    if let Some(slot) = referenced.get_mut(style.0 as usize) {
                        *slot = true;
                    }
                }
            }
        }

        let mut buckets: HashMap<u64, Vec<UiStaticStyleId>> = HashMap::default();
        let mut aliases = vec![UiStaticStyleId(0); old_len];
        let mut unique = Vec::new();

        for (old_index, style) in program.static_styles.iter().enumerate() {
            if !referenced[old_index] {
                continue;
            }
            let fingerprint = static_style_fingerprint(style);
            let bucket = buckets.entry(fingerprint).or_default();
            let existing = bucket.iter().copied().find(|id| {
                unique
                    .get(id.0 as usize)
                    .is_some_and(|candidate: &UiStaticStyle| candidate.patch == style.patch)
            });
            if let Some(existing) = existing {
                aliases[old_index] = existing;
            } else {
                let id = UiStaticStyleId(unique.len() as u32);
                aliases[old_index] = id;
                let mut style = style.clone();
                style.id = id;
                unique.push(style);
                bucket.push(id);
            }
        }

        let changed = unique.len() != old_len
            || program.components.iter().any(|component| {
                component.nodes.iter().any(|node| {
                    node.style.resolved_static.is_some_and(|id| {
                        aliases.get(id.0 as usize).is_some_and(|mapped| *mapped != id)
                    })
                })
            });
        if !changed {
            return false;
        }

        program.static_styles = unique;
        for component in &mut program.components {
            for node in &mut component.nodes {
                if let Some(style) = &mut node.style.resolved_static {
                    if let Some(mapped) = aliases.get(style.0 as usize) {
                        *style = *mapped;
                    }
                }
            }
        }
        true
    }
}

// =============================================================================
// Loop reconciliation and component specialization/inlining
// =============================================================================

impl UiOptimization for KeyedLoopCanonicalization {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            let snapshot = component.nodes.clone();
            for node in &mut component.nodes {
                let UiNodeKind::ForEach(loop_) = &mut node.kind else { continue; };
                let strategy = if let Some(function) = loop_.key_function {
                    UiLoopKeyStrategy::Explicit(function)
                } else if snapshot
                    .get(loop_.template.0 as usize)
                    .is_some_and(|template| template.key.is_some())
                {
                    UiLoopKeyStrategy::TemplateKey
                } else {
                    UiLoopKeyStrategy::Index
                };
                if loop_.key_strategy != strategy {
                    loop_.key_strategy = strategy;
                    changed = true;
                }
            }
        }
        changed
    }
}

fn component_specialization_key(
    component: &UiComponent,
    instance: &UiComponentInstanceNode,
) -> Option<u64> {
    let mut hasher = StableHasher::new();
    instance.component.hash(&mut hasher);

    let mut props = instance.props.clone();
    props.sort_unstable_by_key(|(property, _)| property.0);
    for (property, binding) in props {
        property.hash(&mut hasher);
        let binding = component.binding(binding)?;
        let constant = binding.constant.as_ref()?;
        hash_constant(&mut hasher, constant);
    }

    let mut slots = instance.slots.clone();
    slots.sort_unstable_by_key(|(slot, _)| slot.0);
    for (slot, node) in slots {
        slot.hash(&mut hasher);
        node.hash(&mut hasher);
    }
    Some(hasher.finish())
}

impl UiOptimization for ComponentSpecialization {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            let snapshot = component.clone();
            for node in &mut component.nodes {
                let UiNodeKind::Component(instance) = &mut node.kind else { continue; };
                let key = component_specialization_key(&snapshot, instance);
                if instance.specialization_key != key {
                    instance.specialization_key = key;
                    changed = true;
                }
            }
        }
        changed
    }
}

fn lifecycle_is_empty(lifecycle: &UiLifecycle) -> bool {
    lifecycle.on_mount.is_none()
        && lifecycle.on_unmount.is_none()
        && lifecycle.on_window_opened.is_none()
        && lifecycle.on_window_closed.is_none()
        && lifecycle.on_theme_changed.is_none()
}

fn component_is_inline_eligible(program: &UiProgram, target: &UiComponent, max_nodes: usize) -> bool {
    if target.kind != UiComponentKind::Component
        || target.embedded_root.is_none()
        || target.nodes.len() > max_nodes
        || !target.properties.is_empty()
        || !target.states.is_empty()
        || !target.computed.is_empty()
        || !target.bindings.is_empty()
        || !target.handlers.is_empty()
        || !target.subscriptions.is_empty()
        || !target.windows.is_empty()
        || !target.style_sheets.is_empty()
        || target.theme.is_some()
        || !target.slots.is_empty()
        || !lifecycle_is_empty(&target.lifecycle)
    {
        return false;
    }
    if target.nodes.iter().any(|node| matches!(&node.kind, UiNodeKind::Component(_))) {
        return false;
    }
    // Global animation identity would otherwise have to be cloned/rebound.
    !program.animations.iter().any(|animation| {
        animation.tracks.iter().any(|track| {
            matches!(
                &track.target,
                UiAnimationTarget::Node { component, .. } if *component == target.id
            )
        })
    })
}

fn remap_node_id(id: &mut UiNodeId, map: &[UiNodeId]) {
    if let Some(mapped) = map.get(id.0 as usize) {
        *id = *mapped;
    }
}

fn remap_node_refs_in_node(node: &mut UiNode, map: &[UiNodeId]) {
    for child in &mut node.children {
        remap_node_id(child, map);
    }
    match &mut node.kind {
        UiNodeKind::PaneGrid(widget) => {
            for pane in &mut widget.panes {
                remap_node_id(&mut pane.content, map);
                if let Some(title) = &mut pane.title_bar {
                    remap_node_id(title, map);
                }
            }
        }
        UiNodeKind::Table(widget) => {
            remap_node_id(&mut widget.row_template, map);
            if let Some(header) = &mut widget.header {
                remap_node_id(header, map);
            }
            if let Some(footer) = &mut widget.footer {
                remap_node_id(footer, map);
            }
        }
        UiNodeKind::Tooltip(widget) => remap_node_id(&mut widget.tooltip, map),
        UiNodeKind::Hover(widget) => remap_node_id(&mut widget.overlay, map),
        UiNodeKind::Responsive(widget) => {
            for case in &mut widget.cases {
                remap_node_id(&mut case.root, map);
            }
            remap_node_id(&mut widget.fallback, map);
        }
        UiNodeKind::Lazy(widget) => remap_node_id(&mut widget.subtree, map),
        UiNodeKind::Component(widget) => {
            for (_, node) in &mut widget.slots {
                remap_node_id(node, map);
            }
        }
        UiNodeKind::If(widget) => {
            remap_node_id(&mut widget.then_node, map);
            if let Some(else_node) = &mut widget.else_node {
                remap_node_id(else_node, map);
            }
        }
        UiNodeKind::Switch(widget) => {
            for case in &mut widget.cases {
                remap_node_id(&mut case.node, map);
            }
            if let Some(fallback) = &mut widget.fallback {
                remap_node_id(fallback, map);
            }
        }
        UiNodeKind::ForEach(widget) => {
            remap_node_id(&mut widget.template, map);
            if let Some(empty) = &mut widget.empty {
                remap_node_id(empty, map);
            }
        }
        UiNodeKind::Portal(widget) => remap_node_id(&mut widget.content, map),
        _ => {}
    }
}

fn inline_component_into(caller: &mut UiComponent, instance_node: UiNodeId, target: &UiComponent) -> bool {
    let Some(root) = target.embedded_root else { return false; };
    let base = caller.nodes.len();
    let map: Vec<UiNodeId> = (0..target.nodes.len())
        .map(|index| UiNodeId((base + index) as u32))
        .collect();

    let mut clones = target.nodes.clone();
    for (index, node) in clones.iter_mut().enumerate() {
        node.id = map[index];
        remap_node_refs_in_node(node, &map);
    }
    let new_root = map[root.0 as usize];
    caller.nodes.extend(clones);

    let wrapper = &mut caller.nodes[instance_node.0 as usize];
    wrapper.kind = UiNodeKind::Fragment;
    wrapper.children = vec![new_root];
    true
}

impl UiOptimization for ComponentInlining {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let snapshot = program.components.clone();
        let eligible: Vec<bool> = snapshot
            .iter()
            .map(|component| component_is_inline_eligible(program, component, self.max_nodes))
            .collect();
        let mut changed = false;

        for caller_index in 0..program.components.len() {
            let caller_id = program.components[caller_index].id;
            let callsites: Vec<(UiNodeId, UiComponentId)> = program.components[caller_index]
                .nodes
                .iter()
                .filter_map(|node| match &node.kind {
                    UiNodeKind::Component(instance) => Some((node.id, instance.component)),
                    _ => None,
                })
                .collect();

            for (node, target_id) in callsites {
                if target_id == caller_id
                    || !eligible.get(target_id.0 as usize).copied().unwrap_or(false)
                {
                    continue;
                }
                let Some(target) = snapshot.get(target_id.0 as usize) else { continue; };
                changed |= inline_component_into(&mut program.components[caller_index], node, target);
            }
        }
        changed
    }
}

// =============================================================================
// Layout propagation and binding invalidation analysis
// =============================================================================

fn normalize_length(length: &mut UiLength) -> bool {
    let old = *length;
    match length {
        UiLength::FillPortion(portion) if *portion == 0 => *portion = 1,
        UiLength::Pixels(value) => {
            if !value.is_finite() || *value < 0.0 {
                *value = 0.0;
            }
        }
        UiLength::Percent(value) => {
            if !value.is_finite() || *value < 0.0 {
                *value = 0.0;
            }
        }
        _ => {}
    }
    old != *length
}

fn normalize_length_value(value: &mut UiValue<UiLength>) -> bool {
    match value {
        UiValue::Static(length) => normalize_length(length),
        _ => false,
    }
}

fn normalize_nonnegative_value(value: &mut UiValue<f32>) -> bool {
    let UiValue::Static(value) = value else { return false; };
    let old = *value;
    if !value.is_finite() || *value < 0.0 {
        *value = 0.0;
    }
    old.to_bits() != value.to_bits()
}

fn normalize_optional_nonnegative(value: &mut Option<UiValue<f32>>) -> bool {
    value.as_mut().is_some_and(normalize_nonnegative_value)
}

fn normalize_padding(value: &mut UiValue<UiPadding>) -> bool {
    let UiValue::Static(padding) = value else { return false; };
    let old = *padding;
    for side in [&mut padding.top, &mut padding.right, &mut padding.bottom, &mut padding.left] {
        if !side.is_finite() || *side < 0.0 {
            *side = 0.0;
        }
    }
    old != *padding
}

fn static_f32(value: &Option<UiValue<f32>>) -> Option<f32> {
    match value {
        Some(UiValue::Static(value)) if value.is_finite() => Some(*value),
        _ => None,
    }
}

fn propagate_layout(layout: &mut UiLayout) -> bool {
    let mut changed = false;
    changed |= normalize_length_value(&mut layout.width);
    changed |= normalize_length_value(&mut layout.height);
    changed |= normalize_optional_nonnegative(&mut layout.min_width);
    changed |= normalize_optional_nonnegative(&mut layout.min_height);
    changed |= normalize_optional_nonnegative(&mut layout.max_width);
    changed |= normalize_optional_nonnegative(&mut layout.max_height);
    changed |= normalize_padding(&mut layout.padding);

    if let (Some(min), Some(max)) = (static_f32(&layout.min_width), static_f32(&layout.max_width)) {
        if min > max {
            layout.max_width = Some(UiValue::Static(min));
            changed = true;
        }
    }
    if let (Some(min), Some(max)) = (static_f32(&layout.min_height), static_f32(&layout.max_height)) {
        if min > max {
            layout.max_height = Some(UiValue::Static(min));
            changed = true;
        }
    }

    if let UiValue::Static(UiLength::Pixels(width)) = &mut layout.width {
        let old = *width;
        if let Some(min) = static_f32(&layout.min_width) {
            *width = width.max(min);
        }
        if let Some(max) = static_f32(&layout.max_width) {
            *width = width.min(max);
        }
        changed |= old.to_bits() != width.to_bits();
    }
    if let UiValue::Static(UiLength::Pixels(height)) = &mut layout.height {
        let old = *height;
        if let Some(min) = static_f32(&layout.min_height) {
            *height = height.max(min);
        }
        if let Some(max) = static_f32(&layout.max_height) {
            *height = height.min(max);
        }
        changed |= old.to_bits() != height.to_bits();
    }

    let static_ratio = match &layout.aspect_ratio {
        Some(UiValue::Static(ratio)) => Some(*ratio),
        _ => None,
    };
    if let Some(ratio) = static_ratio {
        if !ratio.is_finite() || ratio <= 0.0 {
            layout.aspect_ratio = None;
            changed = true;
        } else {
            match (&layout.width, &layout.height) {
                (UiValue::Static(UiLength::Pixels(width)), UiValue::Static(UiLength::Shrink)) => {
                    layout.height = UiValue::Static(UiLength::Pixels(*width / ratio));
                    changed = true;
                }
                (UiValue::Static(UiLength::Shrink), UiValue::Static(UiLength::Pixels(height))) => {
                    layout.width = UiValue::Static(UiLength::Pixels(*height * ratio));
                    changed = true;
                }
                _ => {}
            }
        }
    }
    changed
}

impl UiOptimization for LayoutConstantPropagation {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component in &mut program.components {
            for node in &mut component.nodes {
                changed |= propagate_layout(&mut node.layout);
                match &mut node.kind {
                    UiNodeKind::Container(widget) => {
                        changed |= normalize_optional_nonnegative(&mut widget.max_width);
                        changed |= normalize_optional_nonnegative(&mut widget.max_height);
                    }
                    UiNodeKind::Row(widget) | UiNodeKind::Column(widget) => {
                        changed |= normalize_nonnegative_value(&mut widget.spacing);
                        changed |= normalize_padding(&mut widget.padding);
                    }
                    UiNodeKind::Grid(widget) => {
                        changed |= normalize_nonnegative_value(&mut widget.row_spacing);
                        changed |= normalize_nonnegative_value(&mut widget.column_spacing);
                        match &mut widget.sizing {
                            UiGridSizing::AspectRatio(value) => {
                                if let UiValue::Static(value) = value {
                                    let old = *value;
                                    if !value.is_finite() || *value <= 0.0 { *value = 1.0; }
                                    changed |= old.to_bits() != value.to_bits();
                                }
                            }
                            UiGridSizing::EvenlyDistribute(value) => changed |= normalize_length_value(value),
                            UiGridSizing::Tracks(tracks) => {
                                for track in tracks {
                                    match track {
                                        UiGridTrack::Fixed(value) => {
                                            let old = *value;
                                            if !value.is_finite() || *value < 0.0 { *value = 0.0; }
                                            changed |= old.to_bits() != value.to_bits();
                                        }
                                        UiGridTrack::Fraction(value) if *value == 0 => {
                                            *value = 1;
                                            changed = true;
                                        }
                                        UiGridTrack::MinMax(min, max) => {
                                            changed |= normalize_length(min);
                                            changed |= normalize_length(max);
                                            if let (UiLength::Pixels(min_value), UiLength::Pixels(max_value)) = (*min, *max) {
                                                if min_value > max_value {
                                                    *max = UiLength::Pixels(min_value);
                                                    changed = true;
                                                }
                                                if min_value.to_bits() == max_value.to_bits() {
                                                    *track = UiGridTrack::Fixed(min_value);
                                                    changed = true;
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    UiNodeKind::Space(widget) => {
                        changed |= normalize_length_value(&mut widget.width);
                        changed |= normalize_length_value(&mut widget.height);
                    }
                    UiNodeKind::Rule(widget) => changed |= normalize_nonnegative_value(&mut widget.thickness),
                    UiNodeKind::TextInput(widget) => changed |= normalize_padding(&mut widget.padding),
                    UiNodeKind::TextEditor(widget) => {
                        changed |= normalize_padding(&mut widget.padding);
                        changed |= normalize_length_value(&mut widget.height);
                    }
                    UiNodeKind::Scrollable(widget) => {
                        if let Some(spacing) = &mut widget.spacing { changed |= normalize_nonnegative_value(spacing); }
                        for bar in [&mut widget.horizontal, &mut widget.vertical] {
                            if let Some(bar) = bar {
                                changed |= normalize_nonnegative_value(&mut bar.width);
                                changed |= normalize_nonnegative_value(&mut bar.margin);
                                changed |= normalize_nonnegative_value(&mut bar.scroller_width);
                                if let Some(spacing) = &mut bar.spacing { changed |= normalize_nonnegative_value(spacing); }
                            }
                        }
                    }
                    UiNodeKind::PaneGrid(widget) => {
                        changed |= normalize_nonnegative_value(&mut widget.spacing);
                        changed |= normalize_nonnegative_value(&mut widget.resize_leeway);
                    }
                    UiNodeKind::Tooltip(widget) => {
                        changed |= normalize_nonnegative_value(&mut widget.gap);
                        changed |= normalize_nonnegative_value(&mut widget.padding);
                    }
                    UiNodeKind::QrCode(widget) => {
                        if let Some(size) = &mut widget.cell_size { changed |= normalize_nonnegative_value(size); }
                    }
                    _ => {}
                }
            }
        }
        changed
    }
}

fn collect_binding_uses(program: &mut UiProgram, component_index: usize) -> Vec<BindingUseSummary> {
    let count = program.components[component_index].bindings.len();
    let mut uses = vec![BindingUseSummary::default(); count];
    {
        let component = &mut program.components[component_index];
        visit_component_dynamic_refs_mut(
            component,
            |id, class| {
                if let Some(summary) = uses.get_mut(id.0 as usize) {
                    summary.record(class);
                }
            },
            |_| {},
        );
    }
    let component_id = program.components[component_index].id;
    visit_owned_style_sheet_binding_refs_mut(program, component_id, |id, class| {
        if let Some(summary) = uses.get_mut(id.0 as usize) {
            summary.record(class);
        }
    });
    for animation in &program.animations {
        if let UiAnimationTrigger::Binding { component, binding } = &animation.trigger {
            if *component == component_id {
                if let Some(summary) = uses.get_mut(binding.0 as usize) {
                    summary.record(BindingUseClass::Immediate);
                }
            }
        }
    }
    uses
}

impl UiOptimization for BindingSchedulingClassification {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component_index in 0..program.components.len() {
            let uses = collect_binding_uses(program, component_index);
            for (binding, usage) in program.components[component_index].bindings.iter_mut().zip(uses) {
                let scheduling = if binding.constant.is_some() || usage.immediate || usage.structural {
                    UiBindingScheduling::Immediate
                } else if usage.layout {
                    UiBindingScheduling::BeforeLayout
                } else if usage.paint {
                    UiBindingScheduling::BeforePaint
                } else {
                    UiBindingScheduling::FrameCoalesced
                };
                if binding.scheduling != scheduling {
                    binding.scheduling = scheduling;
                    changed = true;
                }
                let layout_affecting = usage.structural || usage.layout;
                if binding.flags.layout_affecting != layout_affecting {
                    binding.flags.layout_affecting = layout_affecting;
                    changed = true;
                }
            }
        }
        changed
    }
}

impl UiOptimization for PaintOnlyInvalidationAnalysis {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component_index in 0..program.components.len() {
            let uses = collect_binding_uses(program, component_index);
            for (binding, usage) in program.components[component_index].bindings.iter_mut().zip(uses) {
                let paint_only = usage.paint && !usage.layout && !usage.structural && !usage.immediate;
                if binding.flags.paint_only != paint_only {
                    binding.flags.paint_only = paint_only;
                    changed = true;
                }
                let layout_affecting = usage.layout || usage.structural;
                if binding.flags.layout_affecting != layout_affecting {
                    binding.flags.layout_affecting = layout_affecting;
                    changed = true;
                }
            }
        }
        changed
    }
}

// =============================================================================
// Static subtree hoisting/cache classification
// =============================================================================

fn node_has_runtime_binding(node: &UiNode) -> bool {
    let mut node = node.clone();
    let mut found = false;
    let mut on_binding = |_: &mut UiBindingId, _: BindingUseClass| found = true;
    let mut on_token = |_: &mut UiStyleTokenId| {};
    if let Some(key) = &mut node.key {
        visit_value_ref(key, BindingUseClass::Structural, &mut on_binding, &mut on_token);
    }
    visit_layout_dynamic_refs_mut(&mut node.layout, &mut on_binding, &mut on_token);
    visit_style_patch_dynamic_refs_mut(&mut node.style.inline, &mut on_binding, &mut on_token);
    visit_value_ref(&mut node.visible, BindingUseClass::Structural, &mut on_binding, &mut on_token);
    visit_value_ref(&mut node.enabled, BindingUseClass::Paint, &mut on_binding, &mut on_token);
    visit_value_ref(&mut node.opacity, BindingUseClass::Paint, &mut on_binding, &mut on_token);
    visit_value_ref(&mut node.transform, BindingUseClass::Paint, &mut on_binding, &mut on_token);
    if let UiClip::Rounded(radius) = &mut node.clip {
        visit_value_ref(radius, BindingUseClass::Paint, &mut on_binding, &mut on_token);
    }
    visit_opt_value_ref(&mut node.cursor, BindingUseClass::Paint, &mut on_binding, &mut on_token);
    if let Some(accessibility) = &mut node.accessibility {
        visit_accessibility_dynamic_refs_mut(accessibility, &mut on_binding, &mut on_token);
    }
    visit_node_kind_dynamic_refs_mut(&mut node.kind, &mut on_binding, &mut on_token);
    found
}

fn render_pure_node(program: &UiProgram, node: &UiNode) -> bool {
    if !node.handlers.is_empty()
        || !node.style.static_transitions.is_empty()
        || !node.style.dynamic_rules.is_empty()
        || node_has_runtime_binding(node)
    {
        return false;
    }
    match &node.kind {
        UiNodeKind::Fragment
        | UiNodeKind::Container(_)
        | UiNodeKind::Row(_)
        | UiNodeKind::Column(_)
        | UiNodeKind::Grid(_)
        | UiNodeKind::Stack(_)
        | UiNodeKind::Space(_)
        | UiNodeKind::Rule(_)
        | UiNodeKind::Text(_)
        | UiNodeKind::Image(_)
        | UiNodeKind::Svg(_)
        | UiNodeKind::QrCode(_) => true,
        UiNodeKind::RichText(text) => text.on_link.is_none(),
        UiNodeKind::Markdown(markdown) => markdown.on_link.is_none(),
        UiNodeKind::Canvas(canvas) => program
            .canvas_programs
            .get(canvas.program.0 as usize)
            .is_some_and(|canvas| canvas.dynamic_builder.is_none()),
        UiNodeKind::Custom(custom) => !custom.stateful,
        _ => false,
    }
}

fn static_subtree(
    program: &UiProgram,
    component: &UiComponent,
    id: UiNodeId,
    memo: &mut [Option<bool>],
    visiting: &mut HashSet<UiNodeId>,
) -> bool {
    if let Some(value) = memo.get(id.0 as usize).copied().flatten() {
        return value;
    }
    if !visiting.insert(id) {
        return false;
    }
    let Some(node) = component.node(id) else { return false; };
    let value = render_pure_node(program, node)
        && node.children.iter().copied().all(|child| static_subtree(program, component, child, memo, visiting));
    visiting.remove(&id);
    if let Some(slot) = memo.get_mut(id.0 as usize) {
        *slot = Some(value);
    }
    value
}

impl UiOptimization for StaticSubtreeHoisting {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let snapshot = program.clone();
        let mut changed = false;
        for component_index in 0..program.components.len() {
            let component = &snapshot.components[component_index];
            let mut memo = vec![None; component.nodes.len()];
            let mut visiting = HashSet::default();
            for node in &component.nodes {
                let is_static = static_subtree(&snapshot, component, node.id, &mut memo, &mut visiting);
                if is_static {
                    let live = &mut program.components[component_index].nodes[node.id.0 as usize];
                    if live.cache == UiCachePolicy::Automatic {
                        live.cache = UiCachePolicy::Subtree;
                        changed = true;
                    }
                }
            }
        }
        changed
    }
}

// =============================================================================
// Resource deduplication
// =============================================================================

fn resource_fingerprint(resource: &UiResource) -> u64 {
    let mut hasher = StableHasher::new();
    resource.kind.hash(&mut hasher);
    resource.cache.hash(&mut hasher);
    match &resource.source {
        UiResourceSource::Path(path) => {
            hasher.write_u8(0);
            path.hash(&mut hasher);
        }
        UiResourceSource::Url(url) => {
            hasher.write_u8(1);
            url.hash(&mut hasher);
        }
        UiResourceSource::Embedded(bytes) => {
            hasher.write_u8(2);
            hasher.write(bytes);
        }
        UiResourceSource::Generated(function) => {
            hasher.write_u8(3);
            function.hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn resource_equivalent(left: &UiResource, right: &UiResource) -> bool {
    left.kind == right.kind && left.source == right.source && left.cache == right.cache
}

fn remap_resource_id(id: &mut UiResourceId, map: &[UiResourceId]) {
    if let Some(mapped) = map.get(id.0 as usize) {
        *id = *mapped;
    }
}

fn remap_font_resources(font: &mut UiFont, map: &[UiResourceId]) {
    if let UiFont::Resource(resource) = font {
        remap_resource_id(resource, map);
    }
}

fn remap_value_font_resources(value: &mut UiValue<UiFont>, map: &[UiResourceId]) {
    if let UiValue::Static(font) = value {
        remap_font_resources(font, map);
    }
}

fn remap_opt_value_font_resources(value: &mut Option<UiValue<UiFont>>, map: &[UiResourceId]) {
    if let Some(value) = value {
        remap_value_font_resources(value, map);
    }
}

fn remap_value_resource(value: &mut UiValue<UiResourceId>, map: &[UiResourceId]) {
    if let UiValue::Static(resource) = value {
        remap_resource_id(resource, map);
    }
}

fn remap_style_patch_resources(patch: &mut UiStylePatch, map: &[UiResourceId]) {
    remap_opt_value_font_resources(&mut patch.text.font, map);
    if let Some(widget) = &mut patch.widget {
        match widget {
            UiWidgetStylePatch::Markdown(style) => {
                remap_opt_value_font_resources(&mut style.font, map);
                remap_opt_value_font_resources(&mut style.inline_code_font, map);
                remap_opt_value_font_resources(&mut style.code_block_font, map);
            }
            _ => {}
        }
    }
}

fn remap_text_format_resources(format: &mut UiTextFormat, map: &[UiResourceId]) {
    remap_opt_value_font_resources(&mut format.font, map);
}

fn remap_component_resources(component: &mut UiComponent, map: &[UiResourceId]) {
    for binding in &mut component.bindings {
        for dependency in &mut binding.dependencies {
            if let UiDependency::Resource(resource) = dependency {
                remap_resource_id(resource, map);
            }
        }
        match &mut binding.constant {
            Some(UiConstantValue::Resource(resource)) => remap_resource_id(resource, map),
            Some(UiConstantValue::Font(font)) => remap_font_resources(font, map),
            _ => {}
        }
    }
    for computed in &mut component.computed {
        for dependency in &mut computed.dependencies {
            if let UiDependency::Resource(resource) = dependency {
                remap_resource_id(resource, map);
            }
        }
    }
    for window in &mut component.windows {
        if let Some(icon) = &mut window.settings.icon {
            remap_resource_id(icon, map);
        }
    }
    for node in &mut component.nodes {
        remap_style_patch_resources(&mut node.style.inline, map);
        match &mut node.kind {
            UiNodeKind::Text(text) => remap_text_format_resources(&mut text.format, map),
            UiNodeKind::RichText(text) => {
                remap_text_format_resources(&mut text.format, map);
                for span in &mut text.spans {
                    remap_opt_value_font_resources(&mut span.font, map);
                }
            }
            UiNodeKind::Checkbox(widget) => {
                if let Some(icon) = &mut widget.icon {
                    remap_font_resources(&mut icon.font, map);
                }
            }
            UiNodeKind::Radio(widget) => remap_text_format_resources(&mut widget.text, map),
            UiNodeKind::Toggler(widget) => remap_text_format_resources(&mut widget.text, map),
            UiNodeKind::TextInput(widget) => {
                remap_opt_value_font_resources(&mut widget.font, map);
                if let Some(icon) = &mut widget.icon {
                    remap_font_resources(&mut icon.font, map);
                }
            }
            UiNodeKind::TextEditor(widget) => remap_opt_value_font_resources(&mut widget.font, map),
            UiNodeKind::Image(widget) => remap_value_resource(&mut widget.source, map),
            UiNodeKind::Svg(widget) => remap_value_resource(&mut widget.source, map),
            _ => {}
        }
    }
}

impl UiOptimization for ResourceDeduplication {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        if program.resources.len() < 2 {
            return false;
        }

        let mut buckets: HashMap<u64, Vec<UiResourceId>> = HashMap::default();
        let mut aliases = vec![UiResourceId(0); program.resources.len()];
        let mut unique: Vec<UiResource> = Vec::new();

        for resource in &program.resources {
            let fingerprint = resource_fingerprint(resource);
            let bucket = buckets.entry(fingerprint).or_default();
            let existing = bucket.iter().copied().find(|id| {
                unique
                    .get(id.0 as usize)
                    .is_some_and(|candidate| resource_equivalent(candidate, resource))
            });
            if let Some(existing) = existing {
                aliases[resource.id.0 as usize] = existing;
                if unique[existing.0 as usize].name.is_none() && resource.name.is_some() {
                    unique[existing.0 as usize].name = resource.name;
                }
            } else {
                let id = UiResourceId(unique.len() as u32);
                aliases[resource.id.0 as usize] = id;
                let mut resource = resource.clone();
                resource.id = id;
                unique.push(resource);
                bucket.push(id);
            }
        }

        if unique.len() == program.resources.len() {
            return false;
        }
        program.resources = unique;

        for resource in &mut program.settings.preload_fonts {
            remap_resource_id(resource, &aliases);
        }
        if let Some(font) = &mut program.settings.default_font {
            remap_font_resources(font, &aliases);
        }
        for token in &mut program.style_tokens {
            if let UiStyleLiteral::Font(font) = &mut token.value {
                remap_font_resources(font, &aliases);
            }
        }
        for style in &mut program.static_styles {
            remap_style_patch_resources(&mut style.patch, &aliases);
        }
        for sheet in &mut program.style_sheets {
            for rule in &mut sheet.rules {
                remap_style_patch_resources(&mut rule.declarations, &aliases);
            }
        }
        for component in &mut program.components {
            remap_component_resources(component, &aliases);
        }
        for canvas in &mut program.canvas_programs {
            for command in &mut canvas.commands {
                match command {
                    UiCanvasCommand::Text(text) => remap_font_resources(&mut text.font, &aliases),
                    UiCanvasCommand::Image(image) => remap_resource_id(&mut image.resource, &aliases),
                    UiCanvasCommand::Svg(svg) => remap_resource_id(&mut svg.resource, &aliases),
                    _ => {}
                }
            }
        }
        for shader in &mut program.shader_programs {
            remap_resource_id(&mut shader.source, &aliases);
        }
        true
    }
}

// =============================================================================
// Canvas display-list folding
// =============================================================================

fn zero_f32(value: f32) -> bool {
    value.to_bits() == 0.0f32.to_bits() || value.to_bits() == (-0.0f32).to_bits()
}

fn one_f32(value: f32) -> bool {
    value.to_bits() == 1.0f32.to_bits()
}

fn identity_transform(transform: &UiTransform) -> bool {
    zero_f32(transform.translate.x)
        && zero_f32(transform.translate.y)
        && one_f32(transform.scale.x)
        && one_f32(transform.scale.y)
        && zero_f32(transform.rotation_radians)
        && transform.origin.x.to_bits() == 0.5f32.to_bits()
        && transform.origin.y.to_bits() == 0.5f32.to_bits()
}

fn canonicalize_canvas_path(path: &mut UiCanvasPath) -> bool {
    let old = path.segments.clone();
    let mut out = Vec::with_capacity(path.segments.len());
    let mut current: Option<UiPoint> = None;
    let mut closed = false;

    for segment in path.segments.drain(..) {
        match segment {
            UiPathSegment::MoveTo(point) => {
                if matches!(out.last(), Some(UiPathSegment::MoveTo(_))) {
                    out.pop();
                }
                current = Some(point);
                closed = false;
                out.push(UiPathSegment::MoveTo(point));
            }
            UiPathSegment::LineTo(point) if current == Some(point) => {}
            UiPathSegment::LineTo(point) => {
                current = Some(point);
                closed = false;
                out.push(UiPathSegment::LineTo(point));
            }
            UiPathSegment::QuadraticTo { control, to } => {
                current = Some(to);
                closed = false;
                out.push(UiPathSegment::QuadraticTo { control, to });
            }
            UiPathSegment::CubicTo { control1, control2, to } => {
                current = Some(to);
                closed = false;
                out.push(UiPathSegment::CubicTo { control1, control2, to });
            }
            UiPathSegment::Circle { center, radius } if !radius.is_finite() || radius <= 0.0 => {}
            UiPathSegment::Ellipse { center: _, radii, rotation_radians: _ }
                if !radii.x.is_finite() || !radii.y.is_finite() || radii.x <= 0.0 || radii.y <= 0.0 => {}
            UiPathSegment::Arc { center: _, radius, start_radians: _, end_radians: _, clockwise: _ }
                if !radius.is_finite() || radius <= 0.0 => {}
            UiPathSegment::Rectangle(rect)
                if !rect.width.is_finite() || !rect.height.is_finite() || rect.width <= 0.0 || rect.height <= 0.0 => {}
            UiPathSegment::RoundedRectangle { rect, .. }
                if !rect.width.is_finite() || !rect.height.is_finite() || rect.width <= 0.0 || rect.height <= 0.0 => {}
            UiPathSegment::Close if closed => {}
            UiPathSegment::Close => {
                closed = true;
                out.push(UiPathSegment::Close);
            }
            other => {
                closed = false;
                out.push(other);
            }
        }
    }
    path.segments = out;
    old != path.segments
}

fn canonicalize_canvas_command(command: &mut UiCanvasCommand) -> bool {
    match command {
        UiCanvasCommand::Clip(path) | UiCanvasCommand::FillPath { path, .. } => {
            canonicalize_canvas_path(path)
        }
        UiCanvasCommand::StrokePath { path, stroke } => {
            let mut changed = canonicalize_canvas_path(path);
            let old = stroke.width;
            if !stroke.width.is_finite() || stroke.width < 0.0 {
                stroke.width = 0.0;
            }
            changed |= old.to_bits() != stroke.width.to_bits();
            changed
        }
        UiCanvasCommand::Text(text) => {
            let old_size = text.size;
            let old_width = text.max_width;
            if !text.size.is_finite() || text.size < 0.0 { text.size = 0.0; }
            if !text.max_width.is_finite() || text.max_width < 0.0 { text.max_width = 0.0; }
            old_size.to_bits() != text.size.to_bits() || old_width.to_bits() != text.max_width.to_bits()
        }
        UiCanvasCommand::Image(image) => {
            let old = image.opacity;
            image.opacity = if image.opacity.is_finite() { image.opacity.clamp(0.0, 1.0) } else { 1.0 };
            old.to_bits() != image.opacity.to_bits()
        }
        _ => false,
    }
}

impl UiOptimization for CanvasDisplayListFolding {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for canvas in &mut program.canvas_programs {
            let mut commands = Vec::with_capacity(canvas.commands.len());
            for mut command in std::mem::take(&mut canvas.commands) {
                changed |= canonicalize_canvas_command(&mut command);
                let noop = match &command {
                    UiCanvasCommand::Transform(transform) => identity_transform(transform),
                    UiCanvasCommand::Translate(vector) => zero_f32(vector.x) && zero_f32(vector.y),
                    UiCanvasCommand::Rotate(angle) => zero_f32(*angle),
                    UiCanvasCommand::Scale(vector) => one_f32(vector.x) && one_f32(vector.y),
                    UiCanvasCommand::Clip(path)
                    | UiCanvasCommand::FillPath { path, .. }
                    | UiCanvasCommand::StrokePath { path, .. } => path.segments.is_empty(),
                    _ => false,
                };
                if noop {
                    changed = true;
                    continue;
                }

                match (commands.last_mut(), &command) {
                    (Some(UiCanvasCommand::Translate(previous)), UiCanvasCommand::Translate(next)) => {
                        previous.x += next.x;
                        previous.y += next.y;
                        changed = true;
                    }
                    (Some(UiCanvasCommand::Rotate(previous)), UiCanvasCommand::Rotate(next)) => {
                        *previous += *next;
                        changed = true;
                    }
                    (Some(UiCanvasCommand::Scale(previous)), UiCanvasCommand::Scale(next)) => {
                        previous.x *= next.x;
                        previous.y *= next.y;
                        changed = true;
                    }
                    _ => commands.push(command),
                }
            }

            // A merge can itself produce identity transforms (for example +x and
            // -x translations), so run one cheap no-op filter before balancing
            // adjacent save/restore pairs.
            let old_len = commands.len();
            commands.retain(|command| match command {
                UiCanvasCommand::Transform(transform) => !identity_transform(transform),
                UiCanvasCommand::Translate(vector) => !(zero_f32(vector.x) && zero_f32(vector.y)),
                UiCanvasCommand::Rotate(angle) => !zero_f32(*angle),
                UiCanvasCommand::Scale(vector) => !(one_f32(vector.x) && one_f32(vector.y)),
                _ => true,
            });
            changed |= old_len != commands.len();

            // Remove adjacent save/restore pairs after no-op folding.
            let mut folded = Vec::with_capacity(commands.len());
            let mut index = 0usize;
            while index < commands.len() {
                if index + 1 < commands.len()
                    && matches!(commands[index], UiCanvasCommand::Save)
                    && matches!(commands[index + 1], UiCanvasCommand::Restore)
                {
                    changed = true;
                    index += 2;
                } else {
                    folded.push(commands[index].clone());
                    index += 1;
                }
            }
            canvas.commands = folded;
        }
        changed
    }
}

// =============================================================================
// Animation canonicalization
// =============================================================================

fn canonicalize_easing(easing: &mut UiEasing) -> bool {
    match easing {
        UiEasing::CubicBezier(x1, y1, x2, y2) => {
            let old = (*x1, *y1, *x2, *y2);
            *x1 = if x1.is_finite() { x1.clamp(0.0, 1.0) } else { 0.0 };
            *x2 = if x2.is_finite() { x2.clamp(0.0, 1.0) } else { 1.0 };
            if !y1.is_finite() { *y1 = 0.0; }
            if !y2.is_finite() { *y2 = 1.0; }
            old.0.to_bits() != x1.to_bits()
                || old.1.to_bits() != y1.to_bits()
                || old.2.to_bits() != x2.to_bits()
                || old.3.to_bits() != y2.to_bits()
        }
        UiEasing::Steps { count, .. } if *count == 0 => {
            *count = 1;
            true
        }
        UiEasing::Spring { mass, stiffness, damping, initial_velocity } => {
            let old = (*mass, *stiffness, *damping, *initial_velocity);
            if !mass.is_finite() || *mass <= 0.0 { *mass = 1.0; }
            if !stiffness.is_finite() || *stiffness <= 0.0 { *stiffness = 100.0; }
            if !damping.is_finite() || *damping < 0.0 { *damping = 10.0; }
            if !initial_velocity.is_finite() { *initial_velocity = 0.0; }
            old.0.to_bits() != mass.to_bits()
                || old.1.to_bits() != stiffness.to_bits()
                || old.2.to_bits() != damping.to_bits()
                || old.3.to_bits() != initial_velocity.to_bits()
        }
        _ => false,
    }
}

impl UiOptimization for AnimationCanonicalization {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for animation in &mut program.animations {
            changed |= canonicalize_easing(&mut animation.easing);
            if matches!(animation.iterations, UiAnimationIterations::Count(0)) {
                if !animation.tracks.is_empty() {
                    animation.tracks.clear();
                    changed = true;
                }
                continue;
            }

            for track in &mut animation.tracks {
                for keyframe in &mut track.keyframes {
                    let old = keyframe.offset;
                    keyframe.offset = if keyframe.offset.is_finite() {
                        keyframe.offset.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    changed |= old.to_bits() != keyframe.offset.to_bits();
                    if let Some(easing) = &mut keyframe.easing {
                        changed |= canonicalize_easing(easing);
                    }
                }
                let old = track.keyframes.clone();
                track.keyframes.sort_by(|left, right| left.offset.total_cmp(&right.offset));
                let mut deduped: Vec<UiKeyframe> = Vec::with_capacity(track.keyframes.len());
                for keyframe in std::mem::take(&mut track.keyframes) {
                    if let Some(previous) = deduped.last_mut() {
                        if previous.offset.to_bits() == keyframe.offset.to_bits() {
                            *previous = keyframe;
                            changed = true;
                            continue;
                        }
                    }
                    deduped.push(keyframe);
                }
                changed |= old != deduped;
                track.keyframes = deduped;
            }
            let old_len = animation.tracks.len();
            animation.tracks.retain(|track| !track.keyframes.is_empty());
            changed |= old_len != animation.tracks.len();
        }
        changed
    }
}

// =============================================================================
// Dead retained-node elimination
// =============================================================================

fn node_references(node: &UiNode, out: &mut Vec<UiNodeId>) {
    structural_child_ids(node, out);
}

fn mark_live_node(component: &UiComponent, id: UiNodeId, live: &mut [bool]) {
    let Some(slot) = live.get_mut(id.0 as usize) else { return; };
    if *slot {
        return;
    }
    *slot = true;
    let Some(node) = component.node(id) else { return; };
    let mut references = Vec::new();
    node_references(node, &mut references);
    for reference in references {
        mark_live_node(component, reference, live);
    }
}

fn compact_component_nodes(program: &mut UiProgram, component_index: usize) -> bool {
    let component_id = program.components[component_index].id;
    let snapshot = program.components[component_index].clone();
    if snapshot.nodes.is_empty() {
        return false;
    }

    let mut live = vec![false; snapshot.nodes.len()];
    if let Some(root) = snapshot.embedded_root {
        mark_live_node(&snapshot, root, &mut live);
    }
    for window in &snapshot.windows {
        mark_live_node(&snapshot, window.root, &mut live);
    }
    for slot in &snapshot.slots {
        if let Some(fallback) = slot.fallback {
            mark_live_node(&snapshot, fallback, &mut live);
        }
    }
    for binding in &snapshot.bindings {
        if let Some(UiConstantValue::Node(node)) = binding.constant.as_ref() {
            mark_live_node(&snapshot, *node, &mut live);
        }
    }
    for animation in &program.animations {
        for track in &animation.tracks {
            if let UiAnimationTarget::Node { component, node } = &track.target {
                if *component == component_id {
                    mark_live_node(&snapshot, *node, &mut live);
                }
            }
        }
    }

    if live.iter().all(|live| *live) {
        return false;
    }

    let mut map = vec![UiNodeId(0); snapshot.nodes.len()];
    let mut nodes = Vec::with_capacity(live.iter().filter(|live| **live).count());
    for (old_index, mut node) in std::mem::take(&mut program.components[component_index].nodes)
        .into_iter()
        .enumerate()
    {
        if live[old_index] {
            let id = UiNodeId(nodes.len() as u32);
            map[old_index] = id;
            node.id = id;
            nodes.push(node);
        }
    }
    for node in &mut nodes {
        remap_node_refs_in_node(node, &map);
    }

    let component = &mut program.components[component_index];
    component.nodes = nodes;
    if let Some(root) = &mut component.embedded_root { remap_node_id(root, &map); }
    for window in &mut component.windows { remap_node_id(&mut window.root, &map); }
    for slot in &mut component.slots {
        if let Some(fallback) = &mut slot.fallback { remap_node_id(fallback, &map); }
    }
    for binding in &mut component.bindings {
        if let Some(UiConstantValue::Node(node)) = &mut binding.constant {
            remap_node_id(node, &map);
        }
    }
    for animation in &mut program.animations {
        for track in &mut animation.tracks {
            if let UiAnimationTarget::Node { component, node } = &mut track.target {
                if *component == component_id {
                    remap_node_id(node, &map);
                }
            }
        }
    }
    true
}

impl UiOptimization for DeadNodeElimination {
    fn optimize_ui(&mut self, program: &mut UiProgram) -> bool {
        let mut changed = false;
        for component_index in 0..program.components.len() {
            changed |= compact_component_nodes(program, component_index);
            changed |= compact_bindings(program, component_index);
        }
        changed
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::napalm_ir::{ModuleId, SourceSpan, StringId, Type};

    fn span() -> SourceSpan {
        SourceSpan { file: None, line: 0, column: 0, length: 0 }
    }

    fn default_node(id: u32, kind: UiNodeKind) -> UiNode {
        UiNode {
            id: UiNodeId(id),
            kind,
            children: Vec::new(),
            key: None,
            name: None,
            classes: Vec::new(),
            layout: UiLayout::default(),
            style: UiNodeStyle::default(),
            visible: UiValue::Static(true),
            enabled: UiValue::Static(true),
            opacity: UiValue::Static(1.0),
            transform: UiValue::Static(UiTransform::default()),
            clip: UiClip::None,
            cursor: None,
            handlers: Vec::new(),
            accessibility: None,
            cache: UiCachePolicy::Automatic,
            source: span(),
        }
    }

    pub fn minimal_program() -> UiProgram {
        let mut program = UiProgram::default();
        program.components.push(UiComponent {
            id: UiComponentId(0),
            module: ModuleId(0),
            name: StringId(0),
            kind: UiComponentKind::Component,
            properties: Vec::new(),
            states: Vec::new(),
            computed: Vec::new(),
            bindings: Vec::new(),
            handlers: Vec::new(),
            subscriptions: Vec::new(),
            nodes: vec![default_node(0, UiNodeKind::Fragment)],
            embedded_root: Some(UiNodeId(0)),
            windows: Vec::new(),
            style_sheets: Vec::new(),
            theme: None,
            slots: Vec::new(),
            lifecycle: UiLifecycle::default(),
            source: span(),
        });
        program
    }

    #[test]
    fn folds_constant_if_and_eliminates_dead_node() {
        let mut program = minimal_program();
        let component = &mut program.components[0];
        component.bindings.push(UiBinding {
            id: UiBindingId(0),
            function: crate::napalm_ir::FuncId(0),
            result: UiValueType::Bool,
            dependencies: Vec::new(),
            equality: UiEqualityPolicy::Value,
            scheduling: UiBindingScheduling::Immediate,
            flags: UiBindingFlags { pure: true, memoizable: true, layout_affecting: true, paint_only: false },
            constant: Some(UiConstantValue::Bool(true)),
            source: span(),
        });
        component.nodes[0].kind = UiNodeKind::If(UiIfNode {
            condition: UiBindingId(0),
            then_node: UiNodeId(1),
            else_node: Some(UiNodeId(2)),
        });
        component.nodes.push(default_node(1, UiNodeKind::Text(UiTextNode {
            content: UiValue::Static(StringId(1)),
            format: UiTextFormat::default(),
        })));
        component.nodes.push(default_node(2, UiNodeKind::Text(UiTextNode {
            content: UiValue::Static(StringId(2)),
            format: UiTextFormat::default(),
        })));

        StaticConditionFold.optimize_ui(&mut program);
        DeadNodeElimination.optimize_ui(&mut program);
        assert_eq!(program.components[0].nodes.len(), 2);
        assert!(matches!(&program.components[0].nodes[0].kind, UiNodeKind::Fragment));
    }

    #[test]
    fn cse_merges_pure_bindings() {
        let mut program = minimal_program();
        let component = &mut program.components[0];
        for id in 0..2 {
            component.bindings.push(UiBinding {
                id: UiBindingId(id),
                function: crate::napalm_ir::FuncId(7),
                result: UiValueType::Number,
                dependencies: vec![UiDependency::State(UiStateId(0))],
                equality: UiEqualityPolicy::Value,
                scheduling: UiBindingScheduling::BeforePaint,
                flags: UiBindingFlags { pure: true, memoizable: true, layout_affecting: false, paint_only: true },
                constant: None,
                source: span(),
            });
        }
        component.nodes[0].opacity = UiValue::Binding(UiBindingId(1));
        BindingCSE.optimize_ui(&mut program);
        assert_eq!(program.components[0].bindings.len(), 1);
        assert!(matches!(&program.components[0].nodes[0].opacity, UiValue::Binding(UiBindingId(0))));
    }

    #[test]
    fn cse_rewrites_component_owned_style_sheet_bindings() {
        let mut program = minimal_program();
        let component = &mut program.components[0];
        for id in 0..2 {
            component.bindings.push(UiBinding {
                id: UiBindingId(id),
                function: crate::napalm_ir::FuncId(11),
                result: UiValueType::Number,
                dependencies: Vec::new(),
                equality: UiEqualityPolicy::Value,
                scheduling: UiBindingScheduling::BeforePaint,
                flags: UiBindingFlags {
                    pure: true,
                    memoizable: true,
                    layout_affecting: false,
                    paint_only: true,
                },
                constant: None,
                source: span(),
            });
        }
        component.style_sheets.push(UiStyleSheetId(0));

        let mut declarations = UiStylePatch::default();
        declarations.common.opacity = Some(UiValue::Binding(UiBindingId(1)));
        program.style_sheets.push(UiStyleSheet {
            id: UiStyleSheetId(0),
            name: None,
            owner: Some(UiComponentId(0)),
            rules: vec![UiStyleRule {
                id: UiStyleRuleId(0),
                selector: UiSelector { chain: Vec::new() },
                media: None,
                declarations,
                transitions: Vec::new(),
                origin: UiStyleOrigin::Component,
                layer: 0,
                important: false,
                specificity: 0,
                order: 0,
                source: span(),
            }],
        });

        BindingCSE.optimize_ui(&mut program);

        assert_eq!(program.components[0].bindings.len(), 1);
        let opacity = program.style_sheets[0].rules[0]
            .declarations
            .common
            .opacity
            .as_ref()
            .expect("style opacity binding must survive CSE");
        assert!(matches!(opacity, UiValue::Binding(UiBindingId(0))));
        program.verify().unwrap();
    }

    #[test]
    fn deduplicates_resources_and_rewrites_image() {
        let mut program = minimal_program();
        for id in 0..2 {
            program.resources.push(UiResource {
                id: UiResourceId(id),
                name: None,
                kind: UiResourceKind::Image,
                source: UiResourceSource::Path(StringId(9)),
                cache: UiResourceCache::Strong,
            });
        }
        program.components[0].nodes[0].kind = UiNodeKind::Image(UiImageNode {
            source: UiValue::Static(UiResourceId(1)),
            content_fit: UiContentFit::Contain,
            filter: UiImageFilter::Linear,
            rotation: UiValue::Static(UiRotation::FloatingRadians(0.0)),
            opacity: UiValue::Static(1.0),
            scale: UiValue::Static(1.0),
            border_radius: UiValue::Static(UiRadius::default()),
            snap: false,
        });
        ResourceDeduplication.optimize_ui(&mut program);
        assert_eq!(program.resources.len(), 1);
        let UiNodeKind::Image(image) = &program.components[0].nodes[0].kind else { panic!() };
        assert!(matches!(&image.source, UiValue::Static(UiResourceId(0))));
    }



    #[test]
    fn precomputes_and_interns_identical_static_styles() {
        let mut program = minimal_program();
        let component = &mut program.components[0];
        component.nodes[0].children = vec![UiNodeId(1), UiNodeId(2)];
        component.nodes.push(default_node(1, UiNodeKind::Text(UiTextNode {
            content: UiValue::Static(StringId(1)),
            format: UiTextFormat::default(),
        })));
        component.nodes.push(default_node(2, UiNodeKind::Text(UiTextNode {
            content: UiValue::Static(StringId(2)),
            format: UiTextFormat::default(),
        })));
        component.style_sheets.push(UiStyleSheetId(0));

        let mut declarations = UiStylePatch::default();
        declarations.common.opacity = Some(UiValue::Static(0.75));
        program.style_sheets.push(UiStyleSheet {
            id: UiStyleSheetId(0),
            name: None,
            owner: None,
            rules: vec![UiStyleRule {
                id: UiStyleRuleId(0),
                selector: UiSelector {
                    chain: vec![UiSelectorPart {
                        combinator: None,
                        atom: UiSelectorAtom {
                            widget: Some(UiWidgetTag::Text),
                            ..UiSelectorAtom::default()
                        },
                    }],
                },
                media: None,
                declarations,
                transitions: vec![UiTransition {
                    property: UiAnimatableProperty::Opacity,
                    duration_ms: 120,
                    delay_ms: 0,
                    easing: UiEasing::EaseOut,
                }],
                origin: UiStyleOrigin::Author,
                layer: 0,
                important: false,
                specificity: 0,
                order: 0,
                source: span(),
            }],
        });

        CanonicalizeSelectors.optimize_ui(&mut program);
        DeadStyleRuleElimination.optimize_ui(&mut program);
        StyleCascadePrecompute.optimize_ui(&mut program);
        StaticStyleInterning.optimize_ui(&mut program);

        assert_eq!(program.static_styles.len(), 1);
        assert_eq!(program.components[0].nodes[1].style.resolved_static, Some(UiStaticStyleId(0)));
        assert_eq!(program.components[0].nodes[2].style.resolved_static, Some(UiStaticStyleId(0)));
        assert_eq!(program.components[0].nodes[1].style.static_transitions.len(), 1);
        assert_eq!(program.components[0].nodes[1].style.static_transitions[0].property, UiAnimatableProperty::Opacity);
    }

    #[test]
    fn classifies_paint_only_binding_without_forcing_layout() {
        let mut program = minimal_program();
        program.components[0].bindings.push(UiBinding {
            id: UiBindingId(0),
            function: crate::napalm_ir::FuncId(3),
            result: UiValueType::Number,
            dependencies: Vec::new(),
            equality: UiEqualityPolicy::Value,
            scheduling: UiBindingScheduling::Immediate,
            flags: UiBindingFlags::default(),
            constant: None,
            source: span(),
        });
        program.components[0].nodes[0].opacity = UiValue::Binding(UiBindingId(0));

        BindingSchedulingClassification.optimize_ui(&mut program);
        PaintOnlyInvalidationAnalysis.optimize_ui(&mut program);
        let binding = &program.components[0].bindings[0];
        assert_eq!(binding.scheduling, UiBindingScheduling::BeforePaint);
        assert!(!binding.flags.layout_affecting);
        assert!(binding.flags.paint_only);
    }

    #[test]
    fn animation_canonicalization_sorts_and_deduplicates_keyframes() {
        let mut program = minimal_program();
        program.animations.push(UiAnimation {
            id: UiAnimationId(0),
            name: StringId(5),
            trigger: UiAnimationTrigger::Mount,
            tracks: vec![UiAnimationTrack {
                target: UiAnimationTarget::Node {
                    component: UiComponentId(0),
                    node: UiNodeId(0),
                },
                property: UiAnimatableProperty::Opacity,
                keyframes: vec![
                    UiKeyframe {
                        offset: 1.2,
                        value: UiAnimatedValue::Number(1.0),
                        easing: None,
                    },
                    UiKeyframe {
                        offset: -1.0,
                        value: UiAnimatedValue::Number(0.0),
                        easing: None,
                    },
                    UiKeyframe {
                        offset: 1.0,
                        value: UiAnimatedValue::Number(0.5),
                        easing: None,
                    },
                ],
            }],
            duration_ms: 100,
            delay_ms: 0,
            iterations: UiAnimationIterations::Once,
            direction: UiAnimationDirection::Normal,
            fill: UiAnimationFill::Forwards,
            easing: UiEasing::Linear,
        });

        AnimationCanonicalization.optimize_ui(&mut program);
        let keyframes = &program.animations[0].tracks[0].keyframes;
        assert_eq!(keyframes.len(), 2);
        assert_eq!(keyframes[0].offset, 0.0);
        assert_eq!(keyframes[1].offset, 1.0);
        assert!(matches!(&keyframes[1].value, UiAnimatedValue::Number(value) if *value == 0.5));
    }

    #[test]
    fn default_pipeline_keeps_minimal_program_verified() {
        let mut program = minimal_program();
        optimization_passes().optimize_ui(&mut program).unwrap();
        program.verify().unwrap();
    }

    #[test]
    fn canvas_folds_adjacent_translations() {
        let mut program = minimal_program();
        program.canvas_programs.push(UiCanvasProgram {
            id: UiCanvasProgramId(0),
            commands: vec![
                UiCanvasCommand::Translate(UiVector { x: 2.0, y: 3.0 }),
                UiCanvasCommand::Translate(UiVector { x: 4.0, y: 5.0 }),
                UiCanvasCommand::Save,
                UiCanvasCommand::Restore,
            ],
            dynamic_builder: None,
        });
        CanvasDisplayListFolding.optimize_ui(&mut program);
        assert_eq!(program.canvas_programs[0].commands.len(), 1);
        let UiCanvasCommand::Translate(value) = &program.canvas_programs[0].commands[0] else { panic!() };
        assert_eq!(*value, UiVector { x: 6.0, y: 8.0 });
    }
}

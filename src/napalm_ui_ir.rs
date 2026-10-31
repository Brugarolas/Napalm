//! Backend-neutral retained UI IR for Napalm.
//!
//! The IR is intentionally not an `iced` AST. Napalm owns components, state,
//! styling, reactivity, accessibility, animation, resources and lifecycle.
//! Backends translate this stable representation to Iced, another native UI
//! toolkit, or a future custom renderer.
//!
//! Design rules:
//! - UI structure is retained and arena-indexed by compact IDs.
//! - Arbitrary computation stays in WIR and is referenced through `FuncId`.
//! - Dynamic UI values are represented by `UiValue<T>` and `UiBinding`.
//! - Bindings carry explicit dependency metadata for fine-grained invalidation.
//! - Styles are selector-based, cascadeable and backend-neutral, but include
//!   typed patches for the complete Iced 0.14 widget styling surface.
//! - Widget-local transient state is backend-owned and keyed by stable node IDs.
//! - Control-flow nodes (`If`, `Switch`, `ForEach`, `Responsive`) are explicit so
//!   they can be optimized before materializing backend widgets.
//! - No `iced` type appears in this file.

#![allow(clippy::large_enum_variant)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::fmt;

use crate::napalm_ir::{FuncId, ModuleId, SourceSpan, StringId, Type};

macro_rules! ui_id {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
    };
}

ui_id!(UiComponentId);
ui_id!(UiNodeId);
ui_id!(UiWindowId);
ui_id!(UiPropertyId);
ui_id!(UiStateId);
ui_id!(UiComputedId);
ui_id!(UiLocalId);
ui_id!(UiBindingId);
ui_id!(UiHandlerId);
ui_id!(UiThemeId);
ui_id!(UiStyleSheetId);
ui_id!(UiStyleRuleId);
ui_id!(UiStyleTokenId);
ui_id!(UiResourceId);
ui_id!(UiAnimationId);
ui_id!(UiCanvasProgramId);
ui_id!(UiShaderProgramId);
ui_id!(UiPaneId);
ui_id!(UiSlotId);
ui_id!(UiSubscriptionId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UiIrVersion {
    pub major: u16,
    pub minor: u16,
}

impl UiIrVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
}

#[derive(Clone, Debug, Default)]
pub struct UiProgram {
    pub version: UiIrVersion,
    pub settings: UiProgramSettings,
    pub required_features: UiFeatureRequirements,
    pub components: Vec<UiComponent>,
    pub themes: Vec<UiTheme>,
    pub style_sheets: Vec<UiStyleSheet>,
    pub style_tokens: Vec<UiStyleToken>,
    pub resources: Vec<UiResource>,
    pub animations: Vec<UiAnimation>,
    pub canvas_programs: Vec<UiCanvasProgram>,
    pub shader_programs: Vec<UiShaderProgram>,
}

impl Default for UiIrVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

#[derive(Clone, Debug)]
pub struct UiProgramSettings {
    pub application_id: Option<StringId>,
    pub preload_fonts: Vec<UiResourceId>,
    pub default_font: Option<UiFont>,
    pub default_text_size: f32,
    pub antialiasing: bool,
    pub vsync: bool,
    pub renderer: UiRendererPreference,
}

impl Default for UiProgramSettings {
    fn default() -> Self {
        Self {
            application_id: None,
            preload_fonts: Vec::new(),
            default_font: None,
            default_text_size: 16.0,
            antialiasing: true,
            vsync: true,
            renderer: UiRendererPreference::Automatic,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UiFeatureRequirements {
    pub image: bool,
    pub svg: bool,
    pub canvas: bool,
    pub qr_code: bool,
    pub markdown: bool,
    pub highlighter: bool,
    pub shader_wgpu: bool,
    pub lazy: bool,
    pub advanced: bool,
}

#[derive(Clone, Debug, Default)]
pub enum UiRendererPreference {
    #[default]
    Automatic,
    Wgpu(UiWgpuSettings),
    TinySkia,
}

#[derive(Clone, Debug, Default)]
pub struct UiWgpuSettings {
    pub present_mode: Option<UiPresentMode>,
    pub backends: Vec<UiGraphicsBackend>,
    pub antialiasing: Option<UiMsaa>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiPresentMode {
    AutoVsync,
    AutoNoVsync,
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiGraphicsBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiMsaa {
    X2,
    X4,
    X8,
    X16,
}

// =============================================================================
// Component model and fine-grained reactivity
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiComponentKind {
    Application,
    Component,
}

#[derive(Clone, Debug)]
pub struct UiComponent {
    pub id: UiComponentId,
    pub module: ModuleId,
    pub name: StringId,
    pub kind: UiComponentKind,
    pub properties: Vec<UiProperty>,
    pub states: Vec<UiState>,
    pub computed: Vec<UiComputed>,
    pub bindings: Vec<UiBinding>,
    pub handlers: Vec<UiHandler>,
    pub subscriptions: Vec<UiSubscription>,
    pub nodes: Vec<UiNode>,
    pub embedded_root: Option<UiNodeId>,
    pub windows: Vec<UiWindow>,
    pub style_sheets: Vec<UiStyleSheetId>,
    pub theme: Option<UiValue<UiThemeId>>,
    pub slots: Vec<UiSlot>,
    pub lifecycle: UiLifecycle,
    pub source: SourceSpan,
}

#[derive(Clone, Debug)]
pub struct UiProperty {
    pub id: UiPropertyId,
    pub name: StringId,
    pub ty: Type,
    pub mutable: bool,
    pub required: bool,
    pub default: Option<UiBindingId>,
    pub source: SourceSpan,
}

#[derive(Clone, Debug)]
pub struct UiState {
    pub id: UiStateId,
    pub name: StringId,
    pub ty: Type,
    pub initial: UiBindingId,
    pub persistence: UiStatePersistence,
    pub source: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiStatePersistence {
    Ephemeral,
    ComponentLifetime,
    WindowLifetime,
    Persistent,
}

#[derive(Clone, Debug)]
pub struct UiComputed {
    pub id: UiComputedId,
    pub name: StringId,
    pub ty: Type,
    pub function: FuncId,
    pub dependencies: Vec<UiDependency>,
    pub equality: UiEqualityPolicy,
    pub source: SourceSpan,
}

#[derive(Clone, Debug)]
pub struct UiSlot {
    pub id: UiSlotId,
    pub name: StringId,
    pub multiple: bool,
    pub fallback: Option<UiNodeId>,
    pub source: SourceSpan,
}

#[derive(Clone, Debug, Default)]
pub struct UiLifecycle {
    pub on_mount: Option<FuncId>,
    pub on_unmount: Option<FuncId>,
    pub on_window_opened: Option<FuncId>,
    pub on_window_closed: Option<FuncId>,
    pub on_theme_changed: Option<FuncId>,
}

#[derive(Clone, Debug)]
pub struct UiBinding {
    pub id: UiBindingId,
    pub function: FuncId,
    pub result: UiValueType,
    pub dependencies: Vec<UiDependency>,
    pub equality: UiEqualityPolicy,
    pub scheduling: UiBindingScheduling,
    pub flags: UiBindingFlags,
    pub source: SourceSpan,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UiBindingFlags {
    pub pure: bool,
    pub memoizable: bool,
    pub layout_affecting: bool,
    pub paint_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiBindingScheduling {
    Immediate,
    BeforeLayout,
    BeforePaint,
    FrameCoalesced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiEqualityPolicy {
    AlwaysDirty,
    Value,
    Identity,
    Bitwise,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiDependency {
    Property(UiPropertyId),
    State(UiStateId),
    Computed(UiComputedId),
    Local(UiLocalId),
    StyleToken(UiStyleTokenId),
    Theme,
    WindowSize(UiWindowId),
    WindowScaleFactor(UiWindowId),
    Environment(UiEnvironmentKey),
    Resource(UiResourceId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiEnvironmentKey {
    ColorScheme,
    Contrast,
    ReducedMotion,
    Locale,
    TextScale,
    Platform,
    PointerKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiValueType {
    Napalm(Type),
    Bool,
    Number,
    String,
    Color,
    Background,
    Length,
    Padding,
    Border,
    Shadow,
    Transform,
    Font,
    Resource,
    Theme,
    Node,
    List(Box<UiValueType>),
    Any,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiValue<T> {
    Static(T),
    Binding(UiBindingId),
    Token(UiStyleTokenId),
}

impl<T> UiValue<T> {
    #[inline]
    pub const fn binding(id: UiBindingId) -> Self {
        Self::Binding(id)
    }

    #[inline]
    pub const fn token(id: UiStyleTokenId) -> Self {
        Self::Token(id)
    }
}

impl<T> From<T> for UiValue<T> {
    fn from(value: T) -> Self {
        Self::Static(value)
    }
}

#[derive(Clone, Debug)]
pub struct UiHandler {
    pub id: UiHandlerId,
    pub event: UiEventKind,
    pub function: FuncId,
    pub policy: UiEventPolicy,
    pub source: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiEventPolicy {
    Normal,
    Capture,
    StopPropagation,
    PreventDefault,
    CaptureAndStop,
}

#[derive(Clone, Debug)]
pub struct UiSubscription {
    pub id: UiSubscriptionId,
    pub source: UiSubscriptionSource,
    pub function: FuncId,
    pub source_span: SourceSpan,
}

#[derive(Clone, Debug)]
pub enum UiSubscriptionSource {
    Frames { window: Option<UiWindowId> },
    WindowEvents { window: Option<UiWindowId> },
    Keyboard,
    Mouse,
    Touch,
    InputMethod,
    SystemTheme,
    TimerEvery { milliseconds: u64 },
    TimerRepeat { milliseconds: u64, count: Option<u32> },
    Custom(StringId),
}

// =============================================================================
// Windows, application shell and imperative UI operations
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiWindow {
    pub id: UiWindowId,
    pub root: UiNodeId,
    pub title: UiValue<StringId>,
    pub settings: UiWindowSettings,
    pub theme: Option<UiValue<UiThemeId>>,
    pub handlers: Vec<UiHandlerId>,
    pub source: SourceSpan,
}

#[derive(Clone, Debug)]
pub struct UiWindowSettings {
    pub size: UiValue<UiSize>,
    pub min_size: Option<UiValue<UiSize>>,
    pub max_size: Option<UiValue<UiSize>>,
    pub position: UiWindowPosition,
    pub visible: UiValue<bool>,
    pub maximized: UiValue<bool>,
    pub fullscreen: UiValue<bool>,
    pub resizable: UiValue<bool>,
    pub closeable: UiValue<bool>,
    pub minimizable: UiValue<bool>,
    pub decorations: UiValue<bool>,
    pub transparent: UiValue<bool>,
    pub blur: UiValue<bool>,
    pub level: UiWindowLevel,
    pub icon: Option<UiResourceId>,
    pub platform: UiWindowPlatformSettings,
    pub exit_on_close_request: bool,
}


#[derive(Clone, Debug, Default)]
pub struct UiWindowPlatformSettings {
    pub windows: UiWindowsWindowSettings,
    pub macos: UiMacOsWindowSettings,
    pub linux: UiLinuxWindowSettings,
}

#[derive(Clone, Debug, Default)]
pub struct UiWindowsWindowSettings {
    pub drag_and_drop: Option<bool>,
    pub skip_taskbar: Option<bool>,
    pub undecorated_shadow: Option<bool>,
    pub corner_preference: Option<UiWindowsCornerPreference>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiWindowsCornerPreference {
    Default,
    DoNotRound,
    Round,
    RoundSmall,
}

#[derive(Clone, Debug, Default)]
pub struct UiMacOsWindowSettings {
    pub title_hidden: Option<bool>,
    pub titlebar_transparent: Option<bool>,
    pub fullsize_content_view: Option<bool>,
}

#[derive(Clone, Debug, Default)]
pub struct UiLinuxWindowSettings {
    pub application_id: Option<StringId>,
    pub override_redirect: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiWindowMode {
    Windowed,
    Fullscreen,
    Hidden,
}

#[derive(Clone, Debug)]
pub enum UiWindowPosition {
    Default,
    Centered,
    Specific(UiValue<UiPoint>),
    Computed(FuncId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiWindowLevel {
    Normal,
    AlwaysOnBottom,
    AlwaysOnTop,
}

#[derive(Clone, Debug)]
pub enum UiCommand {
    Focus(UiNodeId),
    Blur(UiNodeId),
    ScrollTo { node: UiNodeId, offset: UiPoint },
    ScrollBy { node: UiNodeId, delta: UiVector },
    SnapScrollable { node: UiNodeId, x: f32, y: f32 },
    Window(UiWindowCommand),
    ClipboardRead,
    ClipboardWrite(StringId),
    OpenUrl(StringId),
    RequestRedraw(Option<UiWindowId>),
    RequestLayout(Option<UiWindowId>),
    Widget(UiWidgetCommand),
}

#[derive(Clone, Debug)]
pub enum UiWidgetCommand {
    Focus(UiNodeId),
    Unfocus(UiNodeId),
    ScrollTo { node: UiNodeId, offset: UiPoint },
    SnapScrollable { node: UiNodeId, x: f32, y: f32 },
    TextInputMoveCursorToFront(UiNodeId),
    TextInputMoveCursorToEnd(UiNodeId),
    TextInputSelectAll(UiNodeId),
    TextEditorAction { node: UiNodeId, action: UiTextEditorAction },
    QueryVisibleBounds(UiNodeId),
}

#[derive(Clone, Debug)]
pub enum UiTextEditorAction {
    Move(UiTextCursorMotion),
    Select(UiTextCursorMotion),
    SelectAll,
    Edit(UiTextEditAction),
    Undo,
    Redo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTextCursorMotion {
    Left,
    Right,
    Up,
    Down,
    WordLeft,
    WordRight,
    LineStart,
    LineEnd,
    DocumentStart,
    DocumentEnd,
    PageUp,
    PageDown,
}

#[derive(Clone, Debug)]
pub enum UiTextEditAction {
    Insert(StringId),
    Paste(StringId),
    Enter,
    Backspace,
    Delete,
    Indent,
    Unindent,
}

#[derive(Clone, Debug)]
pub enum UiWindowCommand {
    Open(UiWindowId),
    Close(UiWindowId),
    Resize(UiWindowId, UiSize),
    Move(UiWindowId, UiPoint),
    Minimize(UiWindowId, bool),
    Maximize(UiWindowId, bool),
    ToggleMaximize(UiWindowId),
    SetFullscreen(UiWindowId, bool),
    SetMode(UiWindowId, UiWindowMode),
    SetResizable(UiWindowId, bool),
    SetResizeIncrements(UiWindowId, Option<UiSize>),
    ToggleDecorations(UiWindowId),
    SetLevel(UiWindowId, UiWindowLevel),
    SetMinSize(UiWindowId, Option<UiSize>),
    SetMaxSize(UiWindowId, Option<UiSize>),
    Drag(UiWindowId),
    DragResize(UiWindowId, UiResizeEdge),
    ShowSystemMenu(UiWindowId),
    RequestUserAttention(UiWindowId),
    GainFocus(UiWindowId),
    SetIcon(UiWindowId, Option<UiResourceId>),
    SetMousePassthrough(UiWindowId, bool),
    SetAllowAutomaticTabbing(bool),
    Screenshot(UiWindowId),
    RedrawAll,
    RelayoutAll,
}


#[derive(Clone, Debug)]
pub enum UiWindowQuery {
    Latest,
    Oldest,
    Size(UiWindowId),
    Maximized(UiWindowId),
    Minimized(UiWindowId),
    Position(UiWindowId),
    ScaleFactor(UiWindowId),
    Mode(UiWindowId),
    MonitorSize(UiWindowId),
    RawId(UiWindowId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiResizeEdge {
    North,
    South,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

// =============================================================================
// Retained node tree
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiNode {
    pub id: UiNodeId,
    pub kind: UiNodeKind,
    pub children: Vec<UiNodeId>,
    pub key: Option<UiValue<UiKey>>,
    pub name: Option<StringId>,
    pub classes: Vec<StringId>,
    pub layout: UiLayout,
    pub style: UiNodeStyle,
    pub visible: UiValue<bool>,
    pub enabled: UiValue<bool>,
    pub opacity: UiValue<f32>,
    pub transform: UiValue<UiTransform>,
    pub clip: UiClip,
    pub cursor: Option<UiValue<UiCursor>>, 
    pub handlers: Vec<UiHandlerId>,
    pub accessibility: Option<UiAccessibility>,
    pub cache: UiCachePolicy,
    pub source: SourceSpan,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiKey {
    String(StringId),
    Integer(i64),
    Number(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiCachePolicy {
    Automatic,
    Never,
    Layout,
    Paint,
    Subtree,
}

#[derive(Clone, Debug)]
pub enum UiClip {
    None,
    Bounds,
    Rounded(UiValue<UiRadius>),
    Path(UiCanvasPath),
}

#[derive(Clone, Debug)]
pub enum UiNodeKind {
    Fragment,
    Container(UiContainerNode),
    Row(UiFlexNode),
    Column(UiFlexNode),
    Grid(UiGridNode),
    Stack(UiStackNode),
    Space(UiSpaceNode),
    Rule(UiRuleNode),

    Text(UiTextNode),
    RichText(UiRichTextNode),
    Markdown(UiMarkdownNode),

    Button(UiButtonNode),
    Checkbox(UiCheckboxNode),
    Radio(UiRadioNode),
    Toggler(UiTogglerNode),
    TextInput(UiTextInputNode),
    TextEditor(UiTextEditorNode),
    Slider(UiSliderNode),
    VerticalSlider(UiSliderNode),
    PickList(UiPickListNode),
    ComboBox(UiComboBoxNode),
    ProgressBar(UiProgressBarNode),

    Scrollable(UiScrollableNode),
    PaneGrid(UiPaneGridNode),
    Table(UiTableNode),
    Tooltip(UiTooltipNode),
    MouseArea(UiMouseAreaNode),
    Hover(UiHoverNode),
    Float(UiFloatNode),
    Pin(UiPinNode),
    Responsive(UiResponsiveNode),
    Sensor(UiSensorNode),
    Themer(UiThemerNode),
    Opaque,
    Lazy(UiLazyNode),

    Image(UiImageNode),
    Svg(UiSvgNode),
    Canvas(UiCanvasNode),
    Shader(UiShaderNode),
    QrCode(UiQrCodeNode),

    Component(UiComponentInstanceNode),
    Slot(UiSlotNode),
    If(UiIfNode),
    Switch(UiSwitchNode),
    ForEach(UiForEachNode),
    Portal(UiPortalNode),
    Custom(UiCustomWidgetNode),
}

#[derive(Clone, Debug, Default)]
pub struct UiContainerNode {
    pub align_x: UiHorizontal,
    pub align_y: UiVertical,
    pub max_width: Option<UiValue<f32>>,
    pub max_height: Option<UiValue<f32>>,
}

#[derive(Clone, Debug)]
pub struct UiFlexNode {
    pub spacing: UiValue<f32>,
    pub padding: UiValue<UiPadding>,
    pub align: UiCrossAxisAlignment,
    pub justify: UiMainAxisAlignment,
    pub wrap: UiFlexWrap,
    pub clip_overflow: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiFlexWrap {
    NoWrap,
    Wrap,
    WrapReverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiCrossAxisAlignment {
    Start,
    Center,
    End,
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiMainAxisAlignment {
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Clone, Debug)]
pub struct UiGridNode {
    pub columns: UiValue<u16>,
    pub row_spacing: UiValue<f32>,
    pub column_spacing: UiValue<f32>,
    pub sizing: UiGridSizing,
    pub align_x: UiHorizontal,
    pub align_y: UiVertical,
}

#[derive(Clone, Debug)]
pub enum UiGridSizing {
    AspectRatio(UiValue<f32>),
    EvenlyDistribute(UiValue<UiLength>),
    Tracks(Vec<UiGridTrack>),
}

#[derive(Clone, Debug)]
pub enum UiGridTrack {
    Auto,
    Fixed(f32),
    Fraction(u16),
    MinMax(UiLength, UiLength),
}

#[derive(Clone, Debug, Default)]
pub struct UiStackNode {
    pub fit: UiStackFit,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UiStackFit {
    #[default]
    Largest,
    First,
    Explicit,
}

#[derive(Clone, Debug)]
pub struct UiSpaceNode {
    pub width: UiValue<UiLength>,
    pub height: UiValue<UiLength>,
}

#[derive(Clone, Debug)]
pub struct UiRuleNode {
    pub orientation: UiOrientation,
    pub thickness: UiValue<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiOrientation {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug)]
pub struct UiTextNode {
    pub content: UiValue<StringId>,
    pub format: UiTextFormat,
}

#[derive(Clone, Debug)]
pub struct UiRichTextNode {
    pub spans: Vec<UiTextSpan>,
    pub format: UiTextFormat,
    pub on_link: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiTextSpan {
    pub text: UiValue<StringId>,
    pub font: Option<UiValue<UiFont>>,
    pub size: Option<UiValue<f32>>,
    pub color: Option<UiValue<UiColorRef>>,
    pub background: Option<UiValue<UiBackground>>,
    pub underline: bool,
    pub strikethrough: bool,
    pub link: Option<UiValue<StringId>>,
}

#[derive(Clone, Debug)]
pub struct UiMarkdownNode {
    pub source: UiValue<StringId>,
    pub text_size: Option<UiValue<f32>>,
    pub settings: UiMarkdownSettings,
    pub on_link: Option<UiHandlerId>,
}

#[derive(Clone, Debug, Default)]
pub struct UiMarkdownSettings {
    pub h1_size: Option<UiValue<f32>>,
    pub h2_size: Option<UiValue<f32>>,
    pub h3_size: Option<UiValue<f32>>,
    pub h4_size: Option<UiValue<f32>>,
    pub h5_size: Option<UiValue<f32>>,
    pub h6_size: Option<UiValue<f32>>,
    pub code_size: Option<UiValue<f32>>,
    pub spacing: Option<UiValue<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiButtonNode {
    pub on_press: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiCheckboxNode {
    pub checked: UiValue<bool>,
    pub label: Option<UiValue<StringId>>,
    pub size: Option<UiValue<f32>>,
    pub spacing: Option<UiValue<f32>>,
    pub icon: Option<UiCheckboxIcon>,
    pub on_toggle: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiCheckboxIcon {
    pub codepoint: char,
    pub font: UiFont,
    pub size: Option<f32>,
    pub line_height: UiLineHeight,
    pub shaping: UiTextShaping,
}

#[derive(Clone, Debug)]
pub struct UiRadioNode {
    pub value: UiValue<UiRuntimeValue>,
    pub selected: UiValue<Option<UiRuntimeValue>>,
    pub label: UiValue<StringId>,
    pub size: Option<UiValue<f32>>,
    pub spacing: Option<UiValue<f32>>,
    pub text: UiTextFormat,
    pub on_select: UiHandlerId,
}

#[derive(Clone, Debug)]
pub struct UiTogglerNode {
    pub checked: UiValue<bool>,
    pub label: Option<UiValue<StringId>>,
    pub size: Option<UiValue<f32>>,
    pub spacing: Option<UiValue<f32>>,
    pub text: UiTextFormat,
    pub on_toggle: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiTextInputNode {
    pub value: UiValue<StringId>,
    pub placeholder: UiValue<StringId>,
    pub secure: bool,
    pub multiline: Option<UiTextWrapping>,
    pub icon: Option<UiTextInputIcon>,
    pub padding: UiValue<UiPadding>,
    pub size: Option<UiValue<f32>>,
    pub line_height: UiLineHeight,
    pub font: Option<UiValue<UiFont>>,
    pub align_x: UiTextAlignment,
    pub on_input: Option<UiHandlerId>,
    pub on_submit: Option<UiHandlerId>,
    pub on_paste: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiTextInputIcon {
    pub codepoint: char,
    pub font: UiFont,
    pub size: Option<f32>,
    pub spacing: f32,
    pub side: UiSide,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiSide {
    Left,
    Right,
}

#[derive(Clone, Debug)]
pub struct UiTextEditorNode {
    pub content: UiValue<StringId>,
    pub placeholder: UiValue<StringId>,
    pub padding: UiValue<UiPadding>,
    pub height: UiValue<UiLength>,
    pub font: Option<UiValue<UiFont>>,
    pub size: Option<UiValue<f32>>,
    pub line_height: UiLineHeight,
    pub wrapping: UiTextWrapping,
    pub highlighter: Option<UiHighlighter>,
    pub key_binding: Option<FuncId>,
    pub on_action: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiHighlighter {
    pub language: StringId,
    pub theme: Option<StringId>,
}

#[derive(Clone, Debug)]
pub struct UiSliderNode {
    pub min: UiValue<f64>,
    pub max: UiValue<f64>,
    pub value: UiValue<f64>,
    pub step: Option<UiValue<f64>>,
    pub shift_step: Option<UiValue<f64>>,
    pub on_change: Option<UiHandlerId>,
    pub on_release: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiPickListNode {
    pub options: UiValue<UiRuntimeValue>,
    pub selected: UiValue<Option<UiRuntimeValue>>,
    pub placeholder: Option<UiValue<StringId>>,
    pub text_size: Option<UiValue<f32>>,
    pub padding: UiValue<UiPadding>,
    pub menu_height: Option<UiValue<f32>>,
    pub on_select: Option<UiHandlerId>,
    pub on_open: Option<UiHandlerId>,
    pub on_close: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiComboBoxNode {
    pub options: UiValue<UiRuntimeValue>,
    pub value: UiValue<StringId>,
    pub placeholder: UiValue<StringId>,
    pub selected: UiValue<Option<UiRuntimeValue>>,
    pub menu_height: Option<UiValue<f32>>,
    pub on_input: Option<UiHandlerId>,
    pub on_select: Option<UiHandlerId>,
    pub on_option_hovered: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiProgressBarNode {
    pub min: UiValue<f64>,
    pub max: UiValue<f64>,
    pub value: UiValue<f64>,
    pub orientation: UiOrientation,
}

#[derive(Clone, Debug)]
pub struct UiScrollableNode {
    pub direction: UiScrollDirection,
    pub horizontal: Option<UiScrollbarProperties>,
    pub vertical: Option<UiScrollbarProperties>,
    pub spacing: Option<UiValue<f32>>,
    pub anchor_x: UiScrollAnchor,
    pub anchor_y: UiScrollAnchor,
    pub auto_scroll: bool,
    pub on_scroll: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiScrollbarProperties {
    pub visible: UiValue<bool>,
    pub width: UiValue<f32>,
    pub margin: UiValue<f32>,
    pub scroller_width: UiValue<f32>,
    pub spacing: Option<UiValue<f32>>,
    pub anchor: UiScrollAnchor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiScrollDirection {
    Vertical,
    Horizontal,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiScrollAnchor {
    Start,
    End,
}

#[derive(Clone, Debug)]
pub struct UiPaneGridNode {
    pub panes: Vec<UiPane>,
    pub layout: UiPaneLayout,
    pub spacing: UiValue<f32>,
    pub resize_leeway: UiValue<f32>,
    pub on_click: Option<UiHandlerId>,
    pub on_drag: Option<UiHandlerId>,
    pub on_resize: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiPane {
    pub id: UiPaneId,
    pub content: UiNodeId,
    pub title_bar: Option<UiNodeId>,
    pub pinned: bool,
}

#[derive(Clone, Debug)]
pub enum UiPaneLayout {
    Pane(UiPaneId),
    Split {
        axis: UiAxis,
        ratio: f32,
        first: Box<UiPaneLayout>,
        second: Box<UiPaneLayout>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug)]
pub struct UiTableNode {
    pub columns: Vec<UiTableColumn>,
    pub rows: UiValue<UiRuntimeValue>,
    pub row_local: UiLocalId,
    pub index_local: UiLocalId,
    pub row_template: UiNodeId,
    pub header: Option<UiNodeId>,
    pub footer: Option<UiNodeId>,
}

#[derive(Clone, Debug)]
pub struct UiTableColumn {
    pub name: Option<StringId>,
    pub width: UiLength,
    pub min_width: Option<f32>,
    pub max_width: Option<f32>,
    pub align: UiHorizontal,
    pub padding: UiPadding,
}

#[derive(Clone, Debug)]
pub struct UiTooltipNode {
    pub tooltip: UiNodeId,
    pub position: UiTooltipPosition,
    pub gap: UiValue<f32>,
    pub padding: UiValue<f32>,
    pub delay_ms: UiValue<u64>,
    pub snap_within_viewport: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTooltipPosition {
    Top,
    Bottom,
    Left,
    Right,
    FollowCursor,
}

#[derive(Clone, Debug, Default)]
pub struct UiMouseAreaNode {
    pub on_press: Option<UiHandlerId>,
    pub on_release: Option<UiHandlerId>,
    pub on_double_click: Option<UiHandlerId>,
    pub on_right_press: Option<UiHandlerId>,
    pub on_right_release: Option<UiHandlerId>,
    pub on_middle_press: Option<UiHandlerId>,
    pub on_middle_release: Option<UiHandlerId>,
    pub on_enter: Option<UiHandlerId>,
    pub on_exit: Option<UiHandlerId>,
    pub on_move: Option<UiHandlerId>,
    pub on_scroll: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiHoverNode {
    pub overlay: UiNodeId,
}

#[derive(Clone, Debug)]
pub struct UiFloatNode {
    pub scale: UiValue<f32>,
    pub translation: UiFloatTranslation,
}

#[derive(Clone, Debug)]
pub enum UiFloatTranslation {
    None,
    Static(UiValue<UiVector>),
    Computed(FuncId),
}

#[derive(Clone, Debug, Default)]
pub struct UiPinNode {
    pub x: Option<UiValue<f32>>,
    pub y: Option<UiValue<f32>>,
}

#[derive(Clone, Debug)]
pub struct UiResponsiveNode {
    pub cases: Vec<UiResponsiveCase>,
    pub fallback: UiNodeId,
}

#[derive(Clone, Debug)]
pub struct UiResponsiveCase {
    pub condition: UiMediaCondition,
    pub root: UiNodeId,
}

#[derive(Clone, Debug)]
pub struct UiSensorNode {
    pub key: Option<UiValue<UiKey>>,
    pub on_enter: Option<UiHandlerId>,
    pub on_exit: Option<UiHandlerId>,
}

#[derive(Clone, Debug)]
pub struct UiThemerNode {
    pub theme: UiValue<UiThemeId>,
}

#[derive(Clone, Debug)]
pub struct UiLazyNode {
    pub dependency: UiBindingId,
    pub subtree: UiNodeId,
}

#[derive(Clone, Debug)]
pub struct UiImageNode {
    pub source: UiValue<UiResourceId>,
    pub content_fit: UiContentFit,
    pub filter: UiImageFilter,
    pub rotation: UiValue<UiRotation>,
    pub opacity: UiValue<f32>,
    pub scale: UiValue<f32>,
    pub border_radius: UiValue<UiRadius>,
    pub snap: bool,
}

#[derive(Clone, Debug)]
pub struct UiSvgNode {
    pub source: UiValue<UiResourceId>,
    pub content_fit: UiContentFit,
    pub rotation: UiValue<UiRotation>,
    pub opacity: UiValue<f32>,
}

#[derive(Clone, Debug)]
pub struct UiCanvasNode {
    pub program: UiCanvasProgramId,
}

#[derive(Clone, Debug)]
pub struct UiShaderNode {
    pub program: UiShaderProgramId,
    pub uniforms: Vec<UiShaderUniformBinding>,
}

#[derive(Clone, Debug)]
pub struct UiShaderUniformBinding {
    pub name: StringId,
    pub value: UiBindingId,
}

#[derive(Clone, Debug)]
pub struct UiQrCodeNode {
    pub data: UiValue<StringId>,
    pub cell_size: Option<UiValue<f32>>,
    pub quiet_zone: Option<UiValue<u16>>,
    pub error_correction: UiQrErrorCorrection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiQrErrorCorrection {
    Low,
    Medium,
    Quartile,
    High,
}

#[derive(Clone, Debug)]
pub struct UiComponentInstanceNode {
    pub component: UiComponentId,
    pub props: Vec<(UiPropertyId, UiBindingId)>,
    pub slots: Vec<(UiSlotId, UiNodeId)>,
}

#[derive(Clone, Debug)]
pub struct UiSlotNode {
    pub slot: UiSlotId,
}

#[derive(Clone, Debug)]
pub struct UiIfNode {
    pub condition: UiBindingId,
    pub then_node: UiNodeId,
    pub else_node: Option<UiNodeId>,
}

#[derive(Clone, Debug)]
pub struct UiSwitchNode {
    pub scrutinee: UiBindingId,
    pub cases: Vec<UiSwitchCase>,
    pub fallback: Option<UiNodeId>,
}

#[derive(Clone, Debug)]
pub struct UiSwitchCase {
    pub pattern: UiRuntimeValue,
    pub node: UiNodeId,
}

#[derive(Clone, Debug)]
pub struct UiForEachNode {
    pub collection: UiBindingId,
    pub item_local: UiLocalId,
    pub index_local: UiLocalId,
    pub key_function: Option<FuncId>,
    pub template: UiNodeId,
    pub empty: Option<UiNodeId>,
}

#[derive(Clone, Debug)]
pub struct UiPortalNode {
    pub target: UiPortalTarget,
    pub content: UiNodeId,
    pub z_index: i32,
}

#[derive(Clone, Debug)]
pub enum UiPortalTarget {
    Window(UiWindowId),
    OverlayRoot,
    Named(StringId),
}

#[derive(Clone, Debug)]
pub struct UiCustomWidgetNode {
    pub backend_name: StringId,
    pub payload: Vec<u8>,
    pub stateful: bool,
}

// =============================================================================
// Layout, geometry and typography
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiLayout {
    pub width: UiValue<UiLength>,
    pub height: UiValue<UiLength>,
    pub min_width: Option<UiValue<f32>>,
    pub min_height: Option<UiValue<f32>>,
    pub max_width: Option<UiValue<f32>>,
    pub max_height: Option<UiValue<f32>>,
    pub padding: UiValue<UiPadding>,
    pub margin: UiValue<UiPadding>,
    pub align_self_x: Option<UiHorizontal>,
    pub align_self_y: Option<UiVertical>,
    pub aspect_ratio: Option<UiValue<f32>>,
    pub z_index: i32,
}

impl Default for UiLayout {
    fn default() -> Self {
        Self {
            width: UiLength::Shrink.into(),
            height: UiLength::Shrink.into(),
            min_width: None,
            min_height: None,
            max_width: None,
            max_height: None,
            padding: UiPadding::ZERO.into(),
            margin: UiPadding::ZERO.into(),
            align_self_x: None,
            align_self_y: None,
            aspect_ratio: None,
            z_index: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiLength {
    Fill,
    FillPortion(u16),
    Shrink,
    Pixels(f32),
    Percent(f32),
    MinContent,
    MaxContent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiPoint {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiVector {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiSize {
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiPadding {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl UiPadding {
    pub const ZERO: Self = Self {
        top: 0.0,
        right: 0.0,
        bottom: 0.0,
        left: 0.0,
    };

    pub const fn all(value: f32) -> Self {
        Self { top: value, right: value, bottom: value, left: value }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiRadius {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_right: f32,
    pub bottom_left: f32,
}

impl UiRadius {
    pub const fn all(value: f32) -> Self {
        Self {
            top_left: value,
            top_right: value,
            bottom_right: value,
            bottom_left: value,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum UiHorizontal {
    #[default]
    Start,
    Center,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum UiVertical {
    #[default]
    Top,
    Center,
    Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiContentFit {
    Contain,
    Cover,
    Fill,
    None,
    ScaleDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiImageFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiRotation {
    FloatingRadians(f32),
    SolidRadians(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiTransform {
    pub translate: UiVector,
    pub scale: UiVector,
    pub rotation_radians: f32,
    pub origin: UiPoint,
}

impl Default for UiTransform {
    fn default() -> Self {
        Self {
            translate: UiVector::default(),
            scale: UiVector { x: 1.0, y: 1.0 },
            rotation_radians: 0.0,
            origin: UiPoint { x: 0.5, y: 0.5 },
        }
    }
}

#[derive(Clone, Debug)]
pub struct UiTextFormat {
    pub width: UiValue<UiLength>,
    pub height: UiValue<UiLength>,
    pub size: Option<UiValue<f32>>,
    pub font: Option<UiValue<UiFont>>,
    pub line_height: UiLineHeight,
    pub align_x: UiTextAlignment,
    pub align_y: UiVertical,
    pub shaping: UiTextShaping,
    pub wrapping: UiTextWrapping,
}

impl Default for UiTextFormat {
    fn default() -> Self {
        Self {
            width: UiLength::Shrink.into(),
            height: UiLength::Shrink.into(),
            size: None,
            font: None,
            line_height: UiLineHeight::Relative(1.2),
            align_x: UiTextAlignment::Default,
            align_y: UiVertical::Top,
            shaping: UiTextShaping::Basic,
            wrapping: UiTextWrapping::Word,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiLineHeight {
    Relative(f32),
    Absolute(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTextAlignment {
    Default,
    Left,
    Center,
    Right,
    Justified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTextShaping {
    Basic,
    Advanced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTextWrapping {
    None,
    Glyph,
    Word,
    WordOrGlyph,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiFont {
    Default,
    Monospace,
    Family(StringId),
    Resource(UiResourceId),
    Detailed(UiFontDescriptor),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UiFontDescriptor {
    pub family: StringId,
    pub weight: UiFontWeight,
    pub stretch: UiFontStretch,
    pub style: UiFontStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiFontWeight {
    Thin,
    ExtraLight,
    Light,
    Normal,
    Medium,
    Semibold,
    Bold,
    ExtraBold,
    Black,
    Numeric(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiFontStretch {
    UltraCondensed,
    ExtraCondensed,
    Condensed,
    SemiCondensed,
    Normal,
    SemiExpanded,
    Expanded,
    ExtraExpanded,
    UltraExpanded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiFontStyle {
    Normal,
    Italic,
    Oblique,
}

// =============================================================================
// Events and accessibility
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiEventKind {
    Press,
    Release,
    DoubleClick,
    RightPress,
    RightRelease,
    MiddlePress,
    MiddleRelease,
    PointerEnter,
    PointerLeave,
    PointerMove,
    Wheel,
    TouchStart,
    TouchMove,
    TouchEnd,
    KeyPressed,
    KeyReleased,
    ModifiersChanged,
    InputMethodOpened,
    InputMethodPreedit,
    InputMethodCommit,
    InputMethodClosed,
    Focus,
    Blur,
    Input,
    Submit,
    Paste,
    Toggle,
    Select,
    Open,
    Close,
    Slide,
    SlideRelease,
    Scroll,
    Link,
    Edit,
    PaneClick,
    PaneDrag,
    PaneResize,
    SensorEnter,
    SensorExit,
    WindowOpened,
    WindowClosed,
    WindowMoved,
    WindowResized,
    WindowRescaled,
    WindowRedrawRequested,
    WindowCloseRequested,
    WindowFocused,
    WindowUnfocused,
    FileHovered,
    FileDropped,
    FilesHoveredLeft,
    Custom(StringId),
}

/// Normalized event payload passed to WIR event handlers.
///
/// `UiEventKind` identifies the subscription/handler edge while this value carries
/// backend-neutral runtime data. Backends should intern transient strings into the
/// current Napalm string table before invoking the handler.
#[derive(Clone, Debug, PartialEq)]
pub enum UiEventPayload {
    None,
    Pointer(UiPointerEvent),
    Wheel(UiWheelEvent),
    Touch(UiTouchEvent),
    Keyboard(UiKeyboardEvent),
    Modifiers(UiModifiers),
    InputMethod(UiInputMethodEvent),
    Focused(bool),
    Text(StringId),
    Bool(bool),
    Number(f64),
    Selection(UiRuntimeValue),
    Scroll(UiScrollEvent),
    Pane(UiPaneEvent),
    Window(UiWindowEventData),
    File(UiFileEvent),
    Sensor(UiSensorEvent),
    Custom(UiRuntimeValue),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiPointerEvent {
    pub position: Option<UiPoint>,
    pub button: Option<UiMouseButton>,
    pub modifiers: UiModifiers,
    pub click_count: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiMouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiWheelEvent {
    pub position: Option<UiPoint>,
    pub delta: UiScrollDelta,
    pub modifiers: UiModifiers,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiScrollDelta {
    Lines(UiVector),
    Pixels(UiVector),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiTouchEvent {
    pub id: u64,
    pub position: UiPoint,
    pub phase: UiTouchPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiTouchPhase {
    Started,
    Moved,
    Ended,
    Lost,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiKeyboardEvent {
    pub key: UiKeyboardKey,
    pub modified_key: UiKeyboardKey,
    pub physical_key: UiPhysicalKey,
    pub location: UiKeyLocation,
    pub modifiers: UiModifiers,
    pub text: Option<StringId>,
    pub repeat: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiKeyboardKey {
    Character(StringId),
    Named(StringId),
    Unidentified,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiPhysicalKey {
    Code(StringId),
    Unidentified(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiKeyLocation {
    Standard,
    Left,
    Right,
    Numpad,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UiModifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub logo: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiInputMethodEvent {
    Opened,
    Preedit {
        text: StringId,
        selection: Option<(u32, u32)>,
    },
    Commit(StringId),
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiScrollEvent {
    pub absolute_offset: UiPoint,
    pub relative_offset: UiPoint,
    pub viewport: UiRect,
    pub content_bounds: UiRect,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiPaneEvent {
    Clicked(UiPaneId),
    Dragged {
        pane: UiPaneId,
        from: UiPoint,
        to: UiPoint,
    },
    Resized {
        split: u32,
        ratio: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiWindowEventData {
    Opened {
        position: Option<UiPoint>,
        size: UiSize,
    },
    Closed,
    Moved(UiPoint),
    Resized(UiSize),
    Rescaled(f64),
    RedrawRequested,
    CloseRequested,
    Focused(bool),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UiFileEvent {
    Hovered(StringId),
    Dropped(StringId),
    HoverLeft,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiSensorEvent {
    pub visible_fraction: f32,
    pub entered: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiCursor {
    None,
    Hidden,
    Default,
    ContextMenu,
    Help,
    Pointer,
    Progress,
    Wait,
    Cell,
    Crosshair,
    Text,
    Alias,
    Copy,
    Move,
    NoDrop,
    NotAllowed,
    Grab,
    Grabbing,
    AllScroll,
    ResizeColumn,
    ResizeRow,
    ResizeHorizontal,
    ResizeVertical,
    ResizeNorthEastSouthWest,
    ResizeNorthWestSouthEast,
    ZoomIn,
    ZoomOut,
}

#[derive(Clone, Debug)]
pub struct UiAccessibility {
    pub role: UiA11yRole,
    pub name: Option<UiValue<StringId>>,
    pub description: Option<UiValue<StringId>>,
    pub value: Option<UiValue<StringId>>,
    pub placeholder: Option<UiValue<StringId>>,
    pub checked: Option<UiValue<bool>>,
    pub selected: Option<UiValue<bool>>,
    pub expanded: Option<UiValue<bool>>,
    pub required: bool,
    pub invalid: bool,
    pub hidden: bool,
    pub live: UiA11yLive,
    pub heading_level: Option<u8>,
    pub actions: Vec<UiA11yAction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiA11yRole {
    Application,
    Window,
    Group,
    GenericContainer,
    Button,
    Checkbox,
    Radio,
    Switch,
    Slider,
    ProgressIndicator,
    Text,
    Heading,
    Link,
    Image,
    TextInput,
    TextArea,
    List,
    ListItem,
    Menu,
    MenuItem,
    ComboBox,
    Table,
    Row,
    Cell,
    Tab,
    TabList,
    TabPanel,
    Dialog,
    Tooltip,
    ScrollArea,
    Separator,
    Canvas,
    Alert,
    AlertDialog,
    Article,
    Banner,
    Blockquote,
    Caption,
    Code,
    Complementary,
    ContentInfo,
    Definition,
    Deletion,
    Emphasis,
    Feed,
    Figure,
    Form,
    Grid,
    GridCell,
    Insertion,
    ListBox,
    Log,
    Marquee,
    Math,
    Meter,
    Navigation,
    Note,
    Option,
    Paragraph,
    Search,
    Status,
    Strong,
    Subscript,
    Superscript,
    Term,
    Timer,
    Toolbar,
    Tree,
    TreeItem,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiA11yLive {
    Off,
    Polite,
    Assertive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiA11yAction {
    Focus,
    Blur,
    Click,
    Increment,
    Decrement,
    SetValue,
    ShowMenu,
    ScrollIntoView,
}

// =============================================================================
// Resources
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiResource {
    pub id: UiResourceId,
    pub name: Option<StringId>,
    pub kind: UiResourceKind,
    pub source: UiResourceSource,
    pub cache: UiResourceCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiResourceKind {
    Image,
    Svg,
    Font,
    Markdown,
    Shader,
    Binary,
}

#[derive(Clone, Debug)]
pub enum UiResourceSource {
    Path(StringId),
    Url(StringId),
    Embedded(Vec<u8>),
    Generated(FuncId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiResourceCache {
    None,
    Weak,
    Strong,
    Process,
}

// =============================================================================
// Themes and style cascade
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiTheme {
    pub id: UiThemeId,
    pub name: StringId,
    pub mode: UiThemeMode,
    pub preset: Option<UiThemePreset>,
    pub palette: UiPalette,
    pub extended: Option<UiExtendedPalette>,
    pub style_sheets: Vec<UiStyleSheetId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiThemeMode {
    Light,
    Dark,
    System,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiThemePreset {
    Light,
    Dark,
    Dracula,
    Nord,
    SolarizedLight,
    SolarizedDark,
    GruvboxLight,
    GruvboxDark,
    CatppuccinLatte,
    CatppuccinFrappe,
    CatppuccinMacchiato,
    CatppuccinMocha,
    TokyoNight,
    TokyoNightStorm,
    TokyoNightLight,
    KanagawaWave,
    KanagawaDragon,
    KanagawaLotus,
    Moonfly,
    Nightfly,
    Oxocarbon,
    Ferra,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiPalette {
    pub background: UiColor,
    pub text: UiColor,
    pub primary: UiColor,
    pub success: UiColor,
    pub warning: UiColor,
    pub danger: UiColor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiExtendedPalette {
    pub background: UiPaletteScale,
    pub primary: UiPaletteScale,
    pub secondary: UiPaletteScale,
    pub success: UiPaletteScale,
    pub warning: UiPaletteScale,
    pub danger: UiPaletteScale,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiPaletteScale {
    pub weakest: UiColorPair,
    pub weak: UiColorPair,
    pub base: UiColorPair,
    pub strong: UiColorPair,
    pub strongest: UiColorPair,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiColorPair {
    pub color: UiColor,
    pub text: UiColor,
}

#[derive(Clone, Debug)]
pub struct UiStyleToken {
    pub id: UiStyleTokenId,
    pub name: StringId,
    pub ty: UiValueType,
    pub value: UiStyleLiteral,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiStyleLiteral {
    Bool(bool),
    Number(f64),
    String(StringId),
    Color(UiColorRef),
    Background(UiBackground),
    Length(UiLength),
    Padding(UiPadding),
    Border(UiBorder),
    Shadow(UiShadow),
    Radius(UiRadius),
    Font(UiFont),
}

#[derive(Clone, Debug)]
pub struct UiStyleSheet {
    pub id: UiStyleSheetId,
    pub name: Option<StringId>,
    pub rules: Vec<UiStyleRule>,
}

#[derive(Clone, Debug)]
pub struct UiStyleRule {
    pub id: UiStyleRuleId,
    pub selector: UiSelector,
    pub media: Option<UiMediaCondition>,
    pub declarations: UiStylePatch,
    pub transitions: Vec<UiTransition>,
    pub origin: UiStyleOrigin,
    pub layer: u16,
    pub important: bool,
    pub specificity: u32,
    pub order: u32,
    pub source: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiStyleOrigin {
    UserAgent,
    Theme,
    Component,
    Author,
    Inline,
}

#[derive(Clone, Debug)]
pub struct UiSelector {
    pub chain: Vec<UiSelectorPart>,
}

#[derive(Clone, Debug)]
pub struct UiSelectorPart {
    pub combinator: Option<UiSelectorCombinator>,
    pub atom: UiSelectorAtom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiSelectorCombinator {
    Descendant,
    Child,
    AdjacentSibling,
    GeneralSibling,
}

#[derive(Clone, Debug, Default)]
pub struct UiSelectorAtom {
    pub widget: Option<UiWidgetTag>,
    pub name: Option<StringId>,
    pub classes: Vec<StringId>,
    pub states: Vec<UiPseudoClass>,
    pub part: Option<UiStylePart>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiStylePart {
    Root,
    Label,
    Icon,
    Content,
    Placeholder,
    Selection,
    Rail,
    RailFilled,
    RailUnfilled,
    Handle,
    ScrollbarHorizontal,
    ScrollbarVertical,
    Scroller,
    Menu,
    MenuItem,
    MenuItemSelected,
    PaneSplit,
    PaneTitleBar,
    TableHorizontalSeparator,
    TableVerticalSeparator,
    InlineCode,
    CodeBlock,
    Link,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiPseudoClass {
    Active,
    Hover,
    Pressed,
    Disabled,
    Focus,
    FocusHover,
    Checked,
    Unchecked,
    Selected,
    Unselected,
    Dragged,
    Open,
    Closed,
    Invalid,
    PlaceholderShown,
    FirstChild,
    LastChild,
    Odd,
    Even,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiWidgetTag {
    Any,
    Container,
    Row,
    Column,
    Grid,
    Stack,
    Space,
    Rule,
    Text,
    RichText,
    Markdown,
    Button,
    Checkbox,
    Radio,
    Toggler,
    TextInput,
    TextEditor,
    Slider,
    VerticalSlider,
    PickList,
    ComboBox,
    ProgressBar,
    Scrollable,
    PaneGrid,
    Table,
    Tooltip,
    MouseArea,
    Hover,
    Float,
    Pin,
    Responsive,
    Sensor,
    Themer,
    Opaque,
    Lazy,
    Image,
    Svg,
    Canvas,
    Shader,
    QrCode,
    Component,
    Custom,
}

#[derive(Clone, Debug, Default)]
pub struct UiNodeStyle {
    pub inline: UiStylePatch,
}

#[derive(Clone, Debug, Default)]
pub struct UiStylePatch {
    pub common: UiCommonStylePatch,
    pub text: UiTextStylePatch,
    pub widget: Option<UiWidgetStylePatch>,
}

#[derive(Clone, Debug, Default)]
pub struct UiCommonStylePatch {
    pub background: Option<UiOptionalStyleValue<UiBackground>>,
    pub foreground: Option<UiValue<UiColorRef>>,
    pub border: Option<UiValue<UiBorder>>,
    pub shadow: Option<UiValue<UiShadow>>,
    pub radius: Option<UiValue<UiRadius>>,
    pub opacity: Option<UiValue<f32>>,
    pub cursor: Option<UiValue<UiCursor>>,
    pub snap: Option<UiValue<bool>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTextStylePatch {
    pub color: Option<UiOptionalStyleValue<UiColorRef>>,
    pub font: Option<UiValue<UiFont>>,
    pub size: Option<UiValue<f32>>,
    pub weight: Option<UiValue<UiFontWeight>>,
    pub line_height: Option<UiValue<UiLineHeight>>,
    pub letter_spacing: Option<UiValue<f32>>,
    pub underline: Option<UiValue<bool>>,
    pub strikethrough: Option<UiValue<bool>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiOptionalStyleValue<T> {
    None,
    Value(UiValue<T>),
}

#[derive(Clone, Debug)]
pub enum UiWidgetStylePatch {
    Application(UiApplicationStylePatch),
    Container(UiContainerStylePatch),
    Float(UiFloatStylePatch),
    Button(UiButtonStylePatch),
    Checkbox(UiCheckboxStylePatch),
    Radio(UiRadioStylePatch),
    Toggler(UiTogglerStylePatch),
    Text(UiTextWidgetStylePatch),
    TextInput(UiTextInputStylePatch),
    TextEditor(UiTextEditorStylePatch),
    PickList(UiPickListStylePatch),
    Menu(UiMenuStylePatch),
    Slider(UiSliderStylePatch),
    Scrollable(UiScrollableStylePatch),
    ProgressBar(UiProgressBarStylePatch),
    Rule(UiRuleStylePatch),
    Svg(UiSvgStylePatch),
    PaneGrid(UiPaneGridStylePatch),
    Table(UiTableStylePatch),
    QrCode(UiQrCodeStylePatch),
    Markdown(UiMarkdownStylePatch),
}

#[derive(Clone, Debug, Default)]
pub struct UiApplicationStylePatch {
    pub background_color: Option<UiValue<UiColorRef>>,
    pub text_color: Option<UiValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiContainerStylePatch {
    pub text_color: Option<UiOptionalStyleValue<UiColorRef>>,
    pub background: Option<UiOptionalStyleValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
    pub shadow: Option<UiValue<UiShadow>>,
    pub snap: Option<UiValue<bool>>,
}


#[derive(Clone, Debug, Default)]
pub struct UiFloatStylePatch {
    pub shadow: Option<UiValue<UiShadow>>,
    pub shadow_border_radius: Option<UiValue<UiRadius>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiButtonStylePatch {
    pub background: Option<UiOptionalStyleValue<UiBackground>>,
    pub text_color: Option<UiValue<UiColorRef>>,
    pub border: Option<UiValue<UiBorder>>,
    pub shadow: Option<UiValue<UiShadow>>,
    pub snap: Option<UiValue<bool>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiCheckboxStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub icon_color: Option<UiValue<UiColorRef>>,
    pub border: Option<UiValue<UiBorder>>,
    pub text_color: Option<UiOptionalStyleValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiRadioStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub dot_color: Option<UiValue<UiColorRef>>,
    pub border_width: Option<UiValue<f32>>,
    pub border_color: Option<UiValue<UiColorRef>>,
    pub text_color: Option<UiOptionalStyleValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTogglerStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub background_border_width: Option<UiValue<f32>>,
    pub background_border_color: Option<UiValue<UiColorRef>>,
    pub foreground: Option<UiValue<UiBackground>>,
    pub foreground_border_width: Option<UiValue<f32>>,
    pub foreground_border_color: Option<UiValue<UiColorRef>>,
    pub text_color: Option<UiOptionalStyleValue<UiColorRef>>,
    pub border_radius: Option<UiOptionalStyleValue<UiRadius>>,
    pub padding_ratio: Option<UiValue<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTextWidgetStylePatch {
    pub color: Option<UiOptionalStyleValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTextInputStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
    pub icon: Option<UiValue<UiColorRef>>,
    pub placeholder: Option<UiValue<UiColorRef>>,
    pub value: Option<UiValue<UiColorRef>>,
    pub selection: Option<UiValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTextEditorStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
    pub placeholder: Option<UiValue<UiColorRef>>,
    pub value: Option<UiValue<UiColorRef>>,
    pub selection: Option<UiValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiPickListStylePatch {
    pub text_color: Option<UiValue<UiColorRef>>,
    pub placeholder_color: Option<UiValue<UiColorRef>>,
    pub handle_color: Option<UiValue<UiColorRef>>,
    pub background: Option<UiValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiMenuStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
    pub text_color: Option<UiValue<UiColorRef>>,
    pub selected_text_color: Option<UiValue<UiColorRef>>,
    pub selected_background: Option<UiValue<UiBackground>>,
    pub shadow: Option<UiValue<UiShadow>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiSliderStylePatch {
    pub rail: Option<UiSliderRailStyle>,
    pub handle: Option<UiSliderHandleStyle>,
}

#[derive(Clone, Debug)]
pub struct UiSliderRailStyle {
    pub filled: UiValue<UiBackground>,
    pub unfilled: UiValue<UiBackground>,
    pub width: UiValue<f32>,
    pub border: UiValue<UiBorder>,
}

#[derive(Clone, Debug)]
pub struct UiSliderHandleStyle {
    pub shape: UiValue<UiSliderHandleShape>,
    pub background: UiValue<UiBackground>,
    pub border_width: UiValue<f32>,
    pub border_color: UiValue<UiColorRef>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiSliderHandleShape {
    Circle { radius: f32 },
    Rectangle { width: f32, radius: UiRadius },
}

#[derive(Clone, Debug, Default)]
pub struct UiScrollableStylePatch {
    pub container: Option<UiContainerStylePatch>,
    pub vertical_rail: Option<UiScrollRailStyle>,
    pub horizontal_rail: Option<UiScrollRailStyle>,
    pub gap: Option<UiOptionalStyleValue<UiBackground>>,
    pub auto_scroll: Option<UiAutoScrollStyle>,
}

#[derive(Clone, Debug)]
pub struct UiScrollRailStyle {
    pub background: Option<UiValue<UiBackground>>,
    pub border: UiValue<UiBorder>,
    pub scroller: UiScrollerStyle,
}

#[derive(Clone, Debug)]
pub struct UiScrollerStyle {
    pub background: UiValue<UiBackground>,
    pub border: UiValue<UiBorder>,
}

#[derive(Clone, Debug)]
pub struct UiAutoScrollStyle {
    pub background: UiValue<UiBackground>,
    pub border: UiValue<UiBorder>,
    pub shadow: UiValue<UiShadow>,
}

#[derive(Clone, Debug, Default)]
pub struct UiProgressBarStylePatch {
    pub background: Option<UiValue<UiBackground>>,
    pub bar: Option<UiValue<UiBackground>>,
    pub border: Option<UiValue<UiBorder>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiRuleStylePatch {
    pub color: Option<UiValue<UiColorRef>>,
    pub radius: Option<UiValue<UiRadius>>,
    pub fill_mode: Option<UiValue<UiRuleFillMode>>,
    pub snap: Option<UiValue<bool>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiRuleFillMode {
    Full,
    Percent(f32),
    Padded(f32),
    AsymmetricPadding { start: f32, end: f32 },
}

#[derive(Clone, Debug, Default)]
pub struct UiSvgStylePatch {
    pub color: Option<UiOptionalStyleValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiPaneGridStylePatch {
    pub hovered_region: Option<UiPaneRegionStyle>,
    pub picked_split: Option<UiPaneLineStyle>,
    pub hovered_split: Option<UiPaneLineStyle>,
}

#[derive(Clone, Debug)]
pub struct UiPaneRegionStyle {
    pub background: UiValue<UiBackground>,
    pub border: UiValue<UiBorder>,
}

#[derive(Clone, Debug)]
pub struct UiPaneLineStyle {
    pub color: UiValue<UiColorRef>,
    pub width: UiValue<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct UiTableStylePatch {
    pub separator_x: Option<UiValue<UiBackground>>,
    pub separator_y: Option<UiValue<UiBackground>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiQrCodeStylePatch {
    pub cell: Option<UiValue<UiColorRef>>,
    pub background: Option<UiValue<UiColorRef>>,
}

#[derive(Clone, Debug, Default)]
pub struct UiMarkdownStylePatch {
    pub font: Option<UiValue<UiFont>>,
    pub inline_code_background: Option<UiValue<UiBackground>>,
    pub inline_code_padding: Option<UiValue<UiPadding>>,
    pub inline_code_color: Option<UiValue<UiColorRef>>,
    pub inline_code_font: Option<UiValue<UiFont>>,
    pub code_block_font: Option<UiValue<UiFont>>,
    pub link_color: Option<UiValue<UiColorRef>>,
}

#[derive(Clone, Debug)]
pub struct UiTransition {
    pub property: UiAnimatableProperty,
    pub duration_ms: u32,
    pub delay_ms: u32,
    pub easing: UiEasing,
}

#[derive(Clone, Debug)]
pub enum UiMediaCondition {
    True,
    Not(Box<UiMediaCondition>),
    All(Vec<UiMediaCondition>),
    Any(Vec<UiMediaCondition>),
    MinWidth(f32),
    MaxWidth(f32),
    MinHeight(f32),
    MaxHeight(f32),
    MinScaleFactor(f32),
    MaxScaleFactor(f32),
    Theme(UiThemeMode),
    ReducedMotion(bool),
    HighContrast(bool),
    Platform(UiPlatform),
    Pointer(UiPointerKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiPlatform {
    Windows,
    MacOS,
    Linux,
    Web,
    Android,
    Ios,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiPointerKind {
    Mouse,
    Touch,
    Pen,
    None,
}

// =============================================================================
// Paint values and colors
// =============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UiColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiThemeColor {
    Background,
    Text,
    Primary,
    Success,
    Warning,
    Danger,
    BackgroundWeakest,
    BackgroundWeak,
    BackgroundStrong,
    BackgroundStrongest,
    PrimaryWeak,
    PrimaryStrong,
    SecondaryWeak,
    Secondary,
    SecondaryStrong,
    SuccessWeak,
    SuccessStrong,
    WarningWeak,
    WarningStrong,
    DangerWeak,
    DangerStrong,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiColorRef {
    Literal(UiColor),
    Theme(UiThemeColor),
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiBackground {
    Color(UiColorRef),
    LinearGradient(UiLinearGradient),
    RadialGradient(UiRadialGradient),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiLinearGradient {
    pub angle_radians: f32,
    pub stops: Vec<UiGradientStop>,
    pub interpolation: UiColorInterpolation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UiRadialGradient {
    pub center: UiPoint,
    pub radius: f32,
    pub stops: Vec<UiGradientStop>,
    pub interpolation: UiColorInterpolation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiGradientStop {
    pub offset: f32,
    pub color: UiColorRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiColorInterpolation {
    Srgb,
    LinearSrgb,
    Oklab,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiBorder {
    pub color: UiColorRef,
    pub width: f32,
    pub radius: UiRadius,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UiShadow {
    pub color: UiColorRef,
    pub offset: UiVector,
    pub blur_radius: f32,
}

// =============================================================================
// Animation IR
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiAnimation {
    pub id: UiAnimationId,
    pub name: StringId,
    pub trigger: UiAnimationTrigger,
    pub tracks: Vec<UiAnimationTrack>,
    pub duration_ms: u32,
    pub delay_ms: u32,
    pub iterations: UiAnimationIterations,
    pub direction: UiAnimationDirection,
    pub fill: UiAnimationFill,
    pub easing: UiEasing,
}

#[derive(Clone, Debug)]
pub enum UiAnimationTrigger {
    Manual,
    Mount,
    Unmount,
    State(UiPseudoClass),
    Binding(UiBindingId),
    Event(UiEventKind),
}

#[derive(Clone, Debug)]
pub struct UiAnimationTrack {
    pub target: UiAnimationTarget,
    pub property: UiAnimatableProperty,
    pub keyframes: Vec<UiKeyframe>,
}

#[derive(Clone, Debug)]
pub enum UiAnimationTarget {
    Node(UiNodeId),
    Selector(UiSelector),
}

#[derive(Clone, Debug)]
pub struct UiKeyframe {
    pub offset: f32,
    pub value: UiAnimatedValue,
    pub easing: Option<UiEasing>,
}

#[derive(Clone, Debug)]
pub enum UiAnimatedValue {
    Number(f32),
    Color(UiColor),
    Radius(UiRadius),
    Padding(UiPadding),
    Vector(UiVector),
    Transform(UiTransform),
    Shadow(UiShadow),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiAnimatableProperty {
    Opacity,
    Translate,
    Scale,
    Rotation,
    Width,
    Height,
    Padding,
    BackgroundColor,
    ForegroundColor,
    BorderColor,
    BorderWidth,
    BorderRadius,
    Shadow,
    TextColor,
    FontSize,
    SliderValue,
    ScrollX,
    ScrollY,
    ImageScale,
    ImageOpacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiAnimationIterations {
    Once,
    Count(u32),
    Infinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiAnimationDirection {
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiAnimationFill {
    None,
    Forwards,
    Backwards,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UiEasing {
    Linear,
    Ease,
    EaseIn,
    EaseOut,
    EaseInOut,
    CubicBezier(f32, f32, f32, f32),
    Steps { count: u16, jump_at_start: bool },
    Spring { mass: f32, stiffness: f32, damping: f32, initial_velocity: f32 },
}

// =============================================================================
// Canvas display-list IR
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiCanvasProgram {
    pub id: UiCanvasProgramId,
    pub commands: Vec<UiCanvasCommand>,
    pub dynamic_builder: Option<FuncId>,
}

#[derive(Clone, Debug)]
pub enum UiCanvasCommand {
    Save,
    Restore,
    Transform(UiTransform),
    Translate(UiVector),
    Rotate(f32),
    Scale(UiVector),
    Clip(UiCanvasPath),
    FillPath {
        path: UiCanvasPath,
        fill: UiBackground,
        rule: UiFillRule,
    },
    StrokePath {
        path: UiCanvasPath,
        stroke: UiCanvasStroke,
    },
    Text(UiCanvasText),
    Image(UiCanvasImage),
    Svg(UiCanvasSvg),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiCanvasPath {
    pub segments: Vec<UiPathSegment>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiPathSegment {
    MoveTo(UiPoint),
    LineTo(UiPoint),
    QuadraticTo { control: UiPoint, to: UiPoint },
    CubicTo { control1: UiPoint, control2: UiPoint, to: UiPoint },
    Arc {
        center: UiPoint,
        radius: f32,
        start_radians: f32,
        end_radians: f32,
        clockwise: bool,
    },
    Rectangle(UiRect),
    RoundedRectangle { rect: UiRect, radius: UiRadius },
    Circle { center: UiPoint, radius: f32 },
    Ellipse { center: UiPoint, radii: UiVector, rotation_radians: f32 },
    Close,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiFillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug)]
pub struct UiCanvasStroke {
    pub color: UiColorRef,
    pub width: f32,
    pub line_cap: UiLineCap,
    pub line_join: UiLineJoin,
    pub dash: Vec<f32>,
    pub dash_offset: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiLineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiLineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Debug)]
pub struct UiCanvasText {
    pub content: StringId,
    pub position: UiPoint,
    pub max_width: f32,
    pub color: UiColorRef,
    pub size: f32,
    pub line_height: UiLineHeight,
    pub font: UiFont,
    pub align_x: UiTextAlignment,
    pub align_y: UiVertical,
    pub shaping: UiTextShaping,
}

#[derive(Clone, Debug)]
pub struct UiCanvasImage {
    pub resource: UiResourceId,
    pub bounds: UiRect,
    pub filter: UiImageFilter,
    pub rotation_radians: f32,
    pub radius: UiRadius,
    pub opacity: f32,
    pub snap: bool,
}

#[derive(Clone, Debug)]
pub struct UiCanvasSvg {
    pub resource: UiResourceId,
    pub bounds: UiRect,
    pub color: Option<UiColorRef>,
}

// =============================================================================
// Shader IR boundary
// =============================================================================

#[derive(Clone, Debug)]
pub struct UiShaderProgram {
    pub id: UiShaderProgramId,
    pub source: UiResourceId,
    pub vertex_entry: Option<StringId>,
    pub fragment_entry: Option<StringId>,
    pub primitive_builder: Option<FuncId>,
    pub uniforms: Vec<UiShaderUniform>,
}

#[derive(Clone, Debug)]
pub struct UiShaderUniform {
    pub name: StringId,
    pub ty: UiShaderType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiShaderType {
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    I32,
    U32,
    Bool,
    Texture2D,
    Sampler,
}

// =============================================================================
// Opaque runtime values crossing WIR <-> UI IR
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum UiRuntimeValue {
    Null,
    Bool(bool),
    Number(f64),
    String(StringId),
    Opaque(u64),
}

// =============================================================================
// Validation
// =============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UiVerifyError {
    pub component: Option<UiComponentId>,
    pub node: Option<UiNodeId>,
    pub message: String,
}

impl UiProgram {
    pub fn verify(&self) -> Result<(), Vec<UiVerifyError>> {
        let mut errors = Vec::new();

        for (index, component) in self.components.iter().enumerate() {
            if component.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: Some(component.id),
                    node: None,
                    message: format!(
                        "component arena index {index} does not match {:?}",
                        component.id
                    ),
                });
            }
            verify_component(self, component, &mut errors);
        }

        for (index, theme) in self.themes.iter().enumerate() {
            if theme.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("theme arena index {index} does not match {:?}", theme.id),
                });
            }
        }

        for (index, sheet) in self.style_sheets.iter().enumerate() {
            if sheet.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!(
                        "style-sheet arena index {index} does not match {:?}",
                        sheet.id
                    ),
                });
            }
        }

        for (index, resource) in self.resources.iter().enumerate() {
            if resource.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!(
                        "resource arena index {index} does not match {:?}",
                        resource.id
                    ),
                });
            }
        }

        for (index, token) in self.style_tokens.iter().enumerate() {
            if token.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("style-token arena index {index} does not match {:?}", token.id),
                });
            }
        }

        for (index, animation) in self.animations.iter().enumerate() {
            if animation.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("animation arena index {index} does not match {:?}", animation.id),
                });
            }
        }

        for (index, canvas) in self.canvas_programs.iter().enumerate() {
            if canvas.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("canvas-program arena index {index} does not match {:?}", canvas.id),
                });
            }
        }

        for (index, shader) in self.shader_programs.iter().enumerate() {
            if shader.id.0 as usize != index {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("shader-program arena index {index} does not match {:?}", shader.id),
                });
            }
            if self.resource(shader.source).is_none() {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("shader program {:?} references invalid source {:?}", shader.id, shader.source),
                });
            }
        }

        for &font in &self.settings.preload_fonts {
            if self.resource(font).is_none() {
                errors.push(UiVerifyError {
                    component: None,
                    node: None,
                    message: format!("preloaded font references invalid resource {font:?}"),
                });
            }
        }

        for theme in &self.themes {
            for &sheet in &theme.style_sheets {
                if self.style_sheets.get(sheet.0 as usize).is_none() {
                    errors.push(UiVerifyError {
                        component: None,
                        node: None,
                        message: format!("theme {:?} references invalid style sheet {sheet:?}", theme.id),
                    });
                }
            }
        }

        for sheet in &self.style_sheets {
            for (index, rule) in sheet.rules.iter().enumerate() {
                if rule.id.0 as usize != index {
                    errors.push(UiVerifyError {
                        component: None,
                        node: None,
                        message: format!(
                            "style rule arena index {index} in sheet {:?} does not match {:?}",
                            sheet.id, rule.id
                        ),
                    });
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    #[inline]
    pub fn component(&self, id: UiComponentId) -> Option<&UiComponent> {
        self.components.get(id.0 as usize).filter(|component| component.id == id)
    }

    #[inline]
    pub fn theme(&self, id: UiThemeId) -> Option<&UiTheme> {
        self.themes.get(id.0 as usize).filter(|theme| theme.id == id)
    }

    #[inline]
    pub fn resource(&self, id: UiResourceId) -> Option<&UiResource> {
        self.resources.get(id.0 as usize).filter(|resource| resource.id == id)
    }
}

impl UiComponent {
    #[inline]
    pub fn node(&self, id: UiNodeId) -> Option<&UiNode> {
        self.nodes.get(id.0 as usize).filter(|node| node.id == id)
    }

    #[inline]
    pub fn binding(&self, id: UiBindingId) -> Option<&UiBinding> {
        self.bindings.get(id.0 as usize).filter(|binding| binding.id == id)
    }

    #[inline]
    pub fn handler(&self, id: UiHandlerId) -> Option<&UiHandler> {
        self.handlers.get(id.0 as usize).filter(|handler| handler.id == id)
    }
}

fn verify_component(program: &UiProgram, component: &UiComponent, errors: &mut Vec<UiVerifyError>) {
    let fail = |errors: &mut Vec<UiVerifyError>, node, message: String| {
        errors.push(UiVerifyError {
            component: Some(component.id),
            node,
            message,
        });
    };

    if component.kind == UiComponentKind::Application && component.windows.is_empty() {
        fail(errors, None, "application component has no windows".to_owned());
    }

    if component.kind == UiComponentKind::Component && component.embedded_root.is_none() {
        fail(errors, None, "reusable component has no embedded root".to_owned());
    }

    if let Some(root) = component.embedded_root {
        if component.node(root).is_none() {
            fail(errors, Some(root), format!("invalid embedded root {root:?}"));
        }
    }

    for (index, property) in component.properties.iter().enumerate() {
        if property.id.0 as usize != index {
            fail(errors, None, format!("property arena index {index} does not match {:?}", property.id));
        }
        if let Some(default) = property.default {
            if component.binding(default).is_none() {
                fail(errors, None, format!("property {:?} has invalid default binding {default:?}", property.id));
            }
        }
    }

    for (index, state) in component.states.iter().enumerate() {
        if state.id.0 as usize != index {
            fail(errors, None, format!("state arena index {index} does not match {:?}", state.id));
        }
        if component.binding(state.initial).is_none() {
            fail(errors, None, format!("state {:?} has invalid initial binding {:?}", state.id, state.initial));
        }
    }

    for (index, computed) in component.computed.iter().enumerate() {
        if computed.id.0 as usize != index {
            fail(errors, None, format!("computed arena index {index} does not match {:?}", computed.id));
        }
    }

    for (index, subscription) in component.subscriptions.iter().enumerate() {
        if subscription.id.0 as usize != index {
            fail(errors, None, format!("subscription arena index {index} does not match {:?}", subscription.id));
        }
    }

    for (index, window) in component.windows.iter().enumerate() {
        if window.id.0 as usize != index {
            fail(errors, None, format!("window arena index {index} does not match {:?}", window.id));
        }
    }

    for (index, slot) in component.slots.iter().enumerate() {
        if slot.id.0 as usize != index {
            fail(errors, None, format!("slot arena index {index} does not match {:?}", slot.id));
        }
        if let Some(fallback) = slot.fallback {
            if component.node(fallback).is_none() {
                fail(errors, Some(fallback), format!("slot {:?} has invalid fallback node {fallback:?}", slot.id));
            }
        }
    }

    for (index, node) in component.nodes.iter().enumerate() {
        if node.id.0 as usize != index {
            fail(
                errors,
                Some(node.id),
                format!("node arena index {index} does not match {:?}", node.id),
            );
        }

        for &child in &node.children {
            if component.node(child).is_none() {
                fail(errors, Some(node.id), format!("invalid child {child:?}"));
            }
        }

        for &handler in &node.handlers {
            if component.handler(handler).is_none() {
                fail(errors, Some(node.id), format!("invalid handler {handler:?}"));
            }
        }

        verify_node_kind(program, component, node, errors);
    }

    for window in &component.windows {
        if component.node(window.root).is_none() {
            fail(errors, Some(window.root), format!("window {:?} has invalid root", window.id));
        }
        for &handler in &window.handlers {
            if component.handler(handler).is_none() {
                fail(errors, None, format!("window {:?} has invalid handler {handler:?}", window.id));
            }
        }
    }

    for (index, binding) in component.bindings.iter().enumerate() {
        if binding.id.0 as usize != index {
            fail(
                errors,
                None,
                format!("binding arena index {index} does not match {:?}", binding.id),
            );
        }
        verify_dependencies(component, &binding.dependencies, errors);
    }

    for computed in &component.computed {
        verify_dependencies(component, &computed.dependencies, errors);
    }

    for (index, handler) in component.handlers.iter().enumerate() {
        if handler.id.0 as usize != index {
            fail(
                errors,
                None,
                format!("handler arena index {index} does not match {:?}", handler.id),
            );
        }
    }

    for &sheet in &component.style_sheets {
        if program.style_sheets.get(sheet.0 as usize).is_none() {
            fail(errors, None, format!("invalid style sheet {sheet:?}"));
        }
    }
}

fn verify_dependencies(
    component: &UiComponent,
    dependencies: &[UiDependency],
    errors: &mut Vec<UiVerifyError>,
) {
    for dependency in dependencies {
        let valid = match *dependency {
            UiDependency::Property(id) => component.properties.get(id.0 as usize).is_some_and(|item| item.id == id),
            UiDependency::State(id) => component.states.get(id.0 as usize).is_some_and(|item| item.id == id),
            UiDependency::Computed(id) => component.computed.get(id.0 as usize).is_some_and(|item| item.id == id),
            UiDependency::WindowSize(id) | UiDependency::WindowScaleFactor(id) => {
                component.windows.get(id.0 as usize).is_some_and(|item| item.id == id)
            }
            UiDependency::Local(_) | UiDependency::StyleToken(_) | UiDependency::Theme
            | UiDependency::Environment(_) | UiDependency::Resource(_) => true,
        };
        if !valid {
            errors.push(UiVerifyError {
                component: Some(component.id),
                node: None,
                message: format!("invalid binding dependency {dependency:?}"),
            });
        }
    }
}

fn push_node_ref_error(
    component: &UiComponent,
    owner: UiNodeId,
    referenced: UiNodeId,
    role: &str,
    errors: &mut Vec<UiVerifyError>,
) {
    if component.node(referenced).is_none() {
        errors.push(UiVerifyError {
            component: Some(component.id),
            node: Some(owner),
            message: format!("{role} references invalid node {referenced:?}"),
        });
    }
}

fn push_binding_ref_error(
    component: &UiComponent,
    owner: UiNodeId,
    referenced: UiBindingId,
    role: &str,
    errors: &mut Vec<UiVerifyError>,
) {
    if component.binding(referenced).is_none() {
        errors.push(UiVerifyError {
            component: Some(component.id),
            node: Some(owner),
            message: format!("{role} references invalid binding {referenced:?}"),
        });
    }
}

fn push_handler_ref_error(
    component: &UiComponent,
    owner: UiNodeId,
    referenced: Option<UiHandlerId>,
    role: &str,
    errors: &mut Vec<UiVerifyError>,
) {
    if let Some(handler) = referenced {
        if component.handler(handler).is_none() {
            errors.push(UiVerifyError {
                component: Some(component.id),
                node: Some(owner),
                message: format!("{role} references invalid handler {handler:?}"),
            });
        }
    }
}

fn verify_node_handlers(
    component: &UiComponent,
    node: &UiNode,
    errors: &mut Vec<UiVerifyError>,
) {
    let check = |handler, role, errors: &mut Vec<UiVerifyError>| {
        push_handler_ref_error(component, node.id, handler, role, errors);
    };

    match &node.kind {
        UiNodeKind::RichText(widget) => check(widget.on_link, "rich-text link", errors),
        UiNodeKind::Markdown(widget) => check(widget.on_link, "markdown link", errors),
        UiNodeKind::Button(widget) => check(widget.on_press, "button press", errors),
        UiNodeKind::Checkbox(widget) => check(widget.on_toggle, "checkbox toggle", errors),
        UiNodeKind::Radio(widget) => check(Some(widget.on_select), "radio select", errors),
        UiNodeKind::Toggler(widget) => check(widget.on_toggle, "toggler toggle", errors),
        UiNodeKind::TextInput(widget) => {
            check(widget.on_input, "text-input input", errors);
            check(widget.on_submit, "text-input submit", errors);
            check(widget.on_paste, "text-input paste", errors);
        }
        UiNodeKind::TextEditor(widget) => check(widget.on_action, "text-editor action", errors),
        UiNodeKind::Slider(widget) | UiNodeKind::VerticalSlider(widget) => {
            check(widget.on_change, "slider change", errors);
            check(widget.on_release, "slider release", errors);
        }
        UiNodeKind::PickList(widget) => {
            check(widget.on_select, "pick-list select", errors);
            check(widget.on_open, "pick-list open", errors);
            check(widget.on_close, "pick-list close", errors);
        }
        UiNodeKind::ComboBox(widget) => {
            check(widget.on_input, "combo-box input", errors);
            check(widget.on_select, "combo-box select", errors);
            check(widget.on_option_hovered, "combo-box hover", errors);
        }
        UiNodeKind::Scrollable(widget) => check(widget.on_scroll, "scrollable scroll", errors),
        UiNodeKind::PaneGrid(widget) => {
            check(widget.on_click, "pane-grid click", errors);
            check(widget.on_drag, "pane-grid drag", errors);
            check(widget.on_resize, "pane-grid resize", errors);
        }
        UiNodeKind::MouseArea(widget) => {
            check(widget.on_press, "mouse-area press", errors);
            check(widget.on_release, "mouse-area release", errors);
            check(widget.on_double_click, "mouse-area double-click", errors);
            check(widget.on_right_press, "mouse-area right-press", errors);
            check(widget.on_right_release, "mouse-area right-release", errors);
            check(widget.on_middle_press, "mouse-area middle-press", errors);
            check(widget.on_middle_release, "mouse-area middle-release", errors);
            check(widget.on_enter, "mouse-area enter", errors);
            check(widget.on_exit, "mouse-area exit", errors);
            check(widget.on_move, "mouse-area move", errors);
            check(widget.on_scroll, "mouse-area scroll", errors);
        }
        UiNodeKind::Sensor(widget) => {
            check(widget.on_enter, "sensor enter", errors);
            check(widget.on_exit, "sensor exit", errors);
        }
        _ => {}
    }
}

fn pane_layout_references(layout: &UiPaneLayout, id: UiPaneId) -> bool {
    match layout {
        UiPaneLayout::Pane(candidate) => *candidate == id,
        UiPaneLayout::Split { first, second, .. } => {
            pane_layout_references(first, id) || pane_layout_references(second, id)
        }
    }
}

fn verify_node_kind(
    program: &UiProgram,
    component: &UiComponent,
    node: &UiNode,
    errors: &mut Vec<UiVerifyError>,
) {
    verify_node_handlers(component, node, errors);

    match &node.kind {
        UiNodeKind::Canvas(canvas) => {
            if program.canvas_programs.get(canvas.program.0 as usize).is_none() {
                errors.push(UiVerifyError {
                    component: Some(component.id),
                    node: Some(node.id),
                    message: format!("invalid canvas program {:?}", canvas.program),
                });
            }
        }
        UiNodeKind::Shader(shader) => {
            if program.shader_programs.get(shader.program.0 as usize).is_none() {
                errors.push(UiVerifyError {
                    component: Some(component.id),
                    node: Some(node.id),
                    message: format!("invalid shader program {:?}", shader.program),
                });
            }
            for uniform in &shader.uniforms {
                push_binding_ref_error(component, node.id, uniform.value, "shader uniform", errors);
            }
        }
        UiNodeKind::Image(image) => verify_resource_value(program, component, node, &image.source, errors),
        UiNodeKind::Svg(svg) => verify_resource_value(program, component, node, &svg.source, errors),
        UiNodeKind::Component(instance) => {
            let target = program.component(instance.component);
            if target.is_none() {
                errors.push(UiVerifyError {
                    component: Some(component.id),
                    node: Some(node.id),
                    message: format!("invalid component {:?}", instance.component),
                });
            }
            for &(property, binding) in &instance.props {
                push_binding_ref_error(component, node.id, binding, "component prop", errors);
                if let Some(target) = target {
                    if target.properties.get(property.0 as usize).is_none_or(|item| item.id != property) {
                        errors.push(UiVerifyError {
                            component: Some(component.id),
                            node: Some(node.id),
                            message: format!("invalid target component property {property:?}"),
                        });
                    }
                }
            }
            for &(slot, slot_node) in &instance.slots {
                push_node_ref_error(component, node.id, slot_node, "component slot", errors);
                if let Some(target) = target {
                    if target.slots.get(slot.0 as usize).is_none_or(|item| item.id != slot) {
                        errors.push(UiVerifyError {
                            component: Some(component.id),
                            node: Some(node.id),
                            message: format!("invalid target component slot {slot:?}"),
                        });
                    }
                }
            }
        }
        UiNodeKind::If(branch) => {
            push_binding_ref_error(component, node.id, branch.condition, "if condition", errors);
            push_node_ref_error(component, node.id, branch.then_node, "if then", errors);
            if let Some(else_node) = branch.else_node {
                push_node_ref_error(component, node.id, else_node, "if else", errors);
            }
        }
        UiNodeKind::Switch(switch) => {
            push_binding_ref_error(component, node.id, switch.scrutinee, "switch scrutinee", errors);
            for case in &switch.cases {
                push_node_ref_error(component, node.id, case.node, "switch case", errors);
            }
            if let Some(fallback) = switch.fallback {
                push_node_ref_error(component, node.id, fallback, "switch fallback", errors);
            }
        }
        UiNodeKind::ForEach(loop_) => {
            push_binding_ref_error(component, node.id, loop_.collection, "for-each collection", errors);
            push_node_ref_error(component, node.id, loop_.template, "for-each template", errors);
            if let Some(empty) = loop_.empty {
                push_node_ref_error(component, node.id, empty, "for-each empty", errors);
            }
        }
        UiNodeKind::Responsive(responsive) => {
            for case in &responsive.cases {
                push_node_ref_error(component, node.id, case.root, "responsive case", errors);
            }
            push_node_ref_error(component, node.id, responsive.fallback, "responsive fallback", errors);
        }
        UiNodeKind::Tooltip(tooltip) => {
            push_node_ref_error(component, node.id, tooltip.tooltip, "tooltip", errors)
        }
        UiNodeKind::Hover(hover) => {
            push_node_ref_error(component, node.id, hover.overlay, "hover overlay", errors)
        }
        UiNodeKind::Lazy(lazy) => {
            push_binding_ref_error(component, node.id, lazy.dependency, "lazy dependency", errors);
            push_node_ref_error(component, node.id, lazy.subtree, "lazy subtree", errors);
        }
        UiNodeKind::PaneGrid(pane_grid) => {
            for (index, pane) in pane_grid.panes.iter().enumerate() {
                if pane.id.0 as usize != index {
                    errors.push(UiVerifyError {
                        component: Some(component.id),
                        node: Some(node.id),
                        message: format!("pane arena index {index} does not match {:?}", pane.id),
                    });
                }
                push_node_ref_error(component, node.id, pane.content, "pane content", errors);
                if let Some(title_bar) = pane.title_bar {
                    push_node_ref_error(component, node.id, title_bar, "pane title bar", errors);
                }
                if !pane_layout_references(&pane_grid.layout, pane.id) {
                    errors.push(UiVerifyError {
                        component: Some(component.id),
                        node: Some(node.id),
                        message: format!("pane {:?} is not referenced by the pane layout", pane.id),
                    });
                }
            }
        }
        UiNodeKind::Slot(slot_node) => {
            if component.slots.get(slot_node.slot.0 as usize).is_none_or(|slot| slot.id != slot_node.slot) {
                errors.push(UiVerifyError {
                    component: Some(component.id),
                    node: Some(node.id),
                    message: format!("invalid component slot {:?}", slot_node.slot),
                });
            }
        }
        UiNodeKind::Themer(themer) => {
            if let UiValue::Static(theme) = &themer.theme {
                if program.theme(*theme).is_none() {
                    errors.push(UiVerifyError {
                        component: Some(component.id),
                        node: Some(node.id),
                        message: format!("invalid theme {theme:?}"),
                    });
                }
            }
        }
        UiNodeKind::Table(table) => {
            push_node_ref_error(component, node.id, table.row_template, "table row template", errors);
            if let Some(header) = table.header {
                push_node_ref_error(component, node.id, header, "table header", errors);
            }
            if let Some(footer) = table.footer {
                push_node_ref_error(component, node.id, footer, "table footer", errors);
            }
        }
        UiNodeKind::Portal(portal) => {
            push_node_ref_error(component, node.id, portal.content, "portal content", errors)
        }
        _ => {}
    }
}

fn verify_resource_value(
    program: &UiProgram,
    component: &UiComponent,
    node: &UiNode,
    value: &UiValue<UiResourceId>,
    errors: &mut Vec<UiVerifyError>,
) {
    if let UiValue::Static(resource) = value {
        if program.resource(*resource).is_none() {
            errors.push(UiVerifyError {
                component: Some(component.id),
                node: Some(node.id),
                message: format!("invalid resource {resource:?}"),
            });
        }
    }
}

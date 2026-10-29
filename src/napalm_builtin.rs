//! Napalm core runtime and engine-provided primitives.
//!
//! The frontend/lowerer reserve the standard core globals in every module. The
//! interpreter materializes those globals as builtin class objects. This module
//! owns the method/primitive manifest and the native implementations.

use crate::napalm_interpreter::{
    fast_random_seed, num_to_u32, BuiltinClass, Host, HostContext, Interpreter, NativeBuiltin, RuntimeError, Value,
};
use crate::napalm_ir::{Builder, BuiltinId, CmpKind, FuncId, FunctionKind, Inst, IntBinOp, MethodKind, MethodSymbol, PrimitiveId, Program, Type};

use std::io::{self, Write};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{
    DateTime, Datelike, Duration as ChronoDuration, Local, LocalResult, NaiveDate,
    NaiveDateTime, Offset, TimeZone, Timelike, Utc,
};

#[derive(Clone, Copy, Debug)]
pub enum BuiltinTarget {
    Instance(BuiltinClass),
    Static(BuiltinId),
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveBinding {
    pub target: BuiltinTarget,
    pub method: MethodSymbol,
    pub primitive: PrimitiveId,
}

#[derive(Clone, Debug, Default)]
pub struct BuiltinManifest {
    bindings: Vec<PrimitiveBinding>,
    native_bindings: Vec<NativeBinding>,
    primitives: Vec<Option<BuiltinPrimitive>>,
    list_remove: Option<(MethodSymbol, FuncId)>,
    string_add: Option<(MethodSymbol, FuncId)>,
    string_right_add: Option<FuncId>,
    system_print: Option<(MethodSymbol, FuncId)>,
}

#[derive(Clone, Copy, Debug)]
pub struct NativeBinding {
    pub target: BuiltinTarget,
    pub method: MethodSymbol,
    pub native: NativeBuiltin,
}

impl BuiltinManifest {
    pub fn bindings(&self) -> &[PrimitiveBinding] {
        &self.bindings
    }

    fn register(
        &mut self,
        program: &mut Program,
        target: BuiltinTarget,
        name: &str,
        kind: MethodKind,
        arity: u8,
        implementation: BuiltinPrimitive,
    ) -> MethodSymbol {
        let method = program.intern_method(name, kind, arity);
        let primitive = program.fresh_primitive();
        let index = primitive.0 as usize;
        if self.primitives.len() <= index {
            self.primitives.resize(index + 1, None);
        }
        self.primitives[index] = Some(implementation);
        self.bindings.push(PrimitiveBinding {
            target,
            method,
            primitive,
        });
        method
    }

    fn register_native(
        &mut self,
        program: &mut Program,
        target: BuiltinTarget,
        name: &str,
        kind: MethodKind,
        arity: u8,
        native: NativeBuiltin,
    ) -> MethodSymbol {
        let method = program.intern_method(name, kind, arity);
        self.native_bindings.push(NativeBinding { target, method, native });
        method
    }

    fn implementation(&self, primitive: PrimitiveId) -> Option<BuiltinPrimitive> {
        self.primitives
            .get(primitive.0 as usize)
            .copied()
            .flatten()
    }
}

#[derive(Clone, Copy, Debug)]
enum DateField {
    YearLegacy,
    FullYear,
    Month,
    Date,
    Day,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
}

#[derive(Clone, Copy, Debug)]
enum DateSetField {
    YearLegacy,
    FullYear,
    Month,
    Date,
    Hours,
    Minutes,
    Seconds,
    Milliseconds,
}

#[derive(Clone, Copy, Debug)]
enum DateStringFormat {
    String,
    UtcString,
    IsoString,
    DateString,
    TimeString,
    LocaleString,
    LocaleDateString,
    LocaleTimeString,
}

#[derive(Clone, Copy, Debug)]
enum BuiltinPrimitive {
    RandomNew,
    RandomInt,
    RangeCount,
    ListIsEmpty,
    MapIsEmpty,
    StringIsEmpty,
    MapKeysIsEmpty,
    ObjectEq,
    ObjectNe,
    ObjectToString,
    ObjectType,

    BoolNot,
    BoolToString,
    NullNot,
    NullToString,

    NumAdd,
    NumSub,
    NumMul,
    NumPow,
    NumDiv,
    NumMod,
    NumInclusiveRange,
    NumExclusiveRange,
    RangeNew,
    NumLt,
    NumLe,
    NumGt,
    NumGe,
    NumEq,
    NumNe,
    NumNeg,
    NumBitNot,
    NumBitAnd,
    NumBitOr,
    NumBitXor,
    NumShl,
    NumShr,
    NumAbs,
    NumFloor,
    NumCeil,
    NumRound,
    NumSqrt,
    NumSin,
    NumCos,
    NumTan,
    NumLog,
    NumExp,
    NumIsInteger,
    NumIsNan,
    NumIsInfinity,
    NumToString,
    NumFromString,
    NumPi,
    NumTau,
    NumInfinity,
    NumNan,

    StringCount,
    StringAdd,
    StringEq,
    StringNe,
    StringSubscript,
    StringContains,
    StringStartsWith,
    StringEndsWith,
    StringIndexOf,
    StringToString,
    StringIterate,
    StringIteratorValue,
    StringFromCodePoint,

    ListCount,
    ListConcat,
    ListAdd,
    ListInsert,
    ListRemoveAt,
    ListClear,
    ListSubscript,
    ListSubscriptSet,
    ListIterate,
    ListIteratorValue,
    ListToString,
    ListFilled,
    ListNew,
    ListNewNumber,

    MapCount,
    MapContainsKey,
    MapSubscript,
    MapSubscriptSet,
    MapRemove,
    MapClear,
    MapIterate,
    MapIteratorValue,
    MapKeys,
    MapKeysCount,
    MapKeysIterate,
    MapKeysIteratorValue,
    MapEntryKey,
    MapEntryValue,
    MapToString,

    RangeFrom,
    RangeTo,
    RangeMin,
    RangeMax,
    RangeInclusive,
    RangeIterate,
    RangeIteratorValue,
    RangeToString,

    SystemPrint,
    SystemWrite,
    SystemClock,

    DateNew,
    DateUtc,
    DateParse,
    DateNow,
    DateNowMs,
    DateSleep,
    DateGetTime,
    DateGet(DateField, bool),
    DateTimezoneOffset,
    DateSetTime,
    DateSet(DateSetField, bool),
    DateToString(DateStringFormat),
    DateToJson,
}

/// Reserves all method symbols and primitive IDs required by the core runtime.
///
/// Call this before lowering user code into the same `Program`; interning then
/// guarantees that user `Invoke` nodes refer to exactly the same MethodSymbols.
pub fn install_hir_builtins(program: &mut Program) -> BuiltinManifest {
    let mut manifest = BuiltinManifest::default();

    let instance = |class| BuiltinTarget::Instance(class);
    let static_ = |class| BuiltinTarget::Static(class);

    // Object protocol. Bind these to each concrete builtin receiver class below
    // instead of adding inheritance checks to the interpreter hot lookup path.
    for class in [
        BuiltinClass::Null,
        BuiltinClass::Bool,
        BuiltinClass::Num,
        BuiltinClass::String,
        BuiltinClass::List,
        BuiltinClass::Map,
        BuiltinClass::Range,
        BuiltinClass::Closure,
        BuiltinClass::Fiber,
        BuiltinClass::Promise,
        BuiltinClass::MapEntry,
        BuiltinClass::MapKeys,
        BuiltinClass::Random,
        BuiltinClass::FfiLibrary,
        BuiltinClass::FfiFunction,
        BuiltinClass::FfiPointer,
        BuiltinClass::Response,
        BuiltinClass::Headers,
        BuiltinClass::Module,
        BuiltinClass::Date,
    ] {
        manifest.register(program, instance(class), "==", MethodKind::Method, 1, BuiltinPrimitive::ObjectEq);
        manifest.register(program, instance(class), "!=", MethodKind::Method, 1, BuiltinPrimitive::ObjectNe);
        manifest.register(program, instance(class), "toString", MethodKind::Getter, 0, BuiltinPrimitive::ObjectToString);
        manifest.register(program, instance(class), "toString", MethodKind::Method, 0, BuiltinPrimitive::ObjectToString);
    }

    // Null and Bool.
    manifest.register(program, instance(BuiltinClass::Null), "!", MethodKind::Method, 0, BuiltinPrimitive::NullNot);
    manifest.register(program, instance(BuiltinClass::Null), "toString", MethodKind::Getter, 0, BuiltinPrimitive::NullToString);
    manifest.register(program, instance(BuiltinClass::Null), "toString", MethodKind::Method, 0, BuiltinPrimitive::NullToString);
    manifest.register(program, instance(BuiltinClass::Bool), "!", MethodKind::Method, 0, BuiltinPrimitive::BoolNot);
    manifest.register(program, instance(BuiltinClass::Bool), "toString", MethodKind::Getter, 0, BuiltinPrimitive::BoolToString);
    manifest.register(program, instance(BuiltinClass::Bool), "toString", MethodKind::Method, 0, BuiltinPrimitive::BoolToString);

    // Num instance methods and operators.
    let n = instance(BuiltinClass::Num);
    for (name, op) in [
        ("+", BuiltinPrimitive::NumAdd), ("-", BuiltinPrimitive::NumSub),
        ("*", BuiltinPrimitive::NumMul), ("**", BuiltinPrimitive::NumPow),
        ("/", BuiltinPrimitive::NumDiv),
        ("%", BuiltinPrimitive::NumMod), ("<", BuiltinPrimitive::NumLt),
        ("<=", BuiltinPrimitive::NumLe), (">", BuiltinPrimitive::NumGt),
        (">=", BuiltinPrimitive::NumGe), ("==", BuiltinPrimitive::NumEq),
        ("!=", BuiltinPrimitive::NumNe), ("&", BuiltinPrimitive::NumBitAnd),
        ("|", BuiltinPrimitive::NumBitOr), ("^", BuiltinPrimitive::NumBitXor),
        ("<<", BuiltinPrimitive::NumShl), (">>", BuiltinPrimitive::NumShr),
        ("..", BuiltinPrimitive::NumInclusiveRange),
        ("...", BuiltinPrimitive::NumExclusiveRange),
    ] {
        manifest.register(program, n, name, MethodKind::Method, 1, op);
    }
    manifest.register(program, n, "-", MethodKind::Method, 0, BuiltinPrimitive::NumNeg);
    manifest.register(program, n, "~", MethodKind::Method, 0, BuiltinPrimitive::NumBitNot);
    for (name, op) in [
        ("abs", BuiltinPrimitive::NumAbs), ("floor", BuiltinPrimitive::NumFloor),
        ("ceil", BuiltinPrimitive::NumCeil), ("round", BuiltinPrimitive::NumRound),
        ("sqrt", BuiltinPrimitive::NumSqrt), ("sin", BuiltinPrimitive::NumSin),
        ("cos", BuiltinPrimitive::NumCos), ("tan", BuiltinPrimitive::NumTan),
        ("log", BuiltinPrimitive::NumLog), ("exp", BuiltinPrimitive::NumExp),
        ("isInteger", BuiltinPrimitive::NumIsInteger), ("isNan", BuiltinPrimitive::NumIsNan),
        ("isInfinity", BuiltinPrimitive::NumIsInfinity), ("toString", BuiltinPrimitive::NumToString),
    ] {
        manifest.register(program, n, name, MethodKind::Getter, 0, op);
    }
    manifest.register(program, n, "toString", MethodKind::Method, 0, BuiltinPrimitive::NumToString);
    let ns = static_(BuiltinId::Num);
    manifest.register(program, ns, "fromString", MethodKind::Method, 1, BuiltinPrimitive::NumFromString);
    manifest.register(program, ns, "pi", MethodKind::Getter, 0, BuiltinPrimitive::NumPi);
    manifest.register(program, ns, "tau", MethodKind::Getter, 0, BuiltinPrimitive::NumTau);
    manifest.register(program, ns, "infinity", MethodKind::Getter, 0, BuiltinPrimitive::NumInfinity);
    manifest.register(program, ns, "nan", MethodKind::Getter, 0, BuiltinPrimitive::NumNan);

    // String.
    let st = instance(BuiltinClass::String);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::StringCount),
        ("length", MethodKind::Getter, 0, BuiltinPrimitive::StringCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::StringIsEmpty),
        ("==", MethodKind::Method, 1, BuiltinPrimitive::StringEq),
        ("!=", MethodKind::Method, 1, BuiltinPrimitive::StringNe),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::StringSubscript),
        ("contains", MethodKind::Method, 1, BuiltinPrimitive::StringContains),
        ("startsWith", MethodKind::Method, 1, BuiltinPrimitive::StringStartsWith),
        ("endsWith", MethodKind::Method, 1, BuiltinPrimitive::StringEndsWith),
        ("indexOf", MethodKind::Method, 1, BuiltinPrimitive::StringIndexOf),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::StringToString),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::StringIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::StringIteratorValue),
    ] {
        manifest.register(program, st, name, kind, arity, op);
    }
    manifest.register(program, st, "toString", MethodKind::Method, 0, BuiltinPrimitive::StringToString);
    manifest.register(program, static_(BuiltinId::String), "fromCodePoint", MethodKind::Method, 1, BuiltinPrimitive::StringFromCodePoint);

    // Keep String.+ inside Napalm/WIR instead of the host primitive ABI. This lets
    // the RHS use the ordinary virtual `toString()` protocol, including methods
    // implemented by user classes, while still exposing StringConcat to the JIT.
    let string_add_method = program.intern_method("+", MethodKind::Method, 1);
    let to_string_method = program.intern_method("toString", MethodKind::Method, 0);
    let string_module = program.add_module("<core:string>", 0);
    let mut string_add = Builder::new(
        program,
        string_module,
        "String.+",
        FunctionKind::Method,
        vec![Type::String, Type::Top],
        Type::String,
    );
    let left = string_add.param(0);
    let right = string_add.param(1);
    let right = string_add.invoke_typed(right, to_string_method, Vec::new(), Type::String);
    let result = string_add.emit1(Inst::StringConcat { left, right }, Type::String);
    string_add.ret(result);
    manifest.string_add = Some((string_add_method, string_add.finish()));

    // Dynamic `anything + String` needs the same rule even when static analysis
    // could not see the String. The interpreter redirects such Invoke sites to
    // this hidden helper before executing a cached receiver-specific `+` target.
    let mut string_right_add = Builder::new(
        program,
        string_module,
        "<string-right-add>",
        FunctionKind::Method,
        vec![Type::Top, Type::String],
        Type::String,
    );
    let left = string_right_add.param(0);
    let right = string_right_add.param(1);
    let left = string_right_add.invoke_typed(left, to_string_method, Vec::new(), Type::String);
    let result = string_right_add.emit1(Inst::StringConcat { left, right }, Type::String);
    string_right_add.ret(result);
    manifest.string_right_add = Some(string_right_add.finish());

    // List.
    let list = instance(BuiltinClass::List);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::ListCount),
        ("length", MethodKind::Getter, 0, BuiltinPrimitive::ListCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::ListIsEmpty),
        ("+", MethodKind::Method, 1, BuiltinPrimitive::ListConcat),
        ("add", MethodKind::Method, 1, BuiltinPrimitive::ListAdd),
        ("insert", MethodKind::Method, 2, BuiltinPrimitive::ListInsert),
        ("removeAt", MethodKind::Method, 1, BuiltinPrimitive::ListRemoveAt),
        ("clear", MethodKind::Method, 0, BuiltinPrimitive::ListClear),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::ListSubscript),
        ("[]=", MethodKind::SubscriptSetter, 2, BuiltinPrimitive::ListSubscriptSet),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::ListIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::ListIteratorValue),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::ListToString),
    ] {
        manifest.register(program, list, name, kind, arity, op);
    }
    manifest.register(program, list, "toString", MethodKind::Method, 0, BuiltinPrimitive::ListToString);
    manifest.register(program, static_(BuiltinId::List), "filled", MethodKind::Method, 2, BuiltinPrimitive::ListFilled);
    manifest.register(program, static_(BuiltinId::List), "new", MethodKind::Method, 1, BuiltinPrimitive::ListNew);
    // Compiler-only target for `new Array<Number>(n)`. Keeping it on List avoids
    // introducing a second runtime class identity for Array.
    manifest.register(program, static_(BuiltinId::List), "__newNumberArray", MethodKind::Method, 1, BuiltinPrimitive::ListNewNumber);
    let remove = program.intern_method("remove", MethodKind::Method, 1);
    let equals = program.intern_method("==", MethodKind::Method, 1);
    let module = program.add_module("<core:list>", 0);
    let mut b = Builder::new(program, module, "List.remove", FunctionKind::Method,
        vec![Type::List(None), Type::Top], Type::Top);
    let list = b.param(0);
    let value = b.param(1);
    let header = b.new_block(&[]);
    let index = b.add_block_param(header, Type::Int32);
    let body = b.new_block(&[]);
    let found = b.new_block(&[]);
    let advance = b.new_block(&[]);
    let missing = b.new_block(&[]);
    let zero = b.const_i32(0);
    b.jump(header, vec![zero]);
    b.position_at(header);
    let len = b.emit1(Inst::ListLen { list }, Type::Int32);
    let more = b.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: index, right: len }, Type::Bool);
    b.branch(more, body, vec![], missing, vec![]);
    b.position_at(body);
    let item = b.emit1(Inst::ListGet { list, index }, Type::Top);
    // Dispatch == normally: user-defined equality can execute Napalm and trigger GC.
    let equal = b.invoke(item, equals, vec![value]);
    b.branch(equal, found, vec![], advance, vec![]);
    b.position_at(found);
    let removed = b.emit1(Inst::ListRemoveAt { list, index }, Type::Top);
    b.ret(removed);
    b.position_at(advance);
    let one = b.const_i32(1);
    let next = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: index, right: one }, Type::Int32);
    b.jump(header, vec![next]);
    b.position_at(missing);
    let null = b.const_null();
    b.ret(null);
    manifest.list_remove = Some((remove, b.finish()));

    // Map.
    let map = instance(BuiltinClass::Map);
    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::MapCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::MapIsEmpty),
        ("containsKey", MethodKind::Method, 1, BuiltinPrimitive::MapContainsKey),
        ("[]", MethodKind::SubscriptGetter, 1, BuiltinPrimitive::MapSubscript),
        ("[]=", MethodKind::SubscriptSetter, 2, BuiltinPrimitive::MapSubscriptSet),
        ("remove", MethodKind::Method, 1, BuiltinPrimitive::MapRemove),
        ("clear", MethodKind::Method, 0, BuiltinPrimitive::MapClear),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::MapIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::MapIteratorValue),
        ("keys", MethodKind::Getter, 0, BuiltinPrimitive::MapKeys),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::MapToString),
    ] {
        manifest.register(program, map, name, kind, arity, op);
    }

    // Each additional subscript descends into a nested List or Map. Setters
    // reserve one explicit argument for the value within the 16-argument ABI.
    for (target, getter, setter) in [
        (instance(BuiltinClass::List), BuiltinPrimitive::ListSubscript, BuiltinPrimitive::ListSubscriptSet),
        (map, BuiltinPrimitive::MapSubscript, BuiltinPrimitive::MapSubscriptSet),
    ] {
        for arity in 2..=crate::napalm_front::MAX_PARAMETERS as u8 {
            manifest.register(program, target, "[]", MethodKind::SubscriptGetter, arity, getter);
            if arity < crate::napalm_front::MAX_PARAMETERS as u8 {
                manifest.register(program, target, "[]=", MethodKind::SubscriptSetter, arity + 1, setter);
            }
        }
    }

    for (name, kind, arity, op) in [
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::MapKeysCount),
        ("isEmpty", MethodKind::Getter, 0, BuiltinPrimitive::MapKeysIsEmpty),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::MapKeysIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::MapKeysIteratorValue),
    ] {
        manifest.register(program, instance(BuiltinClass::MapKeys), name, kind, arity, op);
    }
    manifest.register(program, instance(BuiltinClass::MapEntry), "key", MethodKind::Getter, 0, BuiltinPrimitive::MapEntryKey);
    manifest.register(program, instance(BuiltinClass::MapEntry), "value", MethodKind::Getter, 0, BuiltinPrimitive::MapEntryValue);

    manifest.register(program, map, "toString", MethodKind::Method, 0, BuiltinPrimitive::MapToString);

    // Range.
    manifest.register(program, static_(BuiltinId::Range), "new", MethodKind::Method, 2, BuiltinPrimitive::RangeNew);
    manifest.register(program, static_(BuiltinId::Range), "new", MethodKind::Method, 3, BuiltinPrimitive::RangeNew);
    let range = instance(BuiltinClass::Range);
    for (name, kind, arity, op) in [
        ("from", MethodKind::Getter, 0, BuiltinPrimitive::RangeFrom),
        ("count", MethodKind::Getter, 0, BuiltinPrimitive::RangeCount),
        ("to", MethodKind::Getter, 0, BuiltinPrimitive::RangeTo),
        ("min", MethodKind::Getter, 0, BuiltinPrimitive::RangeMin),
        ("max", MethodKind::Getter, 0, BuiltinPrimitive::RangeMax),
        ("isInclusive", MethodKind::Getter, 0, BuiltinPrimitive::RangeInclusive),
        ("iterate", MethodKind::Method, 1, BuiltinPrimitive::RangeIterate),
        ("iteratorValue", MethodKind::Method, 1, BuiltinPrimitive::RangeIteratorValue),
        ("toString", MethodKind::Getter, 0, BuiltinPrimitive::RangeToString),
    ] {
        manifest.register(program, range, name, kind, arity, op);
    }

    manifest.register(program, range, "toString", MethodKind::Method, 0, BuiltinPrimitive::RangeToString);

    // Fn.new/arity/call(...) are VM-native because closures own bytecode frame metadata.
    manifest.register_native(program, static_(BuiltinId::Fn), "new", MethodKind::Method, 1, NativeBuiltin::FnNew);
    manifest.register_native(program, instance(BuiltinClass::Closure), "arity", MethodKind::Getter, 0, NativeBuiltin::FnArity);
    for arity in 0..=16u8 {
        manifest.register_native(
            program,
            instance(BuiltinClass::Closure),
            "call",
            MethodKind::Method,
            arity,
            NativeBuiltin::FnCall,
        );
    }

    // Fiber operations manipulate the VM scheduler/frame graph and therefore
    // deliberately bypass the Host primitive ABI.
    manifest.register_native(program, static_(BuiltinId::Fiber), "current", MethodKind::Getter, 0, NativeBuiltin::FiberCurrent);
    manifest.register_native(program, static_(BuiltinId::Fiber), "new", MethodKind::Method, 1, NativeBuiltin::FiberNew);
    manifest.register_native(program, instance(BuiltinClass::Fiber), "error", MethodKind::Getter, 0, NativeBuiltin::FiberError);
    manifest.register_native(program, instance(BuiltinClass::Fiber), "isDone", MethodKind::Getter, 0, NativeBuiltin::FiberIsDone);
    for arity in 0..=1u8 {
        manifest.register_native(program, instance(BuiltinClass::Fiber), "call", MethodKind::Method, arity, NativeBuiltin::FiberCall);
        manifest.register_native(program, instance(BuiltinClass::Fiber), "try", MethodKind::Method, arity, NativeBuiltin::FiberTry);
        manifest.register_native(program, instance(BuiltinClass::Fiber), "transfer", MethodKind::Method, arity, NativeBuiltin::FiberTransfer);
    }
    manifest.register_native(program, instance(BuiltinClass::Fiber), "transferError", MethodKind::Method, 1, NativeBuiltin::FiberTransferError);
    manifest.register_native(program, static_(BuiltinId::Fiber), "yield", MethodKind::Method, 0, NativeBuiltin::FiberYield);
    manifest.register_native(program, static_(BuiltinId::Fiber), "yield", MethodKind::Method, 1, NativeBuiltin::FiberYield);
    manifest.register_native(program, static_(BuiltinId::Fiber), "suspend", MethodKind::Method, 0, NativeBuiltin::FiberSuspend);
    manifest.register_native(program, static_(BuiltinId::Fiber), "abort", MethodKind::Method, 1, NativeBuiltin::FiberAbort);

    // Promise is the async result object. Settlement is VM-native because it
    // owns suspended continuation edges and participates directly in scheduling.
    manifest.register_native(program, static_(BuiltinId::Promise), "resolve", MethodKind::Method, 1, NativeBuiltin::PromiseResolve);
    manifest.register_native(program, static_(BuiltinId::Promise), "reject", MethodKind::Method, 1, NativeBuiltin::PromiseReject);
    manifest.register_native(program, instance(BuiltinClass::Promise), "cancel", MethodKind::Method, 0, NativeBuiltin::PromiseCancel);
    manifest.register_native(program, instance(BuiltinClass::Promise), "isPending", MethodKind::Getter, 0, NativeBuiltin::PromiseIsPending);
    manifest.register_native(program, instance(BuiltinClass::Promise), "isFulfilled", MethodKind::Getter, 0, NativeBuiltin::PromiseIsFulfilled);
    manifest.register_native(program, instance(BuiltinClass::Promise), "isRejected", MethodKind::Getter, 0, NativeBuiltin::PromiseIsRejected);
    manifest.register_native(program, instance(BuiltinClass::Promise), "isCancelled", MethodKind::Getter, 0, NativeBuiltin::PromiseIsCancelled);

    // Timer is a static-only engine class. All operations return cancellable
    // Promises; trailing Napalm blocks are simply the final closure argument.
    manifest.register_native(program, static_(BuiltinId::Timer), "delay", MethodKind::Method, 1, NativeBuiltin::TimerDelay);
    manifest.register_native(program, static_(BuiltinId::Timer), "delay", MethodKind::Method, 2, NativeBuiltin::TimerDelay);
    manifest.register_native(program, static_(BuiltinId::Timer), "microtask", MethodKind::Method, 1, NativeBuiltin::TimerMicrotask);
    manifest.register_native(program, static_(BuiltinId::Timer), "every", MethodKind::Method, 2, NativeBuiltin::TimerEvery);

    // Global fetch() is represented by the ordinary core-static callable object.
    // The VM-native implementation starts asynchronous HTTP work and immediately
    // returns a cancellable Napalm Promise<Response>.
    // Reactive is a compiler/runtime service. Keep its entry points internal so
    // reactivity can be detected before execution and the JIT never compiles a
    // field trace under the wrong tracking policy.
    let reactive = static_(BuiltinId::Reactive);
    manifest.register_native(program, reactive, "__registerComputed", MethodKind::Method, 1, NativeBuiltin::ReactiveComputed);
    manifest.register_native(program, reactive, "__registerWatcher", MethodKind::Method, 1, NativeBuiltin::ReactiveWatch);
    manifest.register_native(program, reactive, "__unwatch", MethodKind::Method, 1, NativeBuiltin::ReactiveUnwatch);
    manifest.register_native(program, reactive, "__beginComputed", MethodKind::Method, 1, NativeBuiltin::ReactiveBeginComputed);
    manifest.register_native(program, reactive, "__finishComputed", MethodKind::Method, 2, NativeBuiltin::ReactiveFinishComputed);
    manifest.register_native(program, reactive, "__abortComputed", MethodKind::Method, 1, NativeBuiltin::ReactiveAbortComputed);

    manifest.register_native(program, static_(BuiltinId::Fetch), "call", MethodKind::Method, 1, NativeBuiltin::FetchCall);
    manifest.register_native(program, static_(BuiltinId::Fetch), "call", MethodKind::Method, 2, NativeBuiltin::FetchCall);

    let response = instance(BuiltinClass::Response);
    manifest.register_native(program, response, "status", MethodKind::Getter, 0, NativeBuiltin::ResponseStatus);
    manifest.register_native(program, response, "statusText", MethodKind::Getter, 0, NativeBuiltin::ResponseStatusText);
    manifest.register_native(program, response, "ok", MethodKind::Getter, 0, NativeBuiltin::ResponseOk);
    manifest.register_native(program, response, "url", MethodKind::Getter, 0, NativeBuiltin::ResponseUrl);
    manifest.register_native(program, response, "redirected", MethodKind::Getter, 0, NativeBuiltin::ResponseRedirected);
    manifest.register_native(program, response, "headers", MethodKind::Getter, 0, NativeBuiltin::ResponseHeaders);
    manifest.register_native(program, response, "data", MethodKind::Getter, 0, NativeBuiltin::ResponseData);
    manifest.register_native(program, response, "bodyUsed", MethodKind::Getter, 0, NativeBuiltin::ResponseBodyUsed);
    manifest.register_native(program, response, "json", MethodKind::Method, 0, NativeBuiltin::ResponseJson);
    manifest.register_native(program, response, "text", MethodKind::Method, 0, NativeBuiltin::ResponseText);

    let headers = instance(BuiltinClass::Headers);
    manifest.register_native(program, headers, "get", MethodKind::Method, 1, NativeBuiltin::HeadersGet);
    manifest.register_native(program, headers, "has", MethodKind::Method, 1, NativeBuiltin::HeadersHas);

    // Dynamic C FFI. The embedding Host::call_foreign API remains a separate
    // mechanism; these objects are user-visible handles to dlopen/LoadLibrary +
    // libffi calls made by the VM itself.
    manifest.register_native(program, static_(BuiltinId::FFI), "open", MethodKind::Method, 1, NativeBuiltin::FfiOpen);
    manifest.register_native(program, static_(BuiltinId::FFI), "null", MethodKind::Getter, 0, NativeBuiltin::FfiNull);
    manifest.register_native(program, static_(BuiltinId::FFI), "lastError", MethodKind::Getter, 0, NativeBuiltin::FfiLastError);

    manifest.register_native(program, instance(BuiltinClass::FfiLibrary), "name", MethodKind::Getter, 0, NativeBuiltin::FfiLibraryName);
    manifest.register_native(program, instance(BuiltinClass::FfiLibrary), "function", MethodKind::Method, 3, NativeBuiltin::FfiLibraryFunction);

    manifest.register_native(program, instance(BuiltinClass::FfiFunction), "name", MethodKind::Getter, 0, NativeBuiltin::FfiFunctionName);
    for arity in 0..=16u8 {
        manifest.register_native(
            program,
            instance(BuiltinClass::FfiFunction),
            "call",
            MethodKind::Method,
            arity,
            NativeBuiltin::FfiFunctionCall,
        );
    }

    manifest.register_native(program, instance(BuiltinClass::FfiPointer), "isNull", MethodKind::Getter, 0, NativeBuiltin::FfiPointerIsNull);
    manifest.register_native(program, instance(BuiltinClass::FfiPointer), "offset", MethodKind::Method, 1, NativeBuiltin::FfiPointerOffset);
    manifest.register_native(program, instance(BuiltinClass::FfiPointer), "readCString", MethodKind::Method, 0, NativeBuiltin::FfiPointerReadCString);
    manifest.register_native(program, instance(BuiltinClass::FfiPointer), "toString", MethodKind::Getter, 0, NativeBuiltin::FfiPointerToString);
    manifest.register_native(program, instance(BuiltinClass::FfiPointer), "toString", MethodKind::Method, 0, NativeBuiltin::FfiPointerToString);

    // System is static-only; Date also owns heap-backed instance methods.
    let system = static_(BuiltinId::System);
    manifest.register(program, system, "print", MethodKind::Method, 0, BuiltinPrimitive::SystemPrint);
    let print_method = manifest.register(program, system, "print", MethodKind::Method, 1, BuiltinPrimitive::SystemPrint);
    let print_primitive = manifest.bindings.last().expect("System.print primitive was just registered").primitive;

    // System.print(value) participates in the exact same virtual conversion
    // protocol as `+` and interpolation. The bound bytecode wrapper overrides the
    // primitive method target below, while CallPrimitive keeps a direct handle to
    // the underlying string-printing primitive and avoids a second method lookup.
    let system_module = program.add_module("<core:system>", 0);
    let mut print = Builder::new(
        program,
        system_module,
        "System.print",
        FunctionKind::StaticMethod,
        vec![Type::Top, Type::Top],
        Type::Top,
    );
    let system_receiver = print.param(0);
    let value = print.param(1);
    let text = print.invoke_typed(value, to_string_method, Vec::new(), Type::String);
    // Preserve the historical `System.print(value)` result (the original value),
    // even though the primitive now receives the protocol-converted String.
    print.emit1(
        Inst::CallPrimitive {
            primitive: print_primitive,
            args: vec![system_receiver, text],
        },
        Type::String,
    );
    print.ret(value);
    manifest.system_print = Some((print_method, print.finish()));

    manifest.register(program, system, "write", MethodKind::Method, 1, BuiltinPrimitive::SystemWrite);
    manifest.register(program, system, "clock", MethodKind::Getter, 0, BuiltinPrimitive::SystemClock);
    manifest.register_native(program, system, "os", MethodKind::Method, 0, NativeBuiltin::SystemOs);
    manifest.register_native(program, system, "arch", MethodKind::Method, 0, NativeBuiltin::SystemArch);
    manifest.register_native(program, system, "gc", MethodKind::Method, 0, NativeBuiltin::SystemGc);

    // Date follows the JavaScript model: one heap object stores one TimeClip'd
    // millisecond value. Constructor/static overloads are represented by distinct
    // MethodSymbols because Napalm dispatch includes arity in the symbol identity.
    let date = static_(BuiltinId::Date);
    for arity in 0..=7u8 {
        manifest.register(program, date, "new", MethodKind::Method, arity, BuiltinPrimitive::DateNew);
        manifest.register(program, date, "UTC", MethodKind::Method, arity, BuiltinPrimitive::DateUtc);
    }
    manifest.register(program, date, "parse", MethodKind::Method, 1, BuiltinPrimitive::DateParse);
    manifest.register(program, date, "now", MethodKind::Method, 0, BuiltinPrimitive::DateNow);
    manifest.register(program, date, "now", MethodKind::Getter, 0, BuiltinPrimitive::DateNow);
    manifest.register(program, date, "nowMs", MethodKind::Method, 0, BuiltinPrimitive::DateNowMs);
    manifest.register(program, date, "nowMs", MethodKind::Getter, 0, BuiltinPrimitive::DateNowMs);
    manifest.register(program, date, "sleep", MethodKind::Method, 1, BuiltinPrimitive::DateSleep);

    let date_instance = instance(BuiltinClass::Date);
    for name in ["getTime", "valueOf"] {
        manifest.register(program, date_instance, name, MethodKind::Method, 0, BuiltinPrimitive::DateGetTime);
    }
    manifest.register(program, date_instance, "getTimezoneOffset", MethodKind::Method, 0, BuiltinPrimitive::DateTimezoneOffset);

    for (name, field, utc) in [
        ("getYear", DateField::YearLegacy, false),
        ("getFullYear", DateField::FullYear, false),
        ("getUTCFullYear", DateField::FullYear, true),
        ("getMonth", DateField::Month, false),
        ("getUTCMonth", DateField::Month, true),
        ("getDate", DateField::Date, false),
        ("getUTCDate", DateField::Date, true),
        ("getDay", DateField::Day, false),
        ("getUTCDay", DateField::Day, true),
        ("getHours", DateField::Hours, false),
        ("getUTCHours", DateField::Hours, true),
        ("getMinutes", DateField::Minutes, false),
        ("getUTCMinutes", DateField::Minutes, true),
        ("getSeconds", DateField::Seconds, false),
        ("getUTCSeconds", DateField::Seconds, true),
        ("getMilliseconds", DateField::Milliseconds, false),
        ("getUTCMilliseconds", DateField::Milliseconds, true),
    ] {
        manifest.register(program, date_instance, name, MethodKind::Method, 0, BuiltinPrimitive::DateGet(field, utc));
    }

    manifest.register(program, date_instance, "setTime", MethodKind::Method, 1, BuiltinPrimitive::DateSetTime);
    for (name, field, utc, arities) in [
        ("setYear", DateSetField::YearLegacy, false, 1u8..=1u8),
        ("setFullYear", DateSetField::FullYear, false, 1u8..=3u8),
        ("setUTCFullYear", DateSetField::FullYear, true, 1u8..=3u8),
        ("setMonth", DateSetField::Month, false, 1u8..=2u8),
        ("setUTCMonth", DateSetField::Month, true, 1u8..=2u8),
        ("setDate", DateSetField::Date, false, 1u8..=1u8),
        ("setUTCDate", DateSetField::Date, true, 1u8..=1u8),
        ("setHours", DateSetField::Hours, false, 1u8..=4u8),
        ("setUTCHours", DateSetField::Hours, true, 1u8..=4u8),
        ("setMinutes", DateSetField::Minutes, false, 1u8..=3u8),
        ("setUTCMinutes", DateSetField::Minutes, true, 1u8..=3u8),
        ("setSeconds", DateSetField::Seconds, false, 1u8..=2u8),
        ("setUTCSeconds", DateSetField::Seconds, true, 1u8..=2u8),
        ("setMilliseconds", DateSetField::Milliseconds, false, 1u8..=1u8),
        ("setUTCMilliseconds", DateSetField::Milliseconds, true, 1u8..=1u8),
    ] {
        for arity in arities {
            manifest.register(program, date_instance, name, MethodKind::Method, arity, BuiltinPrimitive::DateSet(field, utc));
        }
    }

    for (name, format) in [
        ("toUTCString", DateStringFormat::UtcString),
        ("toGMTString", DateStringFormat::UtcString),
        ("toISOString", DateStringFormat::IsoString),
        ("toDateString", DateStringFormat::DateString),
        ("toTimeString", DateStringFormat::TimeString),
    ] {
        manifest.register(program, date_instance, name, MethodKind::Method, 0, BuiltinPrimitive::DateToString(format));
    }
    // Locale/options arguments are accepted for JS source compatibility. The
    // current implementation deliberately uses the host locale and ignores an
    // explicit locale/options object; Intl-style formatting can be layered later.
    for (name, format) in [
        ("toLocaleString", DateStringFormat::LocaleString),
        ("toLocaleDateString", DateStringFormat::LocaleDateString),
        ("toLocaleTimeString", DateStringFormat::LocaleTimeString),
    ] {
        for arity in 0..=2u8 {
            manifest.register(program, date_instance, name, MethodKind::Method, arity, BuiltinPrimitive::DateToString(format));
        }
    }
    // The unified Napalm String protocol accepts both historical getter syntax and
    // JavaScript-style `toString()`. Date overrides the generic Object formatter.
    manifest.register(program, date_instance, "toString", MethodKind::Getter, 0, BuiltinPrimitive::DateToString(DateStringFormat::String));
    manifest.register(program, date_instance, "toString", MethodKind::Method, 0, BuiltinPrimitive::DateToString(DateStringFormat::String));
    manifest.register(program, date_instance, "toJSON", MethodKind::Method, 0, BuiltinPrimitive::DateToJson);
    manifest.register(program, date_instance, "toJSON", MethodKind::Method, 1, BuiltinPrimitive::DateToJson);

    manifest.register(program, static_(BuiltinId::Random), "new", MethodKind::Method, 0, BuiltinPrimitive::RandomNew);
    for arity in [1, 2] {
        manifest.register(program, instance(BuiltinClass::Random), "int", MethodKind::Method, arity, BuiltinPrimitive::RandomInt);
    }

    manifest
}

/// Connects the already-reserved manifest to a concrete interpreter.
pub fn install_vm_builtins(interpreter: &mut Interpreter<'_>, manifest: &BuiltinManifest) {
    if let Some(function) = manifest.string_right_add {
        interpreter.bind_string_right_add_function(function);
    }
    if let Some((method, function)) = manifest.string_add {
        interpreter.bind_builtin_function(BuiltinClass::String, method, function);
    }
    if let Some((method, function)) = manifest.list_remove {
        interpreter.bind_builtin_function(BuiltinClass::List, method, function);
    }
    for binding in manifest.bindings() {
        match binding.target {
            BuiltinTarget::Instance(class) => {
                interpreter.bind_builtin_primitive(class, binding.method, binding.primitive);
            }
            BuiltinTarget::Static(class) => {
                interpreter.bind_builtin_static_primitive(class, binding.method, binding.primitive);
            }
        }
    }
    for binding in &manifest.native_bindings {
        match binding.target {
            BuiltinTarget::Instance(class) => {
                interpreter.bind_builtin_native(class, binding.method, binding.native);
            }
            BuiltinTarget::Static(class) => {
                interpreter.bind_builtin_static_native(class, binding.method, binding.native);
            }
        }
    }
    // Bind wrapper functions after primitive/native tables so wrappers that use a
    // primitive internally (notably System.print/1) remain the visible target.
    if let Some((method, function)) = manifest.system_print {
        interpreter.bind_builtin_static_function(BuiltinId::System, method, function);
    }
    // Worker VMs need the same stock primitive/native manifest as the parent.
    // Store a clone only after all bindings have been installed so `launch` can
    // lazily create isolated worker interpreters with identical core semantics.
    interpreter.set_builtin_manifest(manifest.clone());
}

pub struct BuiltinHost {
    manifest: BuiltinManifest,
    started: Instant,
}

impl BuiltinHost {
    pub fn new(manifest: BuiltinManifest) -> Self {
        Self {
            manifest,
            started: Instant::now(),
        }
    }
}

impl Host for BuiltinHost {
    /// Tell the tracing JIT which primitive bindings have the exact stock Napalm
    /// numeric semantics that it is allowed to inline.  The JIT must not
    /// specialize merely by method name because embedders may replace primitive
    /// implementations.
    fn standard_numeric_method(&self, primitive: PrimitiveId, name: &str, arity: u8) -> bool {
        use BuiltinPrimitive::*;
        matches!(
            (self.manifest.implementation(primitive), name, arity),
            (Some(NumNeg), "-", 0)
                | (Some(NumAdd), "+", 1)
                | (Some(NumSub), "-", 1)
                | (Some(NumMul), "*", 1)
                | (Some(NumDiv), "/", 1)
                | (Some(NumMod), "%", 1)
                | (Some(NumLt), "<", 1)
                | (Some(NumLe), "<=", 1)
                | (Some(NumGt), ">", 1)
                | (Some(NumGe), ">=", 1)
                | (Some(NumEq), "==", 1)
                | (Some(NumNe), "!=", 1)
                | (Some(NumBitNot), "~", 0)
                | (Some(NumBitAnd), "&", 1)
                | (Some(NumBitOr), "|", 1)
                | (Some(NumBitXor), "^", 1)
                | (Some(NumShl), "<<", 1)
                | (Some(NumShr), ">>", 1)
                | (Some(NumAbs), "abs", 0)
                | (Some(NumFloor), "floor", 0)
                | (Some(NumCeil), "ceil", 0)
                | (Some(NumSqrt), "sqrt", 0)
                | (Some(NumRound), "round", 0)
                | (Some(NumSin), "sin", 0)
                | (Some(NumCos), "cos", 0)
                | (Some(NumTan), "tan", 0)
                | (Some(NumLog), "log", 0)
                | (Some(NumExp), "exp", 0)
                | (Some(NumIsNan), "isNan", 0)
                | (Some(NumIsInfinity), "isInfinity", 0)
                | (Some(NumIsInteger), "isInteger", 0)
        )
    }

    fn call_primitive(
        &mut self,
        context: &mut HostContext<'_>,
        primitive: PrimitiveId,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        let implementation = self
            .manifest
            .implementation(primitive)
            .ok_or_else(|| RuntimeError::new(format!("unregistered builtin primitive {primitive:?}")))?;
        execute_primitive(implementation, context, args, self.started)
    }
}

#[inline]
fn receiver(args: &[Value]) -> Result<Value, RuntimeError> {
    args.first().copied().ok_or_else(|| RuntimeError::new("builtin call has no receiver"))
}

#[inline]
fn argument(args: &[Value], index: usize) -> Result<Value, RuntimeError> {
    args.get(index + 1).copied().ok_or_else(|| RuntimeError::new("builtin argument missing"))
}

#[inline]
fn num(value: Value, what: &str) -> Result<f64, RuntimeError> {
    value.as_num().ok_or_else(|| RuntimeError::new(format!("{what} must be a Num")))
}

const DATE_MAX_TIME_MS: f64 = 8_640_000_000_000_000.0;
const DATE_MS_PER_SECOND: i128 = 1_000;
const DATE_MS_PER_MINUTE: i128 = 60 * DATE_MS_PER_SECOND;
const DATE_MS_PER_HOUR: i128 = 60 * DATE_MS_PER_MINUTE;
const DATE_MS_PER_DAY: i128 = 24 * DATE_MS_PER_HOUR;
const DATE_DAY_NAMES: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DATE_MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

#[derive(Clone, Copy, Debug)]
struct DateParts {
    year: i64,
    month: u32,
    day: u32,
    weekday: u32,
    hour: u32,
    minute: u32,
    second: u32,
    millisecond: u32,
    /// Seconds east of UTC. ECMAScript getTimezoneOffset returns the negation.
    offset_seconds: i32,
}

#[inline]
fn time_clip(value: f64) -> f64 {
    if !value.is_finite() || value.abs() > DATE_MAX_TIME_MS {
        f64::NAN
    } else {
        value.trunc()
    }
}

#[inline]
fn current_epoch_millis() -> f64 {
    let value = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs_f64() * 1000.0,
        Err(error) => -(error.duration().as_secs_f64() * 1000.0),
    };
    time_clip(value)
}

#[inline]
fn finite_date_integer(value: f64) -> Option<i128> {
    // All TimeClip-reachable component values fit comfortably below f64's exact
    // integer limit. Larger source values may only produce an invalid Date unless
    // paired with equally extreme cancellation, which Napalm intentionally avoids
    // materializing through gigantic intermediate calendar arithmetic.
    const LIMIT: f64 = 9_007_199_254_740_991.0;
    (value.is_finite() && value.abs() <= LIMIT).then(|| value.trunc() as i128)
}

/// Howard Hinnant-style proleptic Gregorian conversion, expressed independently
/// here so UTC Date operations support the entire ECMAScript TimeClip range.
#[inline]
fn date_days_from_civil(year: i128, month_one: i128, day: i128) -> i128 {
    let adjusted_year = year - if month_one <= 2 { 1 } else { 0 };
    let era = adjusted_year.div_euclid(400);
    let yoe = adjusted_year - era * 400;
    let shifted_month = month_one + if month_one > 2 { -3 } else { 9 };
    let doy = (153 * shifted_month + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[inline]
fn date_civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days as i128 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    (year as i64, month as u32, day as u32)
}

fn date_make_utc_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
    millisecond: f64,
    adjust_short_year: bool,
) -> f64 {
    let (Some(mut year), Some(month), Some(day), Some(hour), Some(minute), Some(second), Some(millisecond)) = (
        finite_date_integer(year),
        finite_date_integer(month),
        finite_date_integer(day),
        finite_date_integer(hour),
        finite_date_integer(minute),
        finite_date_integer(second),
        finite_date_integer(millisecond),
    ) else {
        return f64::NAN;
    };
    if adjust_short_year && (0..=99).contains(&year) {
        year += 1900;
    }
    year += month.div_euclid(12);
    let month_one = month.rem_euclid(12) + 1;
    let days = date_days_from_civil(year, month_one, 1) + day - 1;
    let millis = days
        .saturating_mul(DATE_MS_PER_DAY)
        .saturating_add(hour.saturating_mul(DATE_MS_PER_HOUR))
        .saturating_add(minute.saturating_mul(DATE_MS_PER_MINUTE))
        .saturating_add(second.saturating_mul(DATE_MS_PER_SECOND))
        .saturating_add(millisecond);
    if millis.abs() > DATE_MAX_TIME_MS as i128 {
        f64::NAN
    } else {
        time_clip(millis as f64)
    }
}

fn date_utc_parts(millis: f64) -> Option<DateParts> {
    let millis = time_clip(millis);
    if !millis.is_finite() {
        return None;
    }
    let millis = millis as i64;
    let day_ms = DATE_MS_PER_DAY as i64;
    let days = millis.div_euclid(day_ms);
    let mut rest = millis.rem_euclid(day_ms);
    let hour = (rest / DATE_MS_PER_HOUR as i64) as u32;
    rest %= DATE_MS_PER_HOUR as i64;
    let minute = (rest / DATE_MS_PER_MINUTE as i64) as u32;
    rest %= DATE_MS_PER_MINUTE as i64;
    let second = (rest / DATE_MS_PER_SECOND as i64) as u32;
    let millisecond = (rest % DATE_MS_PER_SECOND as i64) as u32;
    let (year, month_one, day) = date_civil_from_days(days);
    Some(DateParts {
        year,
        month: month_one - 1,
        day,
        weekday: (days + 4).rem_euclid(7) as u32,
        hour,
        minute,
        second,
        millisecond,
        offset_seconds: 0,
    })
}

#[inline]
fn date_pick_local(result: LocalResult<DateTime<Local>>) -> Option<DateTime<Local>> {
    match result {
        LocalResult::Single(value) => Some(value),
        LocalResult::Ambiguous(first, second) => {
            if first.timestamp_millis() <= second.timestamp_millis() { Some(first) } else { Some(second) }
        }
        LocalResult::None => None,
    }
}

fn date_local_datetime(millis: f64) -> Option<DateTime<Local>> {
    let millis = time_clip(millis);
    if !millis.is_finite() {
        return None;
    }
    Utc.timestamp_millis_opt(millis as i64)
        .single()
        .map(|value| value.with_timezone(&Local))
}

fn date_local_parts(millis: f64) -> Option<DateParts> {
    let value = date_local_datetime(millis)?;
    Some(DateParts {
        year: value.year() as i64,
        month: value.month0(),
        day: value.day(),
        weekday: value.weekday().num_days_from_sunday(),
        hour: value.hour(),
        minute: value.minute(),
        second: value.second(),
        millisecond: value.timestamp_subsec_millis(),
        offset_seconds: value.offset().fix().local_minus_utc(),
    })
}

/// Convert normalized wall-clock fields through the host timezone. DST overlaps
/// choose the earlier instant, matching JavaScript's compatible disambiguation.
/// For a DST gap, shift by the observed offset delta and retry.
fn date_local_from_naive(naive: NaiveDateTime) -> Option<f64> {
    match Local.from_local_datetime(&naive) {
        LocalResult::Single(value) => return Some(time_clip(value.timestamp_millis() as f64)),
        LocalResult::Ambiguous(first, second) => {
            let value = if first.timestamp_millis() <= second.timestamp_millis() { first } else { second };
            return Some(time_clip(value.timestamp_millis() as f64));
        }
        LocalResult::None => {}
    }

    let mut before = None;
    let mut after = None;
    for minutes in 1..=24 * 60 {
        if before.is_none() {
            before = date_pick_local(Local.from_local_datetime(&(naive - ChronoDuration::minutes(minutes))));
        }
        if after.is_none() {
            after = date_pick_local(Local.from_local_datetime(&(naive + ChronoDuration::minutes(minutes))));
        }
        if before.is_some() && after.is_some() {
            break;
        }
    }
    let (before, after) = (before?, after?);
    let before_offset = before.offset().fix().local_minus_utc();
    let after_offset = after.offset().fix().local_minus_utc();
    let gap = after_offset - before_offset;
    if gap > 0 {
        let shifted = naive + ChronoDuration::seconds(gap as i64);
        if let Some(value) = date_pick_local(Local.from_local_datetime(&shifted)) {
            return Some(time_clip(value.timestamp_millis() as f64));
        }
    }
    Some(time_clip(after.timestamp_millis() as f64))
}

fn date_make_local_components(
    year: f64,
    month: f64,
    day: f64,
    hour: f64,
    minute: f64,
    second: f64,
    millisecond: f64,
    adjust_short_year: bool,
) -> f64 {
    // Normalize overflow in a timezone-independent civil domain first.
    let normalized = date_make_utc_components(
        year, month, day, hour, minute, second, millisecond, adjust_short_year,
    );
    let Some(parts) = date_utc_parts(normalized) else { return f64::NAN; };
    let Ok(year) = i32::try_from(parts.year) else { return normalized; };
    let Some(date) = NaiveDate::from_ymd_opt(year, parts.month + 1, parts.day) else {
        // Chrono intentionally has a smaller civil range than ECMAScript. UTC
        // arithmetic remains correct for those extreme years; local timezone
        // databases do not have meaningful rules there, so fall back to UTC.
        return normalized;
    };
    let Some(naive) = date.and_hms_milli_opt(
        parts.hour,
        parts.minute,
        parts.second,
        parts.millisecond,
    ) else {
        return f64::NAN;
    };
    date_local_from_naive(naive).unwrap_or(normalized)
}

fn date_number(context: &HostContext<'_>, value: Value) -> Option<f64> {
    if let Some(value) = value.as_num() { return Some(value); }
    if let Some(value) = value.as_bool() { return Some(if value { 1.0 } else { 0.0 }); }
    if value.is_null() { return Some(0.0); }
    if let Some(text) = context.string_utf8(value) {
        let text = text.trim();
        if text.is_empty() { return Some(0.0); }
        return text.parse::<f64>().ok();
    }
    None
}

fn date_component_args(context: &HostContext<'_>, args: &[Value]) -> Option<Vec<f64>> {
    args.iter().copied().map(|value| date_number(context, value)).collect()
}

fn date_parse_date_only(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year = std::str::from_utf8(&bytes[0..4]).ok()?.parse::<f64>().ok()?;
    let month = std::str::from_utf8(&bytes[5..7]).ok()?.parse::<f64>().ok()?;
    let day = std::str::from_utf8(&bytes[8..10]).ok()?.parse::<f64>().ok()?;
    if !(1.0..=12.0).contains(&month) || !(1.0..=31.0).contains(&day) {
        return Some(f64::NAN);
    }
    let value = date_make_utc_components(year, month - 1.0, day, 0.0, 0.0, 0.0, 0.0, false);
    let parts = date_utc_parts(value)?;
    if parts.year as f64 == year && parts.month as f64 == month - 1.0 && parts.day as f64 == day {
        Some(value)
    } else {
        Some(f64::NAN)
    }
}

fn date_parse_string(text: &str) -> f64 {
    let text = text.trim();
    if text.is_empty() {
        return f64::NAN;
    }
    if let Some(value) = date_parse_date_only(text) {
        return value;
    }
    if let Ok(value) = DateTime::parse_from_rfc3339(text) {
        return time_clip(value.timestamp_millis() as f64);
    }
    if let Ok(value) = DateTime::parse_from_rfc2822(text) {
        return time_clip(value.timestamp_millis() as f64);
    }
    if let Ok(value) = NaiveDateTime::parse_from_str(text, "%a, %d %b %Y %H:%M:%S GMT") {
        return time_clip(Utc.from_utc_datetime(&value).timestamp_millis() as f64);
    }

    // Parse the exact form emitted by Date.toString(). The timezone abbreviation
    // is decorative; the numeric GMT offset is authoritative.
    let core = text.split_once(" (").map_or(text, |(head, _)| head);
    if let Ok(value) = DateTime::<chrono::FixedOffset>::parse_from_str(core, "%a %b %d %Y %H:%M:%S GMT%z") {
        return time_clip(value.timestamp_millis() as f64);
    }

    // ISO date-time without an explicit offset is local time in JavaScript.
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M"] {
        if let Ok(value) = NaiveDateTime::parse_from_str(text, format) {
            if let Some(millis) = date_local_from_naive(value) {
                return millis;
            }
        }
    }
    f64::NAN
}

fn date_format_year_iso(year: i64) -> String {
    if (0..=9999).contains(&year) {
        format!("{year:04}")
    } else if year < 0 {
        format!("-{:06}", year.unsigned_abs())
    } else {
        format!("+{year:06}")
    }
}

fn date_format_iso(millis: f64) -> Result<String, RuntimeError> {
    let parts = date_utc_parts(millis)
        .ok_or_else(|| RuntimeError::new("RangeError: Invalid time value"))?;
    Ok(format!(
        "{}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        date_format_year_iso(parts.year),
        parts.month + 1,
        parts.day,
        parts.hour,
        parts.minute,
        parts.second,
        parts.millisecond,
    ))
}

fn date_format_utc_string(millis: f64) -> String {
    let Some(parts) = date_utc_parts(millis) else { return "Invalid Date".to_owned(); };
    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} GMT",
        DATE_DAY_NAMES[parts.weekday as usize],
        parts.day,
        DATE_MONTH_NAMES[parts.month as usize],
        if (0..=9999).contains(&parts.year) { format!("{:04}", parts.year) } else { parts.year.to_string() },
        parts.hour,
        parts.minute,
        parts.second,
    )
}

#[inline]
fn date_format_offset(offset_seconds: i32) -> String {
    let sign = if offset_seconds >= 0 { '+' } else { '-' };
    let value = offset_seconds.unsigned_abs();
    format!("{sign}{:02}{:02}", value / 3600, (value % 3600) / 60)
}

fn date_format_local(millis: f64, format: DateStringFormat) -> Result<String, RuntimeError> {
    if matches!(format, DateStringFormat::IsoString) {
        return date_format_iso(millis);
    }
    if matches!(format, DateStringFormat::UtcString) {
        return Ok(date_format_utc_string(millis));
    }
    let Some(parts) = date_local_parts(millis).or_else(|| date_utc_parts(millis)) else {
        return Ok("Invalid Date".to_owned());
    };
    let zone = date_local_datetime(millis)
        .map(|value| value.format("%Z").to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "UTC".to_owned());
    let offset = date_format_offset(parts.offset_seconds);
    let year = if (0..=9999).contains(&parts.year) { format!("{:04}", parts.year) } else { parts.year.to_string() };
    Ok(match format {
        DateStringFormat::String => format!(
            "{} {} {:02} {} {:02}:{:02}:{:02} GMT{} ({})",
            DATE_DAY_NAMES[parts.weekday as usize],
            DATE_MONTH_NAMES[parts.month as usize],
            parts.day,
            year,
            parts.hour,
            parts.minute,
            parts.second,
            offset,
            zone,
        ),
        DateStringFormat::DateString => format!(
            "{} {} {:02} {}",
            DATE_DAY_NAMES[parts.weekday as usize],
            DATE_MONTH_NAMES[parts.month as usize],
            parts.day,
            year,
        ),
        DateStringFormat::TimeString => format!(
            "{:02}:{:02}:{:02} GMT{} ({})",
            parts.hour, parts.minute, parts.second, offset, zone,
        ),
        DateStringFormat::LocaleString => format!(
            "{:02}/{:02}/{}, {:02}:{:02}:{:02}",
            parts.month + 1, parts.day, year, parts.hour, parts.minute, parts.second,
        ),
        DateStringFormat::LocaleDateString => format!("{:02}/{:02}/{}", parts.month + 1, parts.day, year),
        DateStringFormat::LocaleTimeString => format!("{:02}:{:02}:{:02}", parts.hour, parts.minute, parts.second),
        DateStringFormat::UtcString | DateStringFormat::IsoString => unreachable!(),
    })
}

#[inline]
fn date_field_value(parts: DateParts, field: DateField) -> f64 {
    match field {
        DateField::YearLegacy => parts.year as f64 - 1900.0,
        DateField::FullYear => parts.year as f64,
        DateField::Month => parts.month as f64,
        DateField::Date => parts.day as f64,
        DateField::Day => parts.weekday as f64,
        DateField::Hours => parts.hour as f64,
        DateField::Minutes => parts.minute as f64,
        DateField::Seconds => parts.second as f64,
        DateField::Milliseconds => parts.millisecond as f64,
    }
}

fn date_set_components(
    context: &HostContext<'_>,
    current: f64,
    field: DateSetField,
    utc: bool,
    values: &[Value],
) -> f64 {
    let base = if current.is_finite() {
        if utc { date_utc_parts(current) } else { date_local_parts(current).or_else(|| date_utc_parts(current)) }
    } else if matches!(field, DateSetField::FullYear | DateSetField::YearLegacy) {
        // ECMAScript setFullYear can recover an invalid Date using +0 as its base.
        if utc { date_utc_parts(0.0) } else { date_local_parts(0.0).or_else(|| date_utc_parts(0.0)) }
    } else {
        None
    };
    let Some(parts) = base else { return f64::NAN; };
    let Some(values) = date_component_args(context, values) else { return f64::NAN; };
    let get = |index: usize, default: f64| values.get(index).copied().unwrap_or(default);

    let (year, month, day, hour, minute, second, millisecond) = match field {
        DateSetField::YearLegacy => {
            let mut year = get(0, parts.year as f64);
            if year.is_finite() && (0.0..=99.0).contains(&year.trunc()) {
                year = year.trunc() + 1900.0;
            }
            (
                year,
                parts.month as f64,
                parts.day as f64,
                parts.hour as f64,
                parts.minute as f64,
                parts.second as f64,
                parts.millisecond as f64,
            )
        }
        DateSetField::FullYear => (
            get(0, parts.year as f64),
            get(1, parts.month as f64),
            get(2, parts.day as f64),
            parts.hour as f64,
            parts.minute as f64,
            parts.second as f64,
            parts.millisecond as f64,
        ),
        DateSetField::Month => (
            parts.year as f64,
            get(0, parts.month as f64),
            get(1, parts.day as f64),
            parts.hour as f64,
            parts.minute as f64,
            parts.second as f64,
            parts.millisecond as f64,
        ),
        DateSetField::Date => (
            parts.year as f64,
            parts.month as f64,
            get(0, parts.day as f64),
            parts.hour as f64,
            parts.minute as f64,
            parts.second as f64,
            parts.millisecond as f64,
        ),
        DateSetField::Hours => (
            parts.year as f64,
            parts.month as f64,
            parts.day as f64,
            get(0, parts.hour as f64),
            get(1, parts.minute as f64),
            get(2, parts.second as f64),
            get(3, parts.millisecond as f64),
        ),
        DateSetField::Minutes => (
            parts.year as f64,
            parts.month as f64,
            parts.day as f64,
            parts.hour as f64,
            get(0, parts.minute as f64),
            get(1, parts.second as f64),
            get(2, parts.millisecond as f64),
        ),
        DateSetField::Seconds => (
            parts.year as f64,
            parts.month as f64,
            parts.day as f64,
            parts.hour as f64,
            parts.minute as f64,
            get(0, parts.second as f64),
            get(1, parts.millisecond as f64),
        ),
        DateSetField::Milliseconds => (
            parts.year as f64,
            parts.month as f64,
            parts.day as f64,
            parts.hour as f64,
            parts.minute as f64,
            parts.second as f64,
            get(0, parts.millisecond as f64),
        ),
    };
    if utc {
        date_make_utc_components(year, month, day, hour, minute, second, millisecond, false)
    } else {
        date_make_local_components(year, month, day, hour, minute, second, millisecond, false)
    }
}

#[inline]
fn integer(value: Value, what: &str) -> Result<i64, RuntimeError> {
    let n = num(value, what)?;
    if !n.is_finite() || n.fract() != 0.0 {
        return Err(RuntimeError::new(format!("{what} must be an integer")));
    }
    Ok(n as i64)
}

#[inline]
fn bitwise_operand(value: Value, what: &str) -> Result<u32, RuntimeError> {
    Ok(num_to_u32(num(value, what)?))
}

fn normalized_index(index: i64, len: usize) -> Result<usize, RuntimeError> {
    let len_i = len as i64;
    let index = if index < 0 { len_i + index } else { index };
    if index < 0 || index >= len_i {
        Err(RuntimeError::new("index out of bounds"))
    } else {
        Ok(index as usize)
    }
}

/// Resolve the containing List or Map, leaving the final subscript for the
/// getter or setter. Intermediate list indices cannot be slices; map keys
/// retain their usual semantics regardless of the depth of the lookup.
fn nested_subscript_target(
    context: &HostContext<'_>,
    mut container: Value,
    indices: &[Value],
) -> Result<(Value, Value), RuntimeError> {
    let (&last, path) = indices.split_last()
        .ok_or_else(|| RuntimeError::new("builtin argument missing"))?;
    for &subscript in path {
        container = if let Some(len) = context.list_len(container) {
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_get(container, index)
                .ok_or_else(|| RuntimeError::new("list index out of bounds"))?
        } else if context.map_len(container).is_some() {
            context.map_get(container, subscript)?
        } else {
            return Err(RuntimeError::new("receiver is not List or Map"));
        };
    }
    Ok((container, last))
}

fn slice_bounds(
    from: f64,
    to: f64,
    inclusive: bool,
    len: usize,
) -> Result<(usize, usize, bool), RuntimeError> {
    let length = len as f64;
    // Napalm permits an empty copy at the end, including copies of an empty list.
    if from == length && to == if inclusive { -1.0 } else { length } {
        return Ok((0, 0, false));
    }
    if !from.is_finite() || from.fract() != 0.0 {
        return Err(RuntimeError::new("range start must be an integer"));
    }
    let start = if from < 0.0 { length + from } else { from };
    if start < 0.0 || start >= length {
        return Err(RuntimeError::new("range start out of bounds"));
    }
    if !to.is_finite() || to.fract() != 0.0 {
        return Err(RuntimeError::new("range end must be an integer"));
    }
    let mut end = if to < 0.0 { length + to } else { to };
    if !inclusive {
        if start == end {
            return Ok((start as usize, 0, false));
        }
        end += if end >= start { -1.0 } else { 1.0 };
    }
    if end < 0.0 || end >= length {
        return Err(RuntimeError::new("range end out of bounds"));
    }
    let start = start as usize;
    let end = end as usize;
    Ok((start, start.abs_diff(end) + 1, end < start))
}

fn execute_primitive(
    op: BuiltinPrimitive,
    context: &mut HostContext<'_>,
    args: &[Value],
    started: Instant,
) -> Result<Value, RuntimeError> {
    use BuiltinPrimitive::*;

    let this = receiver(args)?;
    match op {
        RandomNew => {
            Ok(context.allocate_random(fast_random_seed()))
        }
        RandomInt => {
            let (from, to) = if args.len() == 2 {
                (0.0, num(argument(args, 0)?, "upper bound")?)
            } else {
                (num(argument(args, 0)?, "lower bound")?, num(argument(args, 1)?, "upper bound")?)
            };
            const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
            for bound in [from, to] {
                if !bound.is_finite() || bound.fract() != 0.0 || bound.abs() > MAX_SAFE_INTEGER {
                    return Err(RuntimeError::new("Random.int bounds must be safe finite integers"));
                }
            }
            if from >= to {
                return Err(RuntimeError::new("Random.int upper bound must be greater than lower bound"));
            }
            let lower = from as i64;
            let width = (to as i64 - lower) as u64;
            // Rejection avoids the bias of reducing every u64 modulo width.
            let threshold = width.wrapping_neg() % width;
            loop {
                let value = context.random_u64(this)?;
                if value >= threshold {
                    return Ok(Value::from_num((lower + (value % width) as i64) as f64));
                }
            }
        }
        RangeCount => {
            let (from, to, inclusive) = context.range_parts(this)
                .ok_or_else(|| RuntimeError::new("receiver is not Range"))?;
            let distance = (to - from).abs();
            Ok(Value::from_num(if inclusive { distance.floor() + 1.0 } else { distance.ceil() }))
        }
        ListIsEmpty => Ok(Value::from_bool(context.list_len(this)
            .ok_or_else(|| RuntimeError::new("receiver is not List"))? == 0)),
        MapIsEmpty => Ok(Value::from_bool(context.map_len(this)
            .ok_or_else(|| RuntimeError::new("receiver is not Map"))? == 0)),
        StringIsEmpty => Ok(Value::from_bool(context.string_bytes(this)
            .ok_or_else(|| RuntimeError::new("receiver is not String"))?.is_empty())),
        MapKeysIsEmpty => {
            let map = context.map_keys_source(this)?;
            Ok(Value::from_bool(context.map_len(map)
                .ok_or_else(|| RuntimeError::new("receiver is not Map"))? == 0))
        }
        ObjectEq => Ok(Value::from_bool(this.raw_bits() == argument(args, 0)?.raw_bits())),
        ObjectNe => Ok(Value::from_bool(this.raw_bits() != argument(args, 0)?.raw_bits())),
        ObjectToString | BoolToString | NullToString | NumToString | ListToString | MapToString | RangeToString => {
            Ok(context.allocate_string(context.format_value(this)))
        }
        ObjectType => Ok(Value::NULL),
        BoolNot => Ok(Value::from_bool(!this.as_bool().ok_or_else(|| RuntimeError::new("receiver is not Bool"))?)),
        NullNot => Ok(Value::TRUE),

        NumAdd => {
            let lhs = num(this, "receiver")?;
            let rhs = argument(args, 0)?;
            if let Some(rhs_num) = rhs.as_num() {
                Ok(Value::from_num(lhs + rhs_num))
            } else if let Some(rhs_bytes) = context.string_bytes(rhs).map(|bytes| bytes.to_vec()) {
                let mut out = context.format_value(this).into_bytes();
                out.extend_from_slice(&rhs_bytes);
                Ok(context.allocate_bytes(out))
            } else {
                Err(RuntimeError::new("Number.+ expects Number or String"))
            }
        },
        NumSub => Ok(Value::from_num(num(this, "receiver")? - num(argument(args, 0)?, "argument")?)),
        NumMul => Ok(Value::from_num(num(this, "receiver")? * num(argument(args, 0)?, "argument")?)),
        NumPow => Ok(Value::from_num(num(this, "receiver")?.powf(num(argument(args, 0)?, "argument")?))),
        NumDiv => Ok(Value::from_num(num(this, "receiver")? / num(argument(args, 0)?, "argument")?)),
        NumMod => Ok(Value::from_num(num(this, "receiver")? % num(argument(args, 0)?, "argument")?)),
        NumInclusiveRange | NumExclusiveRange => {
            let from = num(this, "range start")?;
            let to = num(argument(args, 0)?, "range end")?;
            Ok(context.allocate_range(from, to, matches!(op, NumInclusiveRange)))
        }
        RangeNew => {
            let from = num(argument(args, 0)?, "range start")?;
            let to = num(argument(args, 1)?, "range end")?;
            let inclusive = if args.len() > 3 {
                argument(args, 2)?.as_bool().ok_or_else(|| RuntimeError::new("inclusive must be a boolean"))?
            } else { false };
            Ok(context.allocate_range(from, to, inclusive))
        }
        NumLt => Ok(Value::from_bool(num(this, "receiver")? < num(argument(args, 0)?, "argument")?)),
        NumLe => Ok(Value::from_bool(num(this, "receiver")? <= num(argument(args, 0)?, "argument")?)),
        NumGt => Ok(Value::from_bool(num(this, "receiver")? > num(argument(args, 0)?, "argument")?)),
        NumGe => Ok(Value::from_bool(num(this, "receiver")? >= num(argument(args, 0)?, "argument")?)),
        NumEq => Ok(Value::from_bool(argument(args, 0)?.as_num().is_some_and(|rhs| num(this, "receiver").unwrap() == rhs))),
        NumNe => Ok(Value::from_bool(!argument(args, 0)?.as_num().is_some_and(|rhs| num(this, "receiver").unwrap() == rhs))),
        NumNeg => Ok(Value::from_num(-num(this, "receiver")?)),
        NumBitNot => Ok(Value::from_num((!bitwise_operand(this, "receiver")?) as f64)),
        NumBitAnd => Ok(Value::from_num((bitwise_operand(this, "receiver")? & bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumBitOr => Ok(Value::from_num((bitwise_operand(this, "receiver")? | bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumBitXor => Ok(Value::from_num((bitwise_operand(this, "receiver")? ^ bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumShl => Ok(Value::from_num(bitwise_operand(this, "receiver")?.wrapping_shl(bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumShr => Ok(Value::from_num(bitwise_operand(this, "receiver")?.wrapping_shr(bitwise_operand(argument(args, 0)?, "argument")?) as f64)),
        NumAbs => Ok(Value::from_num(num(this, "receiver")?.abs())),
        NumFloor => Ok(Value::from_num(num(this, "receiver")?.floor())),
        NumCeil => Ok(Value::from_num(num(this, "receiver")?.ceil())),
        NumRound => Ok(Value::from_num(num(this, "receiver")?.round())),
        NumSqrt => Ok(Value::from_num(num(this, "receiver")?.sqrt())),
        NumSin => Ok(Value::from_num(num(this, "receiver")?.sin())),
        NumCos => Ok(Value::from_num(num(this, "receiver")?.cos())),
        NumTan => Ok(Value::from_num(num(this, "receiver")?.tan())),
        NumLog => Ok(Value::from_num(num(this, "receiver")?.ln())),
        NumExp => Ok(Value::from_num(num(this, "receiver")?.exp())),
        NumIsInteger => { let n = num(this, "receiver")?; Ok(Value::from_bool(n.is_finite() && n.fract() == 0.0)) }
        NumIsNan => Ok(Value::from_bool(num(this, "receiver")?.is_nan())),
        NumIsInfinity => Ok(Value::from_bool(num(this, "receiver")?.is_infinite())),
        NumFromString => {
            let value = argument(args, 0)?;
            let text = context.string_utf8(value).ok_or_else(|| RuntimeError::new("Num.fromString expects String"))?;
            Ok(text.trim().parse::<f64>().map(Value::from_num).unwrap_or(Value::NULL))
        }
        NumPi => Ok(Value::from_num(std::f64::consts::PI)),
        NumTau => Ok(Value::from_num(std::f64::consts::TAU)),
        NumInfinity => Ok(Value::from_num(f64::INFINITY)),
        NumNan => Ok(Value::from_num(f64::NAN)),

        StringCount => Ok(Value::from_num(context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?.len() as f64)),
        StringAdd => {
            // Kept as a compatibility fallback for programs compiled against an
            // older core manifest. New programs bind String.+ to the WIR helper
            // above so user-defined `toString()` overrides participate normally.
            let rhs = argument(args, 0)?;
            let mut out = context.string_bytes(this)
                .ok_or_else(|| RuntimeError::new("receiver is not String"))?
                .to_vec();
            if let Some(bytes) = context.string_bytes(rhs).map(|bytes| bytes.to_vec()) {
                out.extend_from_slice(&bytes);
            } else {
                out.extend_from_slice(context.format_value(rhs).as_bytes());
            }
            Ok(context.allocate_bytes(out))
        }
        StringEq | StringNe => {
            let rhs = argument(args, 0)?;
            let equal = context.string_bytes(this).zip(context.string_bytes(rhs)).is_some_and(|(a,b)| a == b);
            Ok(Value::from_bool(if matches!(op, StringEq) { equal } else { !equal }))
        }
        StringSubscript => {
            let byte = {
                let bytes = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
                let index = normalized_index(integer(argument(args, 0)?, "index")?, bytes.len())?;
                bytes[index]
            };
            Ok(context.allocate_bytes([byte]))
        }
        StringContains | StringStartsWith | StringEndsWith | StringIndexOf => {
            let haystack = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
            let needle = context.string_bytes(argument(args, 0)?).ok_or_else(|| RuntimeError::new("argument must be String"))?;
            let position = haystack.windows(needle.len()).position(|w| w == needle);
            match op {
                StringContains => Ok(Value::from_bool(position.is_some())),
                StringStartsWith => Ok(Value::from_bool(haystack.starts_with(needle))),
                StringEndsWith => Ok(Value::from_bool(haystack.ends_with(needle))),
                StringIndexOf => Ok(Value::from_num(position.map_or(-1.0, |p| p as f64))),
                _ => unreachable!(),
            }
        }
        StringToString => Ok(this),
        StringIterate => {
            let len = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?.len();
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        StringIteratorValue => {
            let byte = {
                let bytes = context.string_bytes(this).ok_or_else(|| RuntimeError::new("receiver is not String"))?;
                let index = normalized_index(integer(argument(args,0)?, "iterator")?, bytes.len())?;
                bytes[index]
            };
            Ok(context.allocate_bytes([byte]))
        }
        StringFromCodePoint => {
            let point = integer(argument(args, 0)?, "code point")?;
            let ch = char::from_u32(point as u32).ok_or_else(|| RuntimeError::new("invalid Unicode code point"))?;
            let mut bytes = [0u8; 4];
            Ok(context.allocate_string(ch.encode_utf8(&mut bytes)))
        }

        ListCount => Ok(Value::from_num(context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))? as f64)),
        ListConcat => {
            let rhs = argument(args, 0)?;
            let left_len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let right_len = context.list_len(rhs).ok_or_else(|| RuntimeError::new("List.+ expects List"))?;
            let capacity = left_len.checked_add(right_len)
                .ok_or_else(|| RuntimeError::new("concatenated list is too large"))?;
            let mut values = Vec::with_capacity(capacity);
            for (list, len) in [(this, left_len), (rhs, right_len)] {
                for index in 0..len {
                    values.push(context.list_get(list, index)
                        .ok_or_else(|| RuntimeError::new("list index out of bounds"))?);
                }
            }
            Ok(context.allocate_list(values))
        }
        ListAdd => context.list_push(this, argument(args,0)?),
        ListInsert => {
            let index = integer(argument(args,0)?, "index")?;
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = if index < 0 { (len as i64 + index + 1).max(0) } else { index } as usize;
            context.list_insert(this, index, argument(args,1)?)
        }
        ListRemoveAt => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = normalized_index(integer(argument(args,0)?, "index")?, len)?;
            context.list_remove_at(this, index)
        }
        ListClear => { context.list_clear(this)?; Ok(Value::NULL) }
        ListSubscript | MapSubscript => {
            let (container, subscript) = nested_subscript_target(context, this, &args[1..])?;
            if context.map_len(container).is_some() {
                return context.map_get(container, subscript);
            }
            let len = context.list_len(container).ok_or_else(|| RuntimeError::new("receiver is not List or Map"))?;
            if let Some((from, to, inclusive)) = context.range_parts(subscript) {
                let (start, count, reverse) = slice_bounds(from, to, inclusive, len)?;
                let mut values = Vec::with_capacity(count);
                for offset in 0..count {
                    let index = if reverse { start - offset } else { start + offset };
                    values.push(context.list_get(container, index)
                        .ok_or_else(|| RuntimeError::new("list index out of bounds"))?);
                }
                return Ok(context.allocate_list(values));
            }
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_get(container,index).ok_or_else(|| RuntimeError::new("list index out of bounds"))
        }
        ListSubscriptSet | MapSubscriptSet => {
            let (&value, indices) = args[1..].split_last()
                .ok_or_else(|| RuntimeError::new("builtin argument missing"))?;
            let (container, subscript) = nested_subscript_target(context, this, indices)?;
            if context.map_len(container).is_some() {
                return context.map_set(container, subscript, value);
            }
            let len = context.list_len(container).ok_or_else(|| RuntimeError::new("receiver is not List or Map"))?;
            let index = normalized_index(integer(subscript, "index")?, len)?;
            context.list_set(container,index,value)
        }
        ListIterate => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        ListIteratorValue => {
            let len = context.list_len(this).ok_or_else(|| RuntimeError::new("receiver is not List"))?;
            let index = normalized_index(integer(argument(args,0)?, "iterator")?,len)?;
            context.list_get(this,index).ok_or_else(|| RuntimeError::new("list iterator out of bounds"))
        }
        ListFilled => {
            let count = integer(argument(args,0)?, "count")?;
            if count < 0 || count > 16_777_216 { return Err(RuntimeError::new("invalid List.filled size")); }
            Ok(context.allocate_list(vec![argument(args,1)?; count as usize]))
        }
        ListNew => {
            let count = integer(argument(args,0)?, "Array size")?;
            if count < 0 || count > 16_777_216 {
                return Err(RuntimeError::new("invalid Array size"));
            }
            // Napalm has no user-visible `undefined`; fresh sparse Array slots read
            // as null until written.
            Ok(context.allocate_list(vec![Value::NULL; count as usize]))
        }
        ListNewNumber => {
            let count = integer(argument(args,0)?, "Array<Number> size")?;
            if count < 0 || count > 16_777_216 {
                return Err(RuntimeError::new("invalid Array<Number> size"));
            }
            context.allocate_typed_list(
                vec![Value::from_num(0.0); count as usize],
                Type::Num,
            )
        }

        MapCount => Ok(Value::from_num(context.map_len(this).ok_or_else(|| RuntimeError::new("receiver is not Map"))? as f64)),
        MapContainsKey => Ok(Value::from_bool(context.map_contains_key(this, argument(args,0)?)?)),
        MapRemove => context.map_remove(this, argument(args,0)?),
        MapClear => { context.map_clear(this)?; Ok(Value::NULL) }
        MapIterate | MapKeysIterate => {
            let map = if matches!(op, MapKeysIterate) { context.map_keys_source(this)? } else { this };
            let len = context.map_len(map).ok_or_else(|| RuntimeError::new("receiver is not Map"))?;
            let next = if argument(args,0)?.is_null() { 0 } else { integer(argument(args,0)?, "iterator")? + 1 };
            Ok(if next >= 0 && (next as usize) < len { Value::from_num(next as f64) } else { Value::FALSE })
        }
        MapIteratorValue => context.map_nth_entry(this, integer(argument(args,0)?, "iterator")? as usize),
        MapKeys => context.map_keys(this),
        MapKeysCount => {
            let map = context.map_keys_source(this)?;
            Ok(Value::from_num(context.map_len(map).ok_or_else(|| RuntimeError::new("receiver is not Map"))? as f64))
        }
        MapKeysIteratorValue => {
            let map = context.map_keys_source(this)?;
            context.map_nth_key(map, integer(argument(args,0)?, "iterator")? as usize)
        }
        MapEntryKey => Ok(context.map_entry_parts(this)?.0),
        MapEntryValue => Ok(context.map_entry_parts(this)?.1),

        RangeFrom => Ok(Value::from_num(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.0)),
        RangeTo => Ok(Value::from_num(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.1)),
        RangeMin => { let (a,b,_) = context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?; Ok(Value::from_num(a.min(b))) }
        RangeMax => { let (a,b,_) = context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?; Ok(Value::from_num(a.max(b))) }
        RangeInclusive => Ok(Value::from_bool(context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?.2)),
        RangeIterate => {
            let (from,to,inclusive)=context.range_parts(this).ok_or_else(|| RuntimeError::new("receiver is not Range"))?;
            let current = if argument(args,0)?.is_null() { from } else { num(argument(args,0)?, "iterator")? + if to >= from {1.0}else{-1.0} };
            let valid = if to >= from { if inclusive { current <= to } else { current < to } } else if inclusive { current >= to } else { current > to };
            Ok(if valid { Value::from_num(current) } else { Value::FALSE })
        }
        RangeIteratorValue => Ok(argument(args,0)?),

        SystemPrint => {
            if args.len() == 1 {
                println!();
                Ok(Value::NULL)
            } else {
                let value = argument(args,0)?;
                println!("{}", context.format_value(value));
                Ok(value)
            }
        }
        SystemWrite => {
            print!("{}", context.format_value(argument(args,0)?));
            io::stdout().flush().map_err(|e| RuntimeError::new(format!("stdout write failed: {e}")))?;
            Ok(argument(args,0)?)
        }
        SystemClock => Ok(Value::from_num(started.elapsed().as_secs_f64())),

        DateNew => {
            let supplied = &args[1..];
            let millis = match supplied {
                [] => current_epoch_millis(),
                [value] => {
                    if let Some(millis) = context.date_millis(*value) {
                        millis
                    } else if let Some(text) = context.string_utf8(*value) {
                        date_parse_string(text)
                    } else {
                        date_number(context, *value).map_or(f64::NAN, time_clip)
                    }
                }
                values => {
                    let Some(values) = date_component_args(context, values) else {
                        return Ok(context.allocate_date(f64::NAN));
                    };
                    date_make_local_components(
                        values[0],
                        values[1],
                        values.get(2).copied().unwrap_or(1.0),
                        values.get(3).copied().unwrap_or(0.0),
                        values.get(4).copied().unwrap_or(0.0),
                        values.get(5).copied().unwrap_or(0.0),
                        values.get(6).copied().unwrap_or(0.0),
                        true,
                    )
                }
            };
            Ok(context.allocate_date(time_clip(millis)))
        }
        DateUtc => {
            let supplied = &args[1..];
            if supplied.len() < 2 {
                return Ok(Value::from_num(f64::NAN));
            }
            let Some(values) = date_component_args(context, supplied) else {
                return Ok(Value::from_num(f64::NAN));
            };
            Ok(Value::from_num(date_make_utc_components(
                values[0],
                values[1],
                values.get(2).copied().unwrap_or(1.0),
                values.get(3).copied().unwrap_or(0.0),
                values.get(4).copied().unwrap_or(0.0),
                values.get(5).copied().unwrap_or(0.0),
                values.get(6).copied().unwrap_or(0.0),
                true,
            )))
        }
        DateParse => {
            let value = argument(args, 0)?;
            let millis = if let Some(text) = context.string_utf8(value) {
                date_parse_string(text)
            } else {
                date_parse_string(&context.format_value(value))
            };
            Ok(Value::from_num(millis))
        }
        DateNow | DateNowMs => Ok(Value::from_num(current_epoch_millis())),
        DateSleep => {
            let ms = num(argument(args,0)?, "milliseconds")?;
            if !ms.is_finite() || ms < 0.0 { return Err(RuntimeError::new("Date.sleep expects non-negative milliseconds")); }
            std::thread::sleep(Duration::from_secs_f64(ms / 1000.0));
            Ok(Value::NULL)
        }
        DateGetTime => Ok(Value::from_num(
            context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?
        )),
        DateGet(field, utc) => {
            let millis = context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?;
            let parts = if utc {
                date_utc_parts(millis)
            } else {
                date_local_parts(millis).or_else(|| date_utc_parts(millis))
            };
            Ok(Value::from_num(parts.map_or(f64::NAN, |parts| date_field_value(parts, field))))
        }
        DateTimezoneOffset => {
            let millis = context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?;
            let offset = if !millis.is_finite() {
                f64::NAN
            } else {
                date_local_parts(millis).map_or(0.0, |parts| -(parts.offset_seconds as f64) / 60.0)
            };
            Ok(Value::from_num(offset))
        }
        DateSetTime => {
            let millis = time_clip(date_number(context, argument(args, 0)?).unwrap_or(f64::NAN));
            context.set_date_millis(this, millis)?;
            Ok(Value::from_num(millis))
        }
        DateSet(field, utc) => {
            let current = context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?;
            let millis = time_clip(date_set_components(context, current, field, utc, &args[1..]));
            context.set_date_millis(this, millis)?;
            Ok(Value::from_num(millis))
        }
        DateToString(format) => {
            let millis = context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?;
            Ok(context.allocate_string(date_format_local(millis, format)?))
        }
        DateToJson => {
            let millis = context.date_millis(this).ok_or_else(|| RuntimeError::new("receiver is not Date"))?;
            if !millis.is_finite() {
                Ok(Value::NULL)
            } else {
                Ok(context.allocate_string(date_format_iso(millis)?))
            }
        }
    }
}

#[cfg(test)]
mod date_compat_tests {
    use super::*;

    #[test]
    fn utc_construction_rolls_fields_and_preserves_pre_epoch_values() {
        assert_eq!(
            date_make_utc_components(2024.0, 1.0, 29.0, 13.0, 45.0, 30.0, 250.0, true),
            1_709_214_330_250.0,
        );
        assert_eq!(
            date_make_utc_components(2023.0, 13.0, 35.0, 0.0, 0.0, 0.0, 0.0, true),
            1_709_683_200_000.0,
        );
        assert_eq!(
            date_make_utc_components(1969.0, 11.0, 31.0, 23.0, 59.0, 59.0, 999.0, true),
            -1.0,
        );
    }

    #[test]
    fn iso_format_supports_ecmascript_extended_years() {
        let negative = date_make_utc_components(-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, true);
        assert_eq!(date_format_iso(negative).unwrap(), "-000001-01-01T00:00:00.000Z");
        assert_eq!(
            date_format_iso(DATE_MAX_TIME_MS).unwrap(),
            "+275760-09-13T00:00:00.000Z",
        );
    }

    #[test]
    fn parser_handles_iso_date_only_offsets_and_http_dates() {
        assert_eq!(date_parse_string("2024-03-05T10:20:30.123Z"), 1_709_634_030_123.0);
        assert_eq!(date_parse_string("2024-03-05"), 1_709_596_800_000.0);
        assert_eq!(date_parse_string("2024-03-05T12:20:30+02:00"), 1_709_634_030_000.0);
        assert_eq!(date_parse_string("Tue, 05 Mar 2024 10:20:30 GMT"), 1_709_634_030_000.0);
        assert!(date_parse_string("not a date").is_nan());
    }
}

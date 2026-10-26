//! Baseline handlers for destructuring and spread. Values are stable heap handles;
//! every allocating instruction publishes its stack map before touching the heap.
use super::*;

/// List mutation body shared by baseline bytecode and JIT runtime fast paths.
/// The caller owns any required safepoint/GC boundary.
pub(super) fn list_append_value(
    vm: &mut Interpreter<'_>,
    list: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    vm.heap.remember_if_reference(list, value);
    let resized = match vm.heap.get_mut(list) {
        Some(HeapObject::List(values)) => {
            let old_capacity = values.capacity();
            values.push(value);
            (values.capacity() != old_capacity).then_some(values.capacity())
        }
        _ => return Err(RuntimeError::new("ListAppend operand is not list")),
    };
    if let Some(capacity) = resized {
        vm.heap
            .set_external_bytes(list, capacity * std::mem::size_of::<Value>());
    }
    vm.heap.refresh_jit_view(list);
    Ok(())
}

pub(super) fn list_insert_value(
    vm: &mut Interpreter<'_>,
    list: Value,
    index: i32,
    value: Value,
) -> Result<(), RuntimeError> {
    vm.heap.remember_if_reference(list, value);
    let resized = match vm.heap.get_mut(list) {
        Some(HeapObject::List(values)) => {
            let mut index = index as i64;
            if index < 0 {
                index = values.len() as i64 + index + 1;
            }
            if index < 0 || index > values.len() as i64 {
                return Err(RuntimeError::new("list insertion index out of bounds"));
            }
            let old_capacity = values.capacity();
            values.insert(index as usize, value);
            (values.capacity() != old_capacity).then_some(values.capacity())
        }
        _ => return Err(RuntimeError::new("ListInsert operand is not list")),
    };
    if let Some(capacity) = resized {
        vm.heap
            .set_external_bytes(list, capacity * std::mem::size_of::<Value>());
    }
    vm.heap.refresh_jit_view(list);
    Ok(())
}

/// Specialized List -> List spread body shared by baseline bytecode and the JIT
/// runtime fast path. The caller owns the safepoint/GC boundary. Self-spread
/// snapshots the source before growing the destination.
pub(super) fn list_spread_list(
    vm: &mut Interpreter<'_>,
    list: Value,
    source: Value,
) -> Result<(), RuntimeError> {
    // Self-spread must snapshot the source before growing the destination. For
    // distinct lists, borrow the immutable source buffer by raw pointer so the
    // hot path avoids allocating and copying a temporary Vec. No GC or reentrant
    // runtime call occurs between taking this pointer and consuming the slice.
    if list == source {
        let values = match vm.heap.get(source) {
            Some(HeapObject::List(values)) => values.clone(),
            _ => return Err(RuntimeError::new("spread source is not a List")),
        };
        if values.iter().any(|value| value.is_heap()) {
            vm.heap.remember_owner(list);
        }
        let capacity = match vm.heap.get_mut(list) {
            Some(HeapObject::List(destination)) => {
                destination.extend_from_slice(&values);
                destination.capacity()
            }
            _ => return Err(RuntimeError::new("spread destination must be a List")),
        };
        vm.heap
            .set_external_bytes(list, capacity * std::mem::size_of::<Value>());
        vm.heap.refresh_jit_view(list);
        return Ok(());
    }

    let (source_ptr, source_len, contains_reference) = match vm.heap.get(source) {
        Some(HeapObject::List(values)) => (
            values.as_ptr(),
            values.len(),
            values.iter().any(|value| value.is_heap()),
        ),
        _ => return Err(RuntimeError::new("spread source is not a List")),
    };

    if contains_reference {
        // One owner barrier is sufficient for all copied references and is much
        // cheaper than calling the write barrier for every spread element.
        vm.heap.remember_owner(list);
    }

    let capacity = match vm.heap.get_mut(list) {
        Some(HeapObject::List(destination)) => {
            // SAFETY: source and destination are distinct stable heap handles.
            // The source Vec is not mutated while this slice is live, and this
            // helper cannot trigger GC after source_ptr is captured.
            let source_values = unsafe { std::slice::from_raw_parts(source_ptr, source_len) };
            destination.extend_from_slice(source_values);
            destination.capacity()
        }
        _ => return Err(RuntimeError::new("spread destination must be a List")),
    };

    vm.heap
        .set_external_bytes(list, capacity * std::mem::size_of::<Value>());
    vm.heap.refresh_jit_view(list);
    Ok(())
}

pub(super) fn list_remove_at_value(
    vm: &mut Interpreter<'_>,
    list: Value,
    index: i32,
) -> Result<Value, RuntimeError> {
    let value = match vm.heap.get_mut(list) {
        Some(HeapObject::List(values)) => {
            let Some(index) = normalize_index(index, values.len()) else {
                return Err(RuntimeError::new("list removal index out of bounds"));
            };
            values.remove(index)
        }
        _ => return Err(RuntimeError::new("ListRemoveAt operand is not list")),
    };
    vm.heap.refresh_jit_view(list);
    Ok(value)
}

pub(super) fn map_remove_value(
    vm: &mut Interpreter<'_>,
    map: Value,
    key_value: Value,
) -> Result<Value, RuntimeError> {
    let key = vm.value_to_map_key(key_value)?;
    let (value, structural) = vm.heap.map_remove_encoded(map, &key)?;
    if structural {
        if let Some(bytes) = vm.heap.map_external_bytes(map) {
            vm.heap.set_external_bytes(map, bytes);
        }
        vm.heap.refresh_jit_view(map);
    }
    Ok(value)
}

pub(super) unsafe fn h_list_spread(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let list = Value::from_word(vm.reg(op.b));
    let source = Value::from_word(vm.reg(op.c));

    // A non-List source is intentionally not an error: the lowerer continues
    // through the generic iterator protocol when the handled flag is false.
    if !matches!(vm.heap.get(source), Some(HeapObject::List(_))) {
        vm.set_reg(op.a, Value::FALSE.word());
        vm.advance(op);
        return;
    }

    match list_spread_list(vm, list, source) {
        Ok(()) => {
            vm.set_reg(op.a, Value::TRUE.word());
            vm.advance(op);
        }
        Err(error) => vm.fail_at(op, error.message),
    }
}

fn map_entries(heap: &Heap, source: Value) -> Result<Vec<(MapKey, Value)>, RuntimeError> {
    let Some(HeapObject::Map(map)) = heap.get(source) else {
        return Err(RuntimeError::new("object spread/destructuring requires a Map"));
    };
    Ok(match &map.storage {
        NapalmMapStorage::Shaped { shape, values } => heap.map_shapes.keys(*shape)
            .iter().cloned().zip(values.iter().copied())
            .map(|(key, value)| (MapKey::String(key), value)).collect(),
        NapalmMapStorage::Dictionary(entries) => entries.iter().map(|(key, &value)| (key.clone(), value)).collect(),
    })
}

fn copy_entries(heap: &mut Heap, destination: Value, entries: Vec<(MapKey, Value)>) -> Result<(), RuntimeError> {
    for (key, value) in entries {
        heap.remember_if_reference(destination, value);
        if let MapKey::Object(handle) = &key {
            heap.remember_if_reference(destination, Value::from_heap(*handle));
        }
        heap.map_set_encoded(destination, key, value, None)?;
    }
    if let Some(bytes) = heap.map_external_bytes(destination) {
        heap.set_external_bytes(destination, bytes);
    }
    heap.refresh_jit_view(destination);
    Ok(())
}

/// Specialized Map -> Map spread body shared by baseline bytecode and the JIT
/// runtime fast path. The caller owns the safepoint/GC boundary.
pub(super) fn map_spread_map(
    vm: &mut Interpreter<'_>,
    destination: Value,
    source: Value,
) -> Result<(), RuntimeError> {
    if source.is_null() {
        return Ok(());
    }
    let entries = map_entries(&vm.heap, source)?;
    copy_entries(&mut vm.heap, destination, entries)
}

pub(super) unsafe fn h_map_spread(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let destination = Value::from_word(vm.reg(op.a));
    let source = Value::from_word(vm.reg(op.b));
    match map_spread_map(vm, destination, source) {
        Ok(()) => vm.advance(op),
        Err(error) => vm.fail_at(op, error.message),
    }
}

pub(super) unsafe fn h_destructure_check(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    let source = Value::from_word(vm.reg(op.b));
    let valid = match vm.heap.get(source) {
        Some(HeapObject::List(_)) => op.c == 1,
        Some(HeapObject::Map(_)) => op.c == 0,
        _ => false,
    };
    if valid {
        vm.set_reg(op.a, source.word());
        vm.advance(op);
    } else {
        vm.fail_at(op, if op.c == 1 { "array destructuring requires a List" }
            else { "object destructuring requires a Map" });
    }
}

fn array_index(key: Value) -> Result<usize, RuntimeError> {
    match key.as_num() {
        Some(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 && n < usize::MAX as f64 => Ok(n as usize),
        _ => Err(RuntimeError::new("destructuring index must be a non-negative integer")),
    }
}

pub(super) unsafe fn h_destructure_get(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    let source = Value::from_word(vm.reg(op.b));
    let key = Value::from_word(vm.reg(op.c));
    let result = match vm.heap.get(source) {
        Some(HeapObject::List(values)) => array_index(key).map(|i| values.get(i).copied().unwrap_or(Value::NULL)),
        Some(HeapObject::Map(_)) => vm.heap.map_key(key).and_then(|key| vm.heap.map_get_encoded(source, &key)),
        _ => Err(RuntimeError::new("destructuring requires a List or Map")),
    };
    match result {
        Ok(value) => { vm.set_reg(op.a, value.word()); vm.advance(op); }
        Err(error) => vm.fail_at(op, error.message),
    }
}

pub(super) fn destructure_rest_list(
    vm: &mut Interpreter<'_>,
    source: Value,
    key: Value,
) -> Result<Value, RuntimeError> {
    let values = match vm.heap.get(source) {
        Some(HeapObject::List(values)) => values,
        _ => return Err(RuntimeError::new("array destructuring requires a List")),
    };
    let start = array_index(key)?.min(values.len());
    let rest = values[start..].to_vec();
    Ok(vm.heap.alloc(HeapObject::List(rest)))
}

pub(super) fn destructure_rest_map(
    vm: &mut Interpreter<'_>,
    source: Value,
    key: Value,
) -> Result<Value, RuntimeError> {
    let Some(HeapObject::List(excluded)) = vm.heap.get(key) else {
        return Err(RuntimeError::new("object rest exclusions must be a List"));
    };
    let excluded: HashSet<MapKey> = excluded
        .iter()
        .map(|&value| vm.heap.map_key(value))
        .collect::<Result<_, _>>()?;
    let entries = map_entries(&vm.heap, source)?
        .into_iter()
        .filter(|(map_key, _)| !excluded.contains(map_key))
        .collect();
    let destination = vm.heap.alloc(HeapObject::Map(NapalmMap::default()));
    copy_entries(&mut vm.heap, destination, entries)?;
    Ok(destination)
}

pub(super) unsafe fn h_destructure_rest(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let source = Value::from_word(vm.reg(op.b));
    let key = Value::from_word(vm.reg(op.c));
    let result = if matches!(vm.heap.get(source), Some(HeapObject::List(_))) {
        destructure_rest_list(vm, source, key)
    } else {
        // Preserve the original object-rest validation order: exclusions are
        // checked before the source is required to be a Map.
        destructure_rest_map(vm, source, key)
    };
    match result {
        Ok(value) => { vm.set_reg(op.a, value.word()); vm.advance(op); }
        Err(error) => vm.fail_at(op, error.message),
    }
}

pub(super) unsafe fn h_invoke_spread(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let mut op = *ptr;
    let receiver = vm.reg(op.b);
    let arguments = Value::from_word(vm.reg(op.c));
    let mut words = [RegisterWord(0); ARGUMENT_REGISTER_COUNT];
    let count = match vm.heap.get(arguments) {
        Some(HeapObject::List(values)) if values.len() <= ARGUMENT_REGISTER_COUNT => {
            for (slot, value) in words.iter_mut().zip(values) { *slot = value.word(); }
            values.len()
        }
        Some(HeapObject::List(_)) => { vm.fail_at(&op, "spread calls support at most 16 arguments"); return; }
        _ => { vm.fail_at(&op, "spread call arguments must be a List"); return; }
    };
    let signature = vm.program.metadata.methods[op.x as usize];
    let method = vm.program.metadata.methods.iter().position(|candidate|
        candidate.name == signature.name && candidate.kind == signature.kind && candidate.arity as usize == count);
    let Some(method) = method else { vm.fail_at(&op, "spread method arity is unavailable"); return; };
    let home = if op.y == u32::MAX { None } else { vm.prepared.class_by_id.get(&ClassId(op.y)).copied() };
    let target = vm.receiver_class(Value::from_word(receiver))
        .and_then(|class| vm.lookup_method(class, MethodSymbol(method as u32), home)
            .ok_or_else(|| RuntimeError::new(format!("spread method with {count} arguments is not defined for receiver"))));
    // Native builtins use the logical arity, not the argument-list register.
    op.c = count as u8;
    match target {
        Ok(target) => vm.invoke_target(&op, target, receiver, &words[..count]),
        Err(error) => vm.fail_at(&op, error.message),
    }
}

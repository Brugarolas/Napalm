//! Baseline handlers for destructuring and spread. Values are stable heap handles;
//! every allocating instruction publishes its stack map before touching the heap.
use super::*;

pub(super) unsafe fn h_list_spread(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let list = Value::from_word(vm.reg(op.b));
    let source = Value::from_word(vm.reg(op.c));
    let values = match vm.heap.get(source) {
        Some(HeapObject::List(values)) => values.clone(),
        _ => {
            vm.set_reg(op.a, Value::FALSE.word());
            vm.advance(op);
            return;
        }
    };
    for &value in &values { vm.heap.remember_if_reference(list, value); }
    let capacity = match vm.heap.get_mut(list) {
        Some(HeapObject::List(destination)) => {
            destination.extend(values);
            destination.capacity()
        }
        _ => { vm.fail_at(op, "spread destination must be a List"); return; }
    };
    vm.heap.set_external_bytes(list, capacity * std::mem::size_of::<Value>());
    vm.heap.refresh_jit_view(list);
    vm.set_reg(op.a, Value::TRUE.word());
    vm.advance(op);
}

fn map_entries(heap: &Heap, source: Value) -> Result<Vec<(MapKey, Value)>, RuntimeError> {
    let Some(HeapObject::Map(map)) = heap.get(source) else {
        return Err(RuntimeError::new("object spread/destructuring requires a Map"));
    };
    Ok(match &map.storage {
        NapalmMapStorage::Shaped { shape, values } => heap.map_shapes.keys(*shape).unwrap_or(&[])
            .iter().cloned().zip(values.iter().copied())
            .map(|(key, value)| (MapKey::String(MapStringKey::new(key)), value)).collect(),
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

pub(super) unsafe fn h_map_spread(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let destination = Value::from_word(vm.reg(op.a));
    let source = Value::from_word(vm.reg(op.b));
    if source.is_null() { vm.advance(op); return; }
    let result = map_entries(&vm.heap, source)
        .and_then(|entries| copy_entries(&mut vm.heap, destination, entries));
    match result {
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

pub(super) unsafe fn h_destructure_rest(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = &*ptr;
    vm.mark_gc_point(op);
    vm.maybe_collect();
    let source = Value::from_word(vm.reg(op.b));
    let key = Value::from_word(vm.reg(op.c));
    let result = (|| -> Result<Value, RuntimeError> {
        if let Some(HeapObject::List(values)) = vm.heap.get(source) {
            let start = array_index(key)?.min(values.len());
            let rest = values[start..].to_vec();
            return Ok(vm.heap.alloc(HeapObject::List(rest)));
        }
        let Some(HeapObject::List(excluded)) = vm.heap.get(key) else {
            return Err(RuntimeError::new("object rest exclusions must be a List"));
        };
        let excluded: HashSet<MapKey> = excluded.iter().map(|&v| vm.heap.map_key(v)).collect::<Result<_, _>>()?;
        let entries = map_entries(&vm.heap, source)?.into_iter()
            .filter(|(key, _)| !excluded.contains(key)).collect();
        let destination = vm.heap.alloc(HeapObject::Map(NapalmMap::default()));
        copy_entries(&mut vm.heap, destination, entries)?;
        Ok(destination)
    })();
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

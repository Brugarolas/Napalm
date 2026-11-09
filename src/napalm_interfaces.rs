//! Persistent structural contracts. Plans are validated before installation;
//! contracts contain no managed references and live in the stable handle sidecar.
//! Handle reclamation drops the sidecar vector together with the managed owner.
use super::*;

pub(super) const TYPE_ERROR: &str = "Structural type error: ";

fn mismatch(path: &str, expected: &Type) -> RuntimeError {
    RuntimeError::new(format!("{TYPE_ERROR}{path} must satisfy {expected:?}"))
}

impl Heap {
    pub(super) fn has_contract(&self, value: Value) -> bool {
        if let Some(c) = self.concurrent_table(value).and_then(MapDictionary::concurrent_contracts) { return !c.read().is_empty(); }
        value
            .heap_handle()
            .is_some_and(|handle| self.handle_table.contracts(handle).is_some())
    }

    pub(super) fn interface_matches(&self, value: Value, expected: &Type) -> bool {
        if let Some(c) = self.concurrent_table(value).and_then(MapDictionary::concurrent_contracts) { return c.read().contains(expected); }
        // A speculative guard may expose Interface-typed SSA values only when
        // a permanent contract already exists, not merely a matching shape.
        value
            .heap_handle()
            .is_some_and(|handle| self.handle_table.contract_contains(handle, expected))
    }

    /// Non-installing structural check behind `implements`/`instanceof` on
    /// plain objects: does the value currently satisfy `expected`?
    pub(super) fn structurally_matches(&self, value: Value, expected: &Type) -> bool {
        let mut plan = Vec::new();
        self.plan_contract(value, expected, "value", &mut plan).is_ok()
    }

    /// Whether instances of `class_slot` fulfil an interface/object type: the
    /// contracts declared with `implements` by the class or an ancestor must
    /// together provide every wanted field with a compatible type. Their fields
    /// are class fields typed by the contract, so the class layout keeps them
    /// valid without installing a per-object contract.
    pub(super) fn class_implements_contract(&self, class_slot: u32, contract: &Type) -> bool {
        let Some(declared) = self.class_implements.get(class_slot as usize) else {
            return false;
        };
        if declared.contains(contract) {
            return true;
        }
        let (Type::Interface(wanted) | Type::Record(wanted)) = contract else {
            return false;
        };
        wanted.iter().all(|want| {
            declared.iter().any(|have| match have {
                Type::Interface(fields) | Type::Record(fields) => fields.iter().any(|field| {
                    field.name == want.name
                        && want.ty.accepts(&field.ty)
                        && (want.optional || !field.optional)
                }),
                _ => false,
            })
        })
    }

    fn plan_contract(
        &self,
        value: Value,
        expected: &Type,
        path: &str,
        plan: &mut Vec<(Value, Type)>,
    ) -> Result<(), RuntimeError> {
        // Installed contracts are permanent invariants: all subsequent writes are
        // checked by the owning container. Rebinding the exact same structural or
        // typed-list contract therefore needs neither another deep traversal nor a
        // duplicate installation plan. This is especially valuable for nested
        // typed object graphs passed repeatedly through typed APIs.
        if matches!(
            expected,
            Type::Interface(_)
                | Type::Record(_)
                | Type::GenericInstance { .. }
                | Type::List(Some(_))
                | Type::GenericMap(_, _)
                | Type::Set(Some(_))
                | Type::Builtin(
                    BuiltinId::Event | BuiltinId::Emitter | BuiltinId::Channel | BuiltinId::Receiver | BuiltinId::ConcurrentMap | BuiltinId::ConcurrentSet | BuiltinId::Generator,
                    Some(_),
                )
        )
            && self.interface_matches(value, expected)
        {
            return Ok(());
        }

        // The finite schema bounds recursion, even when values contain cycles.
        let valid = match expected {
            Type::Top => true,
            Type::Null => value.is_null(),
            Type::Bool => value.is_bool(),
            Type::Num => value.is_num(),
            Type::String => matches!(self.get(value), Some(HeapObject::String(_))),
            Type::Range => matches!(self.get(value), Some(HeapObject::Range { .. })),
            Type::Fiber => matches!(self.get(value), Some(HeapObject::FiberObject { .. })),
            Type::Instance(classes) => match self.get(value) {
                Some(HeapObject::Instance(InstanceObject { class_slot, .. })) | Some(HeapObject::Foreign { class_slot, .. }) => {
                    classes.is_unknown() || self.contract_class_ids.get(*class_slot as usize)
                        .is_some_and(|id| classes.contains(*id))
                }
                _ => false,
            },
            Type::Map => matches!(self.get(value), Some(HeapObject::Map(_))),
            Type::GenericMap(key_type, value_type) => {
                let Some(HeapObject::Map(map)) = self.get(value) else { return Err(mismatch(path, expected)); };
                for i in 0..map.len() {
                    self.plan_map_key(&map.nth_key(&self.map_shapes, i).unwrap(), key_type, plan)?;
                    self.plan_contract(map.nth_value(i).unwrap(), value_type, "map value", plan)?;
                }
                plan.push((value, expected.clone()));
                true
            }
            Type::Set(element) => {
                let Some(HeapObject::Set { map }) = self.get(value) else { return Err(mismatch(path, expected)); };
                if let Some(element) = element {
                    self.plan_contract(*map, &Type::GenericMap(element.clone(), element.clone()), "set element", plan)?;
                    plan.push((value, expected.clone()));
                }
                true
            }
            Type::Closure(_) => matches!(self.get(value), Some(HeapObject::Closure { .. })),
            Type::List(element) => {
                let Some(HeapObject::List(items)) = self.get(value) else {
                    return Err(mismatch(path, expected));
                };
                if let Some(element) = element {
                    for (i, item) in items.iter().enumerate() {
                        self.plan_contract(*item, element, &format!("{path}[{i}]"), plan)?;
                    }
                    plan.push((value, expected.clone()));
                }
                true
            }
            Type::GenericInstance { class, fields, .. } => {
                let Some(HeapObject::Instance(values @ InstanceObject { class_slot, .. })) = self.get(value) else {
                    return Err(mismatch(path, expected));
                };
                if self.contract_class_ids.get(*class_slot as usize) != Some(class) {
                    return Err(mismatch(path, expected));
                }
                let names = self.class_field_names.get(*class_slot as usize);
                for field in fields {
                    let Some(slot) = names.and_then(|names| names.iter().position(|name| *name == field.name)) else {
                        continue;
                    };
                    let item = values.get(slot).copied().unwrap_or(Value::NULL);
                    self.plan_contract(item, &field.ty, &format!("{path}.{}", field.name), plan)?;
                }
                plan.push((value, expected.clone()));
                true
            }
            Type::Interface(fields) | Type::Record(fields) => {
                let map = match self.get(value) {
                    Some(HeapObject::Map(map)) => map,
                    Some(HeapObject::Instance(InstanceObject { class_slot, .. }))
                    | Some(HeapObject::Foreign { class_slot, .. })
                        if self.class_implements_contract(*class_slot, expected) =>
                    {
                        return Ok(());
                    }
                    _ => return Err(mismatch(path, expected)),
                };
                for field in fields {
                    let key = MapKey::String(MapStringKey::new(Arc::from(field.name.as_bytes())));
                    match map.get(&self.map_shapes, &key) {
                        Some(item) => self.plan_contract(
                            item,
                            &field.ty,
                            &format!("{path}.{}", field.name),
                            plan,
                        )?,
                        None if field.optional => (),
                        None => {
                            return Err(RuntimeError::new(format!(
                                "{TYPE_ERROR}missing required field {path}.{}",
                                field.name
                            )));
                        }
                    }
                }

                if matches!(expected, Type::Record(_)) {
                    for index in 0..map.len() {
                        let Some(key) = map.nth_key(&self.map_shapes, index) else {
                            return Err(RuntimeError::new(format!(
                                "{TYPE_ERROR}{path} has inconsistent Map storage"
                            )));
                        };
                        let declared = match &key {
                            MapKey::String(key) => fields
                                .iter()
                                .any(|field| field.name.as_bytes() == key.as_ref()),
                            _ => false,
                        };
                        if !declared {
                            let property = match key {
                                MapKey::String(key) => String::from_utf8_lossy(key.as_ref()).into_owned(),
                                _ => "<non-string key>".to_owned(),
                            };
                            return Err(RuntimeError::new(format!(
                                "{TYPE_ERROR}closed type cannot contain undeclared property '{property}'"
                            )));
                        }
                    }
                }

                plan.push((value, expected.clone()));
                true
            }
            Type::Builtin(BuiltinId::Generator, element) => {
                if !matches!(self.get(value), Some(HeapObject::FiberObject { generator: true, .. })) {
                    return Err(mismatch(path, expected));
                }
                if element.is_some() { plan.push((value, expected.clone())); }
                true
            }
            Type::Builtin(builtin @ (BuiltinId::ConcurrentMap | BuiltinId::ConcurrentSet), element) => {
                let Some(table) = self.concurrent_table(value) else { return Err(mismatch(path, expected)); };
                let correct = matches!((builtin, table), (BuiltinId::ConcurrentMap, MapDictionary::ConcurrentMap(_)) | (BuiltinId::ConcurrentSet, MapDictionary::ConcurrentSet(_)));
                if !correct { return Err(mismatch(path, expected)); }
                if let Some(element) = element {
                    let (owner, contract) = if *builtin == BuiltinId::ConcurrentSet {
                        let Some(HeapObject::Set { map }) = self.get(value) else { return Err(mismatch(path, expected)); };
                        (*map, Type::GenericMap(element.clone(), element.clone()))
                    } else { (value, (**element).clone()) };
                    self.plan_contract(owner, &contract, path, plan)?;
                }
                plan.push((value, expected.clone()));
                true
            }
            Type::Builtin(builtin, element) => {
                let object = self.get(value);
                let state_payload_type = match (builtin, object) {
                    (BuiltinId::Event, Some(HeapObject::Event { state }))
                    | (BuiltinId::Emitter, Some(HeapObject::Emitter { state, .. })) => {
                        unsafe { state.get() }.payload_type.as_ref()
                    }
                    (BuiltinId::Channel, Some(HeapObject::Channel { state, .. }))
                    | (BuiltinId::Receiver, Some(HeapObject::Receiver { state })) => {
                        unsafe { state.get() }.payload_type.as_ref()
                    }
                    // Never install a contract on a Promise: its handle entry reuses
                    // the contract-pointer word for the settled payload. `await`
                    // validates the settled value against `Promise<T>` instead.
                    (BuiltinId::Promise, Some(HeapObject::Promise { .. })) => return Ok(()),
                    _ => return Err(mismatch(path, expected)),
                };

                if let Some(element) = element {
                    if let Some(existing) = state_payload_type {
                        if existing != element.as_ref() {
                            return Err(RuntimeError::new(format!(
                                "{TYPE_ERROR}{path} already carries incompatible payload contract {existing:?}"
                            )));
                        }
                    }

                    // Channel/Receiver may already contain buffered values when a
                    // typed alias is installed. Validate the existing queue before
                    // publishing the persistent shared-state contract.
                    if let Some(state) = match object {
                        Some(HeapObject::Channel { state, .. })
                        | Some(HeapObject::Receiver { state }) => Some(state),
                        _ => None,
                    } {
                        for (index, item) in unsafe { state.get() }.queue.iter().copied().enumerate() {
                            self.plan_contract(item, element, &format!("channel item {index}"), plan)?;
                        }
                    }

                    plan.push((value, expected.clone()));
                }
                true
            }
            Type::Union(types) => {
                let checkpoint = plan.len();
                for ty in types {
                    if self.plan_contract(value, ty, path, plan).is_ok() {
                        return Ok(());
                    }
                    plan.truncate(checkpoint);
                }
                false
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(mismatch(path, expected))
        }
    }

    fn plan_map_key(&self, key: &MapKey, expected: &Type, plan: &mut Vec<(Value, Type)>) -> Result<(), RuntimeError> {
        // String keys are owned bytes, not GC values; all other keys can be checked
        // with the ordinary value checker without allocating temporary strings.
        if let MapKey::String(_) = key {
            fn accepts_string(ty: &Type) -> bool {
                match ty {
                    Type::Top | Type::String => true,
                    Type::Union(types) => types.iter().any(accepts_string),
                    _ => false,
                }
            }
            return if accepts_string(expected) { Ok(()) } else { Err(mismatch("map key", expected)) };
        }
        let value = match key {
            MapKey::Null => Value::NULL,
            MapKey::Bool(v) => Value::from_bool(*v),
            MapKey::Num(bits) => Value::from_num(f64::from_bits(*bits)),
            MapKey::Object(handle) => Value::from_heap(*handle),
            MapKey::String(_) => unreachable!(),
        };
        self.plan_contract(value, expected, "map key", plan)
    }

    fn install_contracts(&mut self, plan: Vec<(Value, Type)>) {
        for (value, ty) in plan {
            let Some(object) = self.object(value) else {
                continue;
            };
            let (handle, shape, already_installed) = {
                let managed = unsafe { managed_object(object) };
                // The Promise handle entry reuses the contract-pointer word for
                // its settled payload, so a contract would corrupt it.
                if matches!(managed.object, HeapObject::Promise { .. }) {
                    continue;
                }
                let installed = self.interface_matches(value, &ty);
                let shape = match &managed.object {
                    HeapObject::Map(map) => map.shaped().map(|(shape, _)| shape),
                    _ => None,
                };
                (managed.handle, shape, installed)
            };
            if already_installed {
                continue;
            }

            // Keep the distinction between "not a shaped map" and "typed shape
            // allocation failed". The latter must invalidate already-compiled
            // shaped traces, so it falls back to dictionary representation.
            let typed_shape = shape.map(|shape| {
                if matches!(ty, Type::GenericMap(_, _)) { None }
                else { self.map_shapes.apply_contract(shape, &ty) }
            });
            let object = self.object(value).expect("validated contract owner");
            if let Some(contracts) = self.concurrent_table(value).and_then(MapDictionary::concurrent_contracts) {
                let mut contracts = contracts.write();
                if !contracts.contains(&ty) { contracts.push(ty); }
                continue;
            }
            let managed = unsafe { managed_object_mut(object) };
            self.handle_table.stage_contract(handle, ty.clone());

            // Event/Emitter and Channel/Receiver share their payload contract in
            // native sidecar state. This makes the contract capability-wide rather
            // than handle-local: an untyped alias cannot bypass a typed endpoint.
            if let Type::Builtin(
                BuiltinId::Event | BuiltinId::Emitter | BuiltinId::Channel | BuiltinId::Receiver | BuiltinId::ConcurrentMap | BuiltinId::ConcurrentSet | BuiltinId::Generator,
                Some(element),
            ) = &ty
            {
                match &mut managed.object {
                    HeapObject::Event { state } | HeapObject::Emitter { state, .. } => {
                        unsafe { state.get_mut() }.payload_type = Some((**element).clone());
                    }
                    HeapObject::Channel { state, .. } | HeapObject::Receiver { state } => {
                        unsafe { state.get_mut() }.payload_type = Some((**element).clone());
                    }
                    _ => {}
                }
            }

            if let HeapObject::Map(map) = &mut managed.object {
                match typed_shape {
                    Some(Some(typed_shape)) => {
                        if let NapalmMapStorage::Shaped { shape, .. } = &mut map.storage {
                            *shape = typed_shape;
                        }
                    }
                    Some(None) => map.promote_to_dictionary(&self.map_shapes),
                    None => {}
                }
            }

            // Contract installation changes both write validity and, for maps,
            // the shape id carrying immutable slot type facts. Publish both to
            // already-compiled/native tracing code before returning.
            self.handle_table.refresh(handle, object, &managed.object);
            handles()
                .entry(handle)
                .has_contract
                .store(1, Ordering::Release);
            let bytes = external_bytes_for(&managed.object);
            self.set_external_bytes(value, bytes);
        }
    }

    /// Install a typed-list contract on a freshly allocated dense Array without
    /// validating its initialization slots. The constructor itself owns those
    /// slots; subsequent writes still go through `check_list_write`.
    pub(super) fn install_fresh_list_contract(
        &mut self,
        value: Value,
        element: Type,
    ) -> Result<(), RuntimeError> {
        if !matches!(self.get(value), Some(HeapObject::List(_))) {
            return Err(RuntimeError::new("typed Array allocation did not produce a List"));
        }
        let contract = Type::List(Some(Box::new(element)));
        if !self.interface_matches(value, &contract) {
            self.install_contracts(vec![(value, contract)]);
        }
        Ok(())
    }

    pub(super) fn bind_contract(
        &mut self,
        value: Value,
        expected: &Type,
    ) -> Result<(), RuntimeError> {
        // Contracts are persistent. Typed parameters and loop-carried aliases may
        // bind the same object repeatedly; avoid rescanning every structural field
        // once this exact contract has already been validated and installed.
        if self.interface_matches(value, expected) {
            return Ok(());
        }
        let mut plan = Vec::new();
        self.plan_contract(value, expected, "object", &mut plan)?;
        self.install_contracts(plan);
        Ok(())
    }

    pub(super) fn check_shaped_map_write(
        &mut self,
        shape: u32,
        slot: u16,
        key: &MapStringKey,
        item: Value,
    ) -> Result<bool, RuntimeError> {
        let contract = self.map_shapes.slot_contract(shape, slot).clone();
        if contract.write_runtime_checked {
            return Ok(false);
        }
        if contract.readonly {
            let property = String::from_utf8_lossy(key.as_ref());
            return Err(RuntimeError::new(format!(
                "{TYPE_ERROR}field '{property}' is readonly"
            )));
        }
        let Some(expected) = contract.write_type else {
            // No write type means this is either an untyped/open property or a
            // declared `any` slot. The shape itself is enough proof that this is
            // an update rather than a closed-record property addition.
            return Ok(true);
        };

        let property = String::from_utf8_lossy(key.as_ref());
        let mut plan = Vec::new();
        self.plan_contract(item, &expected, &property, &mut plan)?;
        self.install_contracts(plan);
        Ok(true)
    }

    pub(super) fn check_map_write(
        &mut self,
        owner: Value,
        key: Option<&MapKey>,
        item: Option<Value>,
    ) -> Result<(), RuntimeError> {
        let Some(object) = self.object(owner) else {
            return Ok(());
        };
        let existing = key.and_then(|key| {
            match &unsafe { managed_object(object) }.object {
                HeapObject::Map(map) => map.get(&self.map_shapes, key),
                _ => None,
            }
        });
        let handle = unsafe { managed_object(object) }.handle;
        let contracts = self.concurrent_table(owner).and_then(MapDictionary::concurrent_contracts)
            .map(|c| c.read().clone()).unwrap_or_else(|| self.handle_table.cloned_contracts(handle));
        if contracts.is_empty() {
            return Ok(());
        }

        let mut plan = Vec::new();
        for contract in &contracts {
            if let Type::GenericMap(key_type, value_type) = contract {
                if let (Some(key), Some(item)) = (key, item) {
                    self.plan_map_key(key, key_type, &mut plan)?;
                    self.plan_contract(item, value_type, "map value", &mut plan)?;
                }
                continue;
            }
            let (fields, closed) = match contract {
                Type::Interface(fields) => (fields, false),
                Type::Record(fields) => (fields, true),
                _ => continue,
            };
            let mut matched = false;
            for field in fields {
                let applies = match key {
                    None => true,
                    Some(MapKey::String(k)) => k.as_ref() == field.name.as_bytes(),
                    _ => false,
                };
                if !applies {
                    continue;
                }
                matched = true;
                if field.readonly {
                    return Err(RuntimeError::new(format!(
                        "{TYPE_ERROR}field '{}' is readonly",
                        field.name
                    )));
                }
                if let Some(item) = item {
                    self.plan_contract(item, &field.ty, &field.name, &mut plan)?;
                } else if !field.optional {
                    return Err(RuntimeError::new(format!(
                        "{TYPE_ERROR}cannot remove required field '{}'",
                        field.name
                    )));
                }
            }

            if closed && key.is_some() && !matched && existing.is_none() && item.is_some() {
                let property = match key {
                    Some(MapKey::String(key)) => String::from_utf8_lossy(key.as_ref()).into_owned(),
                    _ => "<non-string key>".to_owned(),
                };
                return Err(RuntimeError::new(format!(
                    "{TYPE_ERROR}cannot add property '{property}' to a closed type"
                )));
            }
        }
        self.install_contracts(plan);
        Ok(())
    }

    pub(super) fn check_list_write(
        &mut self,
        owner: Value,
        item: Value,
    ) -> Result<(), RuntimeError> {
        let Some(object) = self.object(owner) else {
            return Ok(());
        };
        let mut plan = Vec::new();
        let handle = unsafe { managed_object(object) }.handle;
        let Some(contracts) = self.handle_table.contracts(handle) else {
            return Ok(());
        };
        for contract in contracts.iter() {
            if let Type::List(Some(element)) = contract {
                self.plan_contract(item, element, "list element", &mut plan)?;
            }
        }
        self.install_contracts(plan);
        Ok(())
    }
}

pub(super) unsafe fn h_bind_interface(vm: &mut Interpreter, ptr: *const PreparedOp) {
    let op = unsafe { &*ptr };
    let expected = vm.program.functions[vm.current_function_index()]
        .chunk
        .type_guards[op.x as usize]
        .expected
        .clone();
    let value = Value::from_word(unsafe { vm.reg(op.b) });
    match vm.heap.bind_contract(value, &expected) {
        Ok(()) => unsafe {
            vm.set_reg(op.a, value.word());
            vm.advance(op)
        },
        Err(error) => vm.fail_at(op, error.message),
    }
}

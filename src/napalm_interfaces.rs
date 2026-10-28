//! Persistent structural contracts. Plans are validated before installation;
//! contracts contain no managed references and move/die with their owner in GC.
use super::*;

pub(super) const TYPE_ERROR: &str = "Interface type error: ";

fn mismatch(path: &str, expected: &Type) -> RuntimeError {
    RuntimeError::new(format!("{TYPE_ERROR}{path} must satisfy {expected:?}"))
}

impl Heap {
    pub(super) fn has_contract(&self, value: Value) -> bool {
        self.object(value)
            .is_some_and(|o| unsafe { managed_object(o) }.contracts.is_some())
    }

    pub(super) fn interface_matches(&self, value: Value, expected: &Type) -> bool {
        // A speculative guard may expose Interface-typed SSA values only when
        // a permanent contract already exists, not merely a matching shape.
        self.object(value).is_some_and(|object| {
            unsafe { managed_object(object) }
                .contracts
                .as_ref()
                .is_some_and(|contracts| contracts.contains(expected))
        })
    }

    fn plan_contract(
        &self,
        value: Value,
        expected: &Type,
        path: &str,
        plan: &mut Vec<(Value, Type)>,
    ) -> Result<(), RuntimeError> {
        // The finite schema bounds recursion, even when values contain cycles.
        let valid = match expected {
            Type::Top => true,
            Type::Null => value.is_null(),
            Type::Bool => value.is_bool(),
            Type::Num => value.is_num(),
            Type::String => matches!(self.get(value), Some(HeapObject::String(_))),
            Type::Map => matches!(self.get(value), Some(HeapObject::Map(_))),
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
            Type::Interface(fields) | Type::Record(fields) => {
                let Some(HeapObject::Map(map)) = self.get(value) else {
                    return Err(mismatch(path, expected));
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
                            continue;
                        };
                        let declared = match &key {
                            MapKey::String(key) => fields
                                .iter()
                                .any(|field| field.name.as_bytes() == key.as_ref()),
                            _ => false,
                        };
                        if !declared {
                            let property = match key {
                                MapKey::String(key) => {
                                    String::from_utf8_lossy(key.as_ref()).into_owned()
                                }
                                _ => "<non-string key>".to_owned(),
                            };
                            return Err(RuntimeError::new(format!(
                                "{TYPE_ERROR}property '{property}' is not declared by the closed type"
                            )));
                        }
                    }
                }

                plan.push((value, expected.clone()));
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

    fn install_contracts(&mut self, plan: Vec<(Value, Type)>) {
        for (value, ty) in plan {
            let Some(object) = self.object(value) else {
                continue;
            };
            let (handle, shape, already_installed) = {
                let managed = unsafe { managed_object(object) };
                let installed = managed
                    .contracts
                    .as_ref()
                    .is_some_and(|contracts| contracts.contains(&ty));
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
            let typed_shape = shape.map(|shape| self.map_shapes.apply_contract(shape, &ty));
            let object = self.object(value).expect("validated contract owner");
            let managed = unsafe { managed_object_mut(object) };
            managed
                .contracts
                .get_or_insert_with(|| Box::new(Vec::new()))
                .push(ty);
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
            handles()
                .entry(handle)
                .has_contract
                .store(1, Ordering::Relaxed);
            self.handle_table.refresh(handle, &managed.object);
        }
    }

    pub(super) fn bind_contract(
        &mut self,
        value: Value,
        expected: &Type,
    ) -> Result<(), RuntimeError> {
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
        let contracts = unsafe { managed_object(object) }
            .contracts
            .as_ref()
            .map(|contracts| contracts.as_ref().clone())
            .unwrap_or_default();
        if contracts.is_empty() {
            return Ok(());
        }

        let mut plan = Vec::new();
        for contract in &contracts {
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
        let Some(contracts) = &unsafe { managed_object(object) }.contracts else {
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

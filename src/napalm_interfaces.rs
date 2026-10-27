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
            Type::Interface(fields) => {
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
            let object = self.object(value).expect("validated contract owner");
            let managed = unsafe { managed_object_mut(object) };
            let contracts = managed
                .contracts
                .get_or_insert_with(|| Box::new(Vec::new()));
            if !contracts.contains(&ty) {
                contracts.push(ty);
                // This stable sidecar invalidates stores in traces compiled
                // before an alias acquired its first contract. Reads stay native.
                handles()
                    .entry(managed.handle)
                    .has_contract
                    .store(1, Ordering::Relaxed);
            }
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

    pub(super) fn check_map_write(
        &mut self,
        owner: Value,
        key: Option<&MapKey>,
        item: Option<Value>,
    ) -> Result<(), RuntimeError> {
        let Some(object) = self.object(owner) else {
            return Ok(());
        };
        let mut plan = Vec::new();
        let Some(contracts) = &unsafe { managed_object(object) }.contracts else {
            return Ok(());
        };
        for contract in contracts.iter() {
            let Type::Interface(fields) = contract else {
                continue;
            };
            for field in fields {
                let applies = match key {
                    None => true, // clear
                    Some(MapKey::String(k)) => k.as_ref() == field.name.as_bytes(),
                    _ => false,
                };
                if !applies {
                    continue;
                }
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

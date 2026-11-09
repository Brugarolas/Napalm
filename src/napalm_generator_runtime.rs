//! Generator scheduling and the public iterator-result boundary.
use super::*;

impl Interpreter<'_> {
    fn generator_result(&mut self, value: Value, done: bool) -> Value {
        let value_root = self.add_root(value);
        let result = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
        let result_root = self.add_root(result);
        for (name, value) in [(b"value".as_slice(), value), (b"done".as_slice(), Value::from_bool(done))] {
            let key = MapKey::String(MapStringKey::new(Arc::from(name)));
            self.heap.map_set_encoded(result, key, value, None).expect("fresh generator result");
        }
        self.release_root(result_root);
        self.release_root(value_root);
        result
    }

    fn generator_store_result(&mut self, link: FiberLink, value: Value, done: bool) {
        let iterate = link.generator.unwrap().1;
        let result = if iterate && done { Value::FALSE } else { self.generator_result(value, done) };
        let caller = self.fiber_mut(link.caller);
        if let Some(frame) = caller.frames.last() {
            caller.registers[frame.reg_base as usize + link.dst.index()] = result.word();
        }
        self.refresh_hot_frame();
    }

    fn generator_raise_at(&mut self, raw_pc: u32, error: Value) {
        if self.unwind_to_structured_handler_at(raw_pc, Some(error), None) { return; }
        let function = self.current_prepared();
        let index = function.raw_pcs.iter().position(|pc| *pc == raw_pc).unwrap_or(self.hot_pc as usize);
        let op = &function.ops[index] as *const PreparedOp;
        self.abort_current_fiber(unsafe { &*op }, error);
    }

    pub(super) fn deliver_generator_completion(&mut self, link: FiberLink, completion: Value) {
        let (mode, value) = match self.heap.get(completion) {
            Some(HeapObject::List(items)) if items.len() == 2 => (items[0].as_num(), items[1]),
            _ => { self.fail_terminal(RuntimeError::new("invalid generator completion")); return; }
        };
        if mode == Some(2.0) {
            self.generator_raise_at(link.generator.unwrap().0, value);
        } else {
            self.generator_store_result(link, value, true);
        }
    }

    pub(super) fn deliver_generator_yield(&mut self, link: FiberLink, value: Value, generator: u32) {
        let object = self.fiber(generator).object;
        let contracts: Vec<Type> = object.heap_handle()
            .and_then(|handle| self.heap.handle_table.contracts(handle))
            .map(|contracts| contracts.iter().filter_map(|ty| match ty {
                Type::Builtin(BuiltinId::Generator, Some(element)) => Some(*element.clone()),
                _ => None,
            }).collect()).unwrap_or_default();
        for ty in contracts {
            if let Err(error) = self.heap.bind_contract(value, &ty) {
                self.fail_terminal(error);
                return;
            }
        }
        self.generator_store_result(link, value, false);
    }

    /// IteratorClose for `for..of` exits. Non-generators and finished or
    /// never-started generators complete without a GC point or allocation;
    /// only a suspended generator is resumed with a return completion.
    pub(super) fn close_generator(&mut self, op: &PreparedOp, object: Value) {
        let target = match self.heap.get(object) {
            Some(HeapObject::FiberObject { fiber, generator: true, .. }) => Some(*fiber),
            _ => None,
        };
        if let Some(target) = target {
            match self.fiber(target).status {
                FiberStatus::Done | FiberStatus::Failed => {}
                FiberStatus::New => {
                    let state = self.fiber_mut(target);
                    state.status = FiberStatus::Done;
                    state.starter = None;
                }
                _ => return self.resume_generator(op, object, Value::NULL, 1, false),
            }
        }
        unsafe { self.set_reg(op.a, Value::NULL.word()); self.advance(op); }
    }

    pub(super) fn resume_generator(&mut self, op: &PreparedOp, object: Value, argument: Value, mode: u8, iterate: bool) {
        let target = match self.heap.get(object) {
            Some(HeapObject::FiberObject { fiber, generator: true, .. }) => *fiber,
            _ => { self.fail_at(op, "receiver is not a Generator"); return; }
        };
        let status = self.fiber(target).status;
        if status == FiberStatus::Running || self.fiber(target).link.is_some() {
            self.fail_at(op, "generator is already running");
            return;
        }
        let raw_pc = self.raw_pc_for_op(op);
        self.mark_gc_point(op);
        self.maybe_collect();
        if matches!(status, FiberStatus::Done | FiberStatus::Failed) || (status == FiberStatus::New && mode != 0) {
            let state = self.fiber_mut(target);
            state.status = FiberStatus::Done;
            state.starter = None;
            if mode == 2 {
                if !self.raise_value_exception(op, argument) { self.abort_current_fiber(op, argument); }
            } else {
                let value = if mode == 1 { argument } else { Value::NULL };
                let result = if iterate { Value::FALSE } else { self.generator_result(value, true) };
                unsafe { self.set_reg(op.a, result.word()); self.advance(op); }
            }
            return;
        }
        // The first .next argument is ignored. Subsequent resume packets carry
        // normal/return/throw completions back to the yield expression.
        let argument = if iterate || status == FiberStatus::New { Value::NULL } else { argument };
        let packet = self.heap.alloc(HeapObject::List(vec![Value::from_num(mode as f64), argument]));
        let root = self.add_root(packet);
        self.activate_fiber(op, object, packet, true, false);
        if self.current_fiber == target {
            self.fiber_mut(target).link.as_mut().unwrap().generator = Some((raw_pc, iterate));
        }
        self.release_root(root);
    }
}

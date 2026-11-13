use super::*;
use crate::napalm_build_pipeline::BuildCompilationPipeline;
use crate::napalm_ir::{verify_function, Builder};

fn lower(source: &str) -> (Program, FuncId) {
    let parsed = crate::napalm_front_main::parse("aot.nap", source);
    assert!(!parsed.has_errors(), "{:#?}", parsed.diagnostics);
    let mut hir = Program::new();
    crate::napalm_builtin::install_hir_builtins(&mut hir);
    let output = crate::napalm_lowerer::lower_module(&mut hir, "aot.nap", source, "aot", &parsed.module);
    assert!(!output.has_errors(), "{:#?}", output.diagnostics);
    let entry = output.lowered.unwrap().entry;
    (hir, entry)
}

fn build(source: &str) -> (Program, FuncId) {
    let (mut hir, entry) = lower(source);
    BuildCompilationPipeline::default().optimize_hir(&mut hir).unwrap();
    (hir, entry)
}

/// Source functions of the test module (the lowerer names them `<closure@N>`).
fn closures(program: &Program, entry: FuncId) -> Vec<&Function> {
    let module = program.function(entry).unwrap().module;
    program.functions.iter().filter(|f| f.module == module && f.kind == FunctionKind::Closure).collect()
}

fn kinds(function: &Function) -> impl Iterator<Item = &Inst> {
    function.blocks.iter().flat_map(|b| &b.insts).map(|id| &function.nodes[id.0 as usize].kind)
}

fn has_constant(function: &Function, expected: f64) -> bool {
    kinds(function).any(|kind| matches!(kind, Inst::Const(Constant::Num(v)) if *v == expected))
}

#[test]
fn partial_evaluation_folds_recursive_calls_and_tree_shaking_drops_the_callee() {
    let (hir, entry) = build(r#"
        function fact(n: number): number {
            if (n <= 1) { return 1; }
            return n * fact(n - 1);
        }
        function sum(n: number, acc: number): number {
            if (n == 0) { return acc; }
            return sum(n - 1, acc + n);
        }
        const result = [fact(10), sum(100, 0)];
    "#);
    let body = hir.function(entry).unwrap();
    assert!(has_constant(body, 3_628_800.0) && has_constant(body, 5050.0));
    assert!(closures(&hir, entry).is_empty(), "folded callees stay reachable");
}

#[test]
fn whole_cfg_inlining_replaces_calls_to_branchy_helpers() {
    let (hir, entry) = build(r#"
        function clamp(x: number, lo: number, hi: number): number {
            if (x < lo) { return lo; }
            if (x > hi) { return hi; }
            return x;
        }
        let total = 0;
        for (let i = 0; i < 10; i++) { total = total + clamp(i, 2, 7); }
        const result = total;
    "#);
    assert!(closures(&hir, entry).is_empty(), "clamp should be inlined and then shaken out");
    for function in &hir.functions { verify_function(function).unwrap(); }
}

#[test]
fn inlining_never_unrolls_recursive_callees() {
    let (hir, entry) = build(r#"
        function countdown(n: number): number {
            if (n <= 0) { return 0; }
            return 1 + countdown(n - 1);
        }
        let n = 0;
        for (let i = 0; i < 3; i++) { n = n + countdown(i * 1000); }
        const result = n;
    "#);
    let functions = closures(&hir, entry);
    assert_eq!(functions.len(), 1, "the recursive callee stays a single function");
    let countdown = functions[0].id;
    let calls = |function: &Function| kinds(function)
        .filter(|kind| matches!(kind, Inst::CallDirect { callee, .. } if *callee == countdown)).count();
    assert_eq!(calls(hir.function(entry).unwrap()), 1);
    assert!(hir.functions.iter().all(|f| f.specialization.is_none()));
}

#[test]
fn unprofitable_specialization_leaves_the_fixed_point_stable() {
    // Constant loop bounds fold nothing: no clone may be made, and discarding
    // the candidate must not touch snapshot-visible state (IDs, sites).
    let (mut hir, _) = lower(r#"
        class Bench {
            static step(x): Number { return x * 1.5 + 0.25; }
            static run(count): Number {
                let total: Number = 0;
                let i: Number = 0;
                while (i < count) { total = total + this.step(i); i = i + 1; }
                return total;
            }
        }
        Bench.run(1000);
        const result = Bench.run(1000000);
    "#);
    let mut optimizer = crate::napalm_build_pipeline::optimization_passes();
    let snapshot = |hir: &Program| serde_json::to_vec(hir).unwrap();
    // A test watchdog, not a production iteration limit.
    let stable = (0..64).any(|_| {
        let before = snapshot(&hir);
        optimizer.optimize_program(&mut hir);
        before == snapshot(&hir)
    });
    assert!(stable, "build passes failed to converge");
    assert!(hir.functions.iter().all(|f| f.specialization.is_none()));
}

#[test]
fn auto_memoization_survives_every_build_round() {
    // Round 1 classifies createTree as RecursiveShared; literal hoisting and
    // inlining then reshape the program so the whole-program proof no longer
    // succeeds. Memoization has priority and must be kept to the fixed point.
    let (hir, entry) = build(r#"
        function checksum(node) {
            if (node.left == null) { return 1; }
            return 1 + checksum(node.left) + checksum(node.right);
        }
        function createTree(depth) {
            if (depth > 0) {
                var d = depth - 1;
                return { left: createTree(d), right: createTree(d) };
            }
            return { left: null, right: null };
        }
        function main() {
            var sum = 0;
            var depth = 4;
            while (depth <= 10) { sum = sum + checksum(createTree(depth)); depth += 2; }
            return sum;
        }
        const result = main();
    "#);
    assert!(closures(&hir, entry).iter()
        .any(|f| f.memoization == crate::napalm_ir::MemoizationKind::RecursiveShared),
        "createTree lost its RecursiveShared memoization");
}

/// f(x: any) = check(x: Num); if (1 + 2 < 1) { return x } else { return 1 + 2 }
fn operator_function(hir: &mut Program) -> FuncId {
    let module = hir.add_module("main", 0);
    let plus = hir.intern_method("+", MethodKind::Method, 1);
    let less = hir.intern_method("<", MethodKind::Method, 1);
    let mut b = Builder::new(hir, module, "f", FunctionKind::Closure, vec![Type::Top], Type::Num);
    let param = b.param(0);
    let checked = b.emit1(Inst::BindInterface { value: param, expected: Type::Num, layout: None }, Type::Num);
    let one = b.const_num(1.0);
    let two = b.const_num(2.0);
    let site = b.fresh_site();
    let sum = b.emit1(Inst::Invoke { site, receiver: one, method: plus, args: vec![two] }, Type::Num);
    let site = b.fresh_site();
    let cond = b.emit1(Inst::Invoke { site, receiver: sum, method: less, args: vec![one] }, Type::Bool);
    let (then_block, else_block) = (b.new_block(&[]), b.new_block(&[]));
    b.branch(cond, then_block, vec![], else_block, vec![]);
    b.position_at(then_block);
    b.ret(checked);
    b.position_at(else_block);
    b.ret(sum);
    b.finish()
}

#[test]
fn sccp_folds_builtin_operators_and_branches_but_keeps_contract_checks() {
    let mut hir = Program::new();
    let id = operator_function(&mut hir);
    assert!(Sccp.optimize_program(&mut hir));
    let f = hir.function(id).unwrap();
    verify_function(f).unwrap();
    assert!(!kinds(f).any(|k| matches!(k, Inst::Invoke { .. } | Inst::Branch { .. })));
    assert!(kinds(f).any(|k| matches!(k, Inst::BindInterface { .. })), "contract checks are writes");
    assert!(has_constant(f, 3.0));
    assert!(!Sccp.optimize_program(&mut hir), "SCCP must reach its own fixed point");
}

#[test]
fn specialization_clones_keep_the_source_name_and_are_shared_by_identical_facts() {
    let mut hir = Program::new();
    let module = hir.add_module("main", 0);
    let less = hir.intern_method("<", MethodKind::Method, 1);
    // g(x) = x < 6 ? x : 0 — a constant x folds the branch away.
    let mut g = Builder::new(&mut hir, module, "g", FunctionKind::Closure, vec![Type::Num], Type::Num);
    let param = g.param(0);
    let six = g.const_num(6.0);
    let site = g.fresh_site();
    let cond = g.emit1(Inst::Invoke { site, receiver: param, method: less, args: vec![six] }, Type::Bool);
    let (small, large) = (g.new_block(&[]), g.new_block(&[]));
    g.branch(cond, small, vec![], large, vec![]);
    g.position_at(small);
    g.ret(param);
    g.position_at(large);
    let zero = g.const_num(0.0);
    g.ret(zero);
    let g = g.finish();
    // h(x) = x: a constant argument removes no work, so no clone is made.
    let mut h = Builder::new(&mut hir, module, "h", FunctionKind::Closure, vec![Type::Num], Type::Num);
    let param = h.param(0);
    h.ret(param);
    let h = h.finish();
    let mut body = Builder::new(&mut hir, module, "main", FunctionKind::ModuleBody, vec![], Type::Num);
    // NaN and infinities must not collapse into one key (lossy float text).
    for value in [5.0, 5.0, 7.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let arg = body.const_num(value);
        body.emit1(Inst::CallDirect { conv: CallConv::Napalm, callee: g, args: vec![arg] }, Type::Num);
    }
    let arg = body.const_num(1.0);
    let last = body.emit1(Inst::CallDirect { conv: CallConv::Napalm, callee: h, args: vec![arg] }, Type::Num);
    body.ret(last);
    let main = body.finish();

    assert!(Specialization.optimize_program(&mut hir));
    let clones: Vec<_> = hir.functions.iter().filter(|f| f.specialization.is_some()).collect();
    assert_eq!(clones.len(), 5);
    assert!(clones.iter().all(|f| hir.string(f.signature.name) == Some("g")
        && f.specialization.unwrap().origin == g));
    let callees: Vec<_> = kinds(hir.function(main).unwrap()).filter_map(|k| match k {
        Inst::CallDirect { callee, .. } => Some(*callee), _ => None,
    }).collect();
    assert_eq!(callees[0], callees[1]);
    assert!(callees[1..6].windows(2).all(|w| w[0] != w[1]));
    assert_eq!(callees[6], h, "unprofitable call sites keep the generic callee");
    assert!(!Specialization.optimize_program(&mut hir), "clones are reused, not re-cloned");
}

#[test]
fn licm_hoists_invariant_arithmetic_into_the_preheader() {
    let mut hir = Program::new();
    let module = hir.add_module("main", 0);
    let mut b = Builder::new(&mut hir, module, "loop", FunctionKind::Closure, vec![Type::Int32, Type::Int32], Type::Num);
    let (x, y) = (b.param(0), b.param(1));
    let entry = b.current_block();
    let zero = b.const_i32(0);
    let ten = b.const_i32(10);
    let header = b.new_block(&[Type::Int32]);
    let (body, exit) = (b.new_block(&[]), b.new_block(&[]));
    b.jump(header, vec![zero]);
    b.position_at(header);
    let i = b.block_param(header, 0);
    let cond = b.emit1(Inst::CmpInt { kind: CmpKind::Lt, left: i, right: ten }, Type::Bool);
    b.branch(cond, body, vec![], exit, vec![]);
    b.position_at(body);
    let invariant = b.emit1(Inst::IntBin { op: IntBinOp::Mul, left: x, right: y }, Type::Int32);
    let next = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: i, right: invariant }, Type::Int32);
    b.jump(header, vec![next]);
    b.position_at(exit);
    let result = b.emit1(Inst::Int32ToNum { value: i }, Type::Num);
    b.ret(result);
    let id = b.finish();

    assert!(Licm.optimize_function(hir.function_mut(id).unwrap()));
    let f = hir.function(id).unwrap();
    verify_function(f).unwrap();
    let ValueDef::InstResult(inst, _) = f.values[invariant.0 as usize].def else { unreachable!() };
    assert!(f.blocks[entry.0 as usize].insts.contains(&inst), "x * y belongs in the preheader");
    let ValueDef::InstResult(inst, _) = f.values[next.0 as usize].def else { unreachable!() };
    assert!(f.blocks[body.0 as usize].insts.contains(&inst), "the induction update is variant");
}

#[test]
fn pre_moves_partially_redundant_expressions_onto_the_missing_path() {
    let mut hir = Program::new();
    let module = hir.add_module("main", 0);
    let params = vec![Type::Int32, Type::Int32, Type::Bool];
    let mut b = Builder::new(&mut hir, module, "pre", FunctionKind::Closure, params, Type::Num);
    let (x, y, flag) = (b.param(0), b.param(1), b.param(2));
    let (left, right, join) = (b.new_block(&[]), b.new_block(&[]), b.new_block(&[]));
    b.branch(flag, left, vec![], right, vec![]);
    b.position_at(left);
    let early = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: x, right: y }, Type::Int32);
    let early = b.emit1(Inst::Int32ToNum { value: early }, Type::Num);
    let _ = early;
    b.jump(join, vec![]);
    b.position_at(right);
    b.jump(join, vec![]);
    b.position_at(join);
    let late = b.emit1(Inst::IntBin { op: IntBinOp::Add, left: x, right: y }, Type::Int32);
    let late = b.emit1(Inst::Int32ToNum { value: late }, Type::Num);
    b.ret(late);
    let id = b.finish();

    assert!(Pre.optimize_function(hir.function_mut(id).unwrap()));
    let f = hir.function(id).unwrap();
    verify_function(f).unwrap();
    let adds = |block: BlockId| f.blocks[block.0 as usize].insts.iter()
        .filter(|id| matches!(f.nodes[id.0 as usize].kind, Inst::IntBin { op: IntBinOp::Add, .. })).count();
    assert_eq!((adds(left), adds(right), adds(join)), (1, 1, 0));
    // Both x + y and its Number conversion are partially redundant.
    assert_eq!(f.blocks[join.0 as usize].params.len(), 2, "the join receives the values from both paths");
}

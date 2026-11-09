//! Synchronous generators reuse the VM's GC-traced, resumable fiber stacks.
//! The only new control-flow expression is GeneratorYield: its return arm must
//! leave the owning function (and run its cleanup), not a helper closure.
use super::*;

#[derive(Clone, Default)]
pub(super) struct GeneratorSignature {
    yielded: Option<TypeAnnotation>,
    pub(super) returned: Option<TypeAnnotation>,
    resumed: Option<TypeAnnotation>,
}

impl Parser<'_> {
    pub(super) fn generator_body(&mut self, params: Vec<Ident>, annotation: Option<TypeAnnotation>, start: usize) -> Result<FunctionBody> {
        let mut signature = GeneratorSignature::default();
        if let Some(annotation) = &annotation {
            let spelling = &self.source[annotation.span.as_range()];
            let compact = compact_type_spelling(spelling);
            if compact != "Generator" {
                let Some(("Generator", arguments)) = split_generic_type(&compact) else {
                    return self.error("a generator result annotation must be Generator<Yield, Return, Next>");
                };
                let args = split_type_top_level(arguments, ',');
                if args.is_empty() || args.len() > 3 {
                    return self.error("Generator accepts one to three type arguments");
                }
                // As for ordinary functions, `void` (and `undefined`, `any`) leaves
                // that channel unconstrained instead of meaning "exactly null".
                let types: Vec<_> = args.iter().map(|s| match compact_type_spelling(s).as_str() {
                    "void" | "undefined" | "any" | "unknown" => None,
                    _ => Some(classify_variable_type(s, annotation.span)),
                }).collect();
                signature.yielded = types.first().cloned().flatten();
                signature.returned = types.get(1).cloned().flatten();
                signature.resumed = types.get(2).cloned().flatten();
            }
        }
        let enclosing = self.generator.replace(signature.clone());
        self.lines();
        let block = self.block();
        self.generator = enclosing;
        let StmtKind::Block(body) = block?.kind else { unreachable!() };
        let param_types = self.take_parameter_types(&params);
        let mut statements = self.parameter_bindings(&params)?;
        statements.extend(body);
        let span = self.end(start);
        if let Some(returned) = &signature.returned {
            // Falling off the end is an ordinary return, and still obeys R.
            statements.push(Stmt { span, kind: StmtKind::Return(Some(Expr::new(
                ExprKind::TypedCollection {
                    expression: Box::new(Expr::new(ExprKind::Null, span)),
                    annotation: returned.clone(),
                }, span))) });
        }
        let mut ast = DecoratorAst { next_span: &mut self.synthetic_span, anchor: span };
        let implementation = Expr::new(ExprKind::Closure(FunctionBody {
            params: vec![], param_types: vec![],
            // IteratorClose injects return(null), even when ordinary source
            // returns have a non-null R. Widen the internal ABI honestly.
            return_type: signature.returned.map(|returned| TypeAnnotation::new(
                TypeAnnotationKind::Union(vec![returned, TypeAnnotation::new(TypeAnnotationKind::Null, span)]), span)),
            kind: FunctionBodyKind::Statements(statements), span: ast.span(),
        }), span);
        // The root frame converts both ordinary returns and uncaught exceptions
        // into private completion packets. User values cannot forge a completion.
        let error = ast.ident("generatorError");
        let call = invoke(implementation, "call", vec![], span);
        let root = ast.closure(vec![], vec![Stmt {
            span,
            kind: StmtKind::Try {
                body: Box::new(ast.gen_return(ast.gen_packet(0.0, call))),
                catch: Some(CatchClause {
                    binding: Some(error.clone()),
                    body: Box::new(ast.gen_return(ast.gen_packet(2.0, ast.read(&error)))),
                    span,
                }),
                finally_branch: None,
            },
        }]);
        let constructor = invoke(ast.gen_global("Generator"), "__new", vec![root], span);
        Ok(FunctionBody {
            params, param_types,
            return_type: Some(TypeAnnotation::new(TypeAnnotationKind::BuiltinGeneric(
                "Generator".into(), signature.yielded.map(Box::new)), span)),
            kind: FunctionBodyKind::Statements(vec![ast.gen_return(constructor)]), span,
        })
    }

    pub(super) fn generator_yield(&mut self) -> Result<Expr> {
        let start = self.peek().span.start;
        self.bump();
        let Some(signature) = self.generator.clone() else {
            return self.error("yield is only valid directly inside a generator function");
        };
        let line = self.peek().kind == Kind::Line;
        let delegated = !line && self.eat("*");
        let value = if !delegated && (line || matches!(self.text(), ";" | "}" | ")" | "]" | ",") || self.peek().kind == Kind::Eof) {
            Expr::new(ExprKind::Null, self.end(start))
        } else {
            self.lines();
            self.expr(1)?
        };
        let span = self.end(start);
        let value = if delegated {
            let mut ast = DecoratorAst { next_span: &mut self.synthetic_span, anchor: span };
            ast.gen_delegate(value, signature.yielded.as_ref())
        } else { value };
        Ok(Expr::new(ExprKind::GeneratorYield {
            value: Box::new(value), delegated,
            yield_type: signature.yielded,
            // yield* evaluates to the delegate's return, not a .next() argument.
            next_type: if delegated { None } else { signature.resumed },
        }, span))
    }
}

impl DecoratorAst<'_> {
    fn gen_global(&self, name: &str) -> Expr {
        self.read(&Spanned::new(name.into(), self.anchor))
    }
    fn gen_variable(&self, name: &Ident, value: Expr) -> Stmt {
        let mut statement = self.variable(name, value);
        if let StmtKind::Variable { annotation, .. } = &mut statement.kind {
            *annotation = Some(TypeAnnotation::new(TypeAnnotationKind::Any, self.anchor));
        }
        statement
    }
    fn gen_number(&self, n: f64) -> Expr { Expr::new(ExprKind::Number(n), self.anchor) }
    fn gen_null(&self) -> Expr { Expr::new(ExprKind::Null, self.anchor) }
    fn gen_packet(&self, mode: f64, value: Expr) -> Expr {
        Expr::new(ExprKind::List(vec![self.gen_number(mode), value]), self.anchor)
    }
    fn gen_index(&self, name: &Ident, index: f64) -> Expr {
        Expr::new(ExprKind::Subscript { receiver: Box::new(self.read(name)), args: vec![self.gen_number(index)] }, self.anchor)
    }
    fn gen_property(&self, name: &Ident, key: &str) -> Expr {
        Expr::new(ExprKind::Invoke(InvokeExpr {
            receiver: Some(Box::new(self.read(name))), name: Spanned::new(key.into(), self.anchor),
            args: vec![], syntax: InvokeSyntax::Getter,
        }), self.anchor)
    }
    fn gen_return(&self, value: Expr) -> Stmt { Stmt { kind: StmtKind::Return(Some(value)), span: self.anchor } }
    fn gen_if(&self, condition: Expr, body: Vec<Stmt>) -> Stmt {
        Stmt { kind: StmtKind::If { condition, then_branch: Box::new(Stmt {
            kind: StmtKind::Block(body), span: self.anchor,
        }), else_branch: None }, span: self.anchor }
    }
    fn gen_equal(&self, a: Expr, b: Expr) -> Expr {
        Expr::new(ExprKind::Binary { left: Box::new(a), operator: BinaryOp::Equal, right: Box::new(b) }, self.anchor)
    }
    fn gen_checked(&self, value: Expr, annotation: Option<&TypeAnnotation>) -> Expr {
        match annotation {
            Some(annotation) => Expr::new(ExprKind::TypedCollection { expression: Box::new(value), annotation: annotation.clone() }, self.anchor),
            None => value,
        }
    }
    fn gen_suspend(&self, value: Expr, annotation: Option<&TypeAnnotation>) -> Expr {
        invoke(self.gen_global("Fiber"), "yield", vec![self.gen_checked(value, annotation)], self.anchor)
    }

    fn gen_delegate(&mut self, value: Expr, yielded: Option<&TypeAnnotation>) -> Expr {
        let span = self.anchor;
        let target = self.ident("delegate");
        let result = self.ident("delegateResult");
        let packet = self.ident("delegateResume");
        let iterator = self.ident("delegateIterator");
        let sequence_packet = self.ident("sequenceResume");
        let mode = || self.gen_index(&packet, 0.0);
        let sent = || self.gen_index(&packet, 1.0);
        let call = |name: &str, args| invoke(self.read(&target), name, args, span);
        let mut generator_loop = vec![
            self.gen_if(self.gen_property(&result, "done"), vec![
                self.gen_return(Expr::new(ExprKind::List(vec![
                    // A delegate which completes .return() returns from the parent.
                    Expr::new(ExprKind::Conditional {
                        condition: Box::new(self.gen_equal(mode(), self.gen_number(1.0))),
                        then_expr: Box::new(self.gen_number(1.0)), else_expr: Box::new(self.gen_number(0.0)),
                    }, span), self.gen_property(&result, "value")]), span)),
            ]),
            self.assign(&packet, self.gen_suspend(self.gen_property(&result, "value"), yielded)),
        ];
        for (n, method) in [(0.0, "next"), (1.0, "return"), (2.0, "throw")] {
            generator_loop.push(self.gen_if(self.gen_equal(mode(), self.gen_number(n)), vec![
                self.assign(&result, call(method, vec![sent()])),
            ]));
        }
        let is_generator = invoke(self.gen_global("Generator"), "__is", vec![self.read(&target)], span);
        let generator_branch = self.gen_if(is_generator, vec![
            self.gen_variable(&packet, self.gen_packet(0.0, self.gen_null())),
            self.gen_variable(&result, call("next", vec![])),
            Stmt { kind: StmtKind::While { increment: None, condition: Expr::new(ExprKind::Bool(true), span),
                body: Box::new(Stmt { kind: StmtKind::Block(generator_loop), span }) }, span },
        ]);
        // All other Napalm sequences use iterate/iteratorValue. Their protocol
        // has no return value or exception injection; yield* completes with null.
        let sequence_loop = vec![
            self.assign(&iterator, call("iterate", vec![self.read(&iterator)])),
            self.gen_if(Expr::new(ExprKind::Conditional { condition: Box::new(self.read(&iterator)), then_expr: Box::new(Expr::new(ExprKind::Bool(false), span)), else_expr: Box::new(Expr::new(ExprKind::Bool(true), span)) }, span),
                vec![self.gen_return(self.gen_packet(0.0, self.gen_null()))]),
            self.gen_variable(&sequence_packet, self.gen_suspend(call("iteratorValue", vec![self.read(&iterator)]), yielded)),
            self.gen_if(Expr::new(ExprKind::Binary { left: Box::new(self.gen_index(&sequence_packet, 0.0)), operator: BinaryOp::NotEqual,
                right: Box::new(self.gen_number(0.0)) }, span), vec![self.gen_return(self.read(&sequence_packet))]),
        ];
        let body = vec![generator_branch, self.gen_variable(&iterator, self.gen_null()),
            Stmt { kind: StmtKind::While { increment: None, condition: Expr::new(ExprKind::Bool(true), span),
                body: Box::new(Stmt { kind: StmtKind::Block(sequence_loop), span }) }, span }];
        let helper = self.closure(vec![target], body);
        invoke(helper, "call", vec![value], span)
    }
}

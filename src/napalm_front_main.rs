//! Napalm's TypeScript-inspired surface syntax. Runtime-relevant variable types are retained.
//!
//! The frontend produces the existing engine AST directly, retaining source byte
//! spans. The legacy Napalm parser is used only by historical backend regressions.
pub use crate::napalm_front::*;
use logos::Logos;
use bumpalo::{Bump, collections::Vec as BumpVec};
use smallvec::SmallVec;
use compact_str::CompactString;
use crate::napalm_collections::{HashMap, HashSet};

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
#[logos(skip r"[ \t\r]+")]
#[logos(skip(r"//[^\r\n]*", allow_greedy = true))]
enum Kind {
    #[token("/*", comment)]
    Comment,
    #[token("\n")]
    Line,
    #[regex(r"[a-zA-Z_$][a-zA-Z0-9_$]*")]
    Word,
    #[regex(
        r"0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][+-]?[0-9]+)?"
    )]
    Number,
    #[token("\"", string)]
    #[token("'", string)]
    #[token("`", string)]
    String,
    #[regex(r"===|!==|>>>|\?\?=|&&=|\|\|=|\*\*=|\?\.|\?\?|\.\.\.|==|!=|<=|>=|=>|\+\+|--|\+=|-=|\*=|/=|%=|\*\*|&&|\|\||<<|>>|[{}()\[\],;:.?+*/%~!&|^=<>@-]")]
    Symbol,
    Eof,
}

fn comment(lex: &mut logos::Lexer<'_, Kind>) -> bool {
    if let Some(end) = lex.remainder().find("*/") {
        lex.bump(end + 2);
        true
    } else {
        false
    }
}

// Scan strings and nested ${...} expressions without losing their original spans.
fn quoted_end(bytes: &[u8], start: usize, depth: usize) -> Option<usize> {
    if depth > MAX_INTERPOLATION_DEPTH {
        return None;
    }
    let quote = bytes[start];
    let mut i = start + 1;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2;
            continue;
        }
        if bytes[i] == quote {
            return Some(i + 1);
        }
        if quote != b'`' && bytes[i] == b'\n' {
            return None;
        }
        if quote == b'`' && bytes[i..].starts_with(b"${") {
            i = interpolation_end(bytes, i + 2, depth + 1)? + 1;
        } else {
            i += 1;
        }
    }
    None
}

fn interpolation_end(bytes: &[u8], mut i: usize, depth: usize) -> Option<usize> {
    let mut braces = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' | b'`' => i = quoted_end(bytes, i, depth)?,
            b'/' if bytes[i..].starts_with(b"//") => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes[i..].starts_with(b"/*") => {
                i += 2;
                while i + 1 < bytes.len() && &bytes[i..i + 2] != b"*/" {
                    i += 1;
                }
                i += 2;
            }
            b'{' => {
                braces += 1;
                i += 1;
            }
            b'}' if braces == 0 => return Some(i),
            b'}' => {
                braces -= 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    None
}

fn string(lex: &mut logos::Lexer<'_, Kind>) -> bool {
    let start = lex.span().start;
    if let Some(end) = quoted_end(lex.source().as_bytes(), start, 0) {
        lex.bump(end - lex.span().end);
        true
    } else {
        false
    }
}

#[derive(Clone, Copy, Debug)]
struct Tok {
    kind: Kind,
    span: Span,
}

#[derive(Clone, Debug)]
enum BindingPattern {
    Ident(Ident),
    Array { items: Vec<Option<BindingPattern>>, rest: Option<Box<BindingPattern>>, span: Span },
    Object { items: Vec<(Expr, BindingPattern)>, rest: Option<Box<BindingPattern>>, span: Span },
    Default { pattern: Box<BindingPattern>, value: Expr },
}

impl BindingPattern {
    fn span(&self) -> Span {
        match self {
            BindingPattern::Ident(ident) => ident.span,
            BindingPattern::Default { pattern, value } => pattern.span().join(value.span),
            BindingPattern::Array { span, .. } | BindingPattern::Object { span, .. } => *span,
        }
    }

    fn is_ident(&self) -> bool {
        matches!(self, BindingPattern::Ident(_))
    }
}

type Result<T> = std::result::Result<T, ()>;

struct Parser<'a> {
    interfaces: Vec<InterfaceDecl>,
    type_aliases: Vec<TypeAliasDecl>,
    source: &'a str,
    arena: &'a Bump,
    tokens: BumpVec<'a, Tok>,
    // Only splitting a `>>` token mutates the stream during type lookahead.
    token_splits: SmallVec<[(usize, Tok); 4]>,
    pos: usize,
    depth: usize,
    diagnostics: Vec<Diagnostic>,
    exports: Vec<(Ident, Ident)>,
    immutable: HashSet<Span>,
    parameter_patterns: HashMap<Span, BindingPattern>,
}

fn classify_variable_type(source: &str, span: Span) -> TypeAnnotation {
    let parts = split_type_top_level(source, '|');
    if parts.len() > 1 {
        return TypeAnnotation::new(TypeAnnotationKind::Union(parts.into_iter()
            .map(|part| classify_variable_type(part, span)).collect()), span);
    }
    if source.trim_start().starts_with('{') {
        let arena = Bump::new();
        let mut parser = Parser::new(source, 0, source.len(), &arena);
        parser.lines();
        if let Ok(fields) = parser.interface_fields() {
            let mut annotation = TypeAnnotation::new(TypeAnnotationKind::Object(fields), span);
            while parser.eat("[") {
                if !parser.eat("]") { return TypeAnnotation::new(TypeAnnotationKind::Opaque(source.into()), span); }
                annotation = TypeAnnotation::new(TypeAnnotationKind::List(Some(Box::new(annotation))), span);
            }
            parser.lines();
            if parser.peek().kind == Kind::Eof { return annotation; }
            if parser.eat("|") {
                let other = classify_variable_type(&source[parser.peek().span.start..], span);
                return TypeAnnotation::new(TypeAnnotationKind::Union(vec![annotation, other]), span);
            }
        }
        return TypeAnnotation::new(TypeAnnotationKind::Never, span);
    }
    let mut source = source.trim();
    if let Some(rest) = source.strip_prefix("readonly") {
        let boundary = rest
            .chars()
            .next()
            .is_none_or(|ch| !(ch == '_' || ch == '$' || ch.is_ascii_alphanumeric()));
        if boundary {
            source = rest.trim_start();
        }
    }
    let compact: CompactString = source.chars().filter(|ch| !ch.is_whitespace()).collect();
    classify_compact_type(&compact, span)
}

fn classify_compact_type(source: &str, span: Span) -> TypeAnnotation {
    use TypeAnnotationKind as K;

    let source = source.trim();

    if source.len() >= 2 && source.starts_with('(') && source.ends_with(')')
        && encloses_complete_type(source, '(', ')')
    {
        return classify_compact_type(&source[1..source.len() - 1], span);
    }

    let union = split_type_top_level(source, '|');
    if union.len() > 1 {
        return TypeAnnotation::new(
            K::Union(
                union
                    .into_iter()
                    .map(|part| classify_compact_type(part, span))
                    .collect(),
            ),
            span,
        );
    }

    if source.ends_with("[]") {
        let element = classify_compact_type(&source[..source.len() - 2], span);
        return TypeAnnotation::new(K::List(Some(Box::new(element))), span);
    }

    if source.starts_with('[') && source.ends_with(']') {
        let inner = &source[1..source.len() - 1];
        let elements = split_type_top_level(inner, ',');
        let element = match elements.as_slice() {
            [] => None,
            [one] => Some(Box::new(classify_compact_type(one, span))),
            many => Some(Box::new(TypeAnnotation::new(
                K::Union(
                    many.iter()
                        .map(|part| classify_compact_type(part, span))
                        .collect(),
                ),
                span,
            ))),
        };
        return TypeAnnotation::new(K::List(element), span);
    }

    if source.starts_with('{') && source.ends_with('}') {
        return TypeAnnotation::new(K::Opaque(source.to_owned()), span);
    }

    if source.contains("=>") {
        return TypeAnnotation::new(K::Closure, span);
    }

    if let Some((base, args)) = split_generic_type(source) {
        let arguments = split_type_top_level(args, ',');
        return match base {
            "Array" | "ReadonlyArray" | "List" => {
                let element = arguments
                    .first()
                    .map(|part| Box::new(classify_compact_type(part, span)));
                TypeAnnotation::new(K::List(element), span)
            }
            "Map" | "ReadonlyMap" | "Record" => TypeAnnotation::new(K::Map, span),
            "Function" | "Fn" => TypeAnnotation::new(K::Closure, span),
            _ => TypeAnnotation::new(K::Named(base.to_owned()), span),
        };
    }

    let lower = source.to_ascii_lowercase();
    let kind = match lower.as_str() {
        "any" => K::Any,
        "never" => K::Never,
        "null" => K::Null,
        "boolean" | "bool" => K::Bool,
        "number" | "num" => K::Num,
        "string" => K::String,
        "object" => K::Opaque(source.to_owned()),
        "function" | "fn" => K::Closure,
        "true" | "false" => K::Bool,
        _ if source.starts_with('"') || source.starts_with('\'') || source.starts_with('`') => K::String,
        _ if source.parse::<f64>().is_ok() => K::Num,
        _ if source
            .split('.')
            .all(|part| !part.is_empty() && part.chars().all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())) =>
        {
            K::Named(source.to_owned())
        }
        _ => K::Opaque(source.to_owned()),
    };
    TypeAnnotation::new(kind, span)
}

fn split_generic_type(source: &str) -> Option<(&str, &str)> {
    let open = source.find('<')?;
    if !source.ends_with('>') {
        return None;
    }
    let base = &source[..open];
    let args = &source[open + 1..source.len() - 1];
    let mut depth = 0isize;
    for byte in args.bytes() {
        match byte {
            b'<' => depth += 1,
            b'>' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    (depth == 0).then_some((base, args))
}

fn split_type_top_level(source: &str, separator: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut angle = 0isize;
    let mut paren = 0isize;
    let mut bracket = 0isize;
    let mut brace = 0isize;
    for (index, ch) in source.char_indices() {
        match ch {
            '<' => angle += 1,
            '>' => angle -= 1,
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        if ch == separator && angle == 0 && paren == 0 && bracket == 0 && brace == 0 {
            out.push(&source[start..index]);
            start = index + ch.len_utf8();
        }
    }
    out.push(&source[start..]);
    out
}

fn encloses_complete_type(source: &str, open: char, close: char) -> bool {
    let mut depth = 0isize;
    for (index, ch) in source.char_indices() {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 && index + ch.len_utf8() != source.len() {
                return false;
            }
        }
    }
    depth == 0
}

pub fn parse(_name: &str, source: &str) -> ParseOutput {
    let arena = Bump::new();
    let mut p = Parser::new(source, 0, source.len(), &arena);
    let statements = p.statements(None);
    let module = ModuleAst {
        interfaces: p.interfaces,
        type_aliases: p.type_aliases,
        statements,
        span: Span::new(0, source.len()),
        exports: Some(p.exports),
    };
    validate_bindings(&module, &p.immutable, &mut p.diagnostics);
    ParseOutput {
        module,
        diagnostics: p.diagnostics,
    }
}

impl<'a> Parser<'a> {
    fn new(source: &'a str, mut start: usize, end: usize, arena: &'a Bump) -> Self {
        if start == 0 {
            if source.starts_with('\u{feff}') {
                start = 3;
            }
            if source[start..].starts_with("#!/") {
                start += source[start..].find('\n').unwrap_or(end - start);
            }
        }
        let mut tokens = BumpVec::new_in(arena);
        let mut diagnostics = Vec::new();
        let mut lexer = Kind::lexer(&source[start..end]);
        while let Some(kind) = lexer.next() {
            let span = Span::new(start + lexer.span().start, start + lexer.span().end);
            match kind {
                Ok(Kind::Comment) => {
                    if source[span.as_range()].contains('\n') {
                        tokens.push(Tok {
                            kind: Kind::Line,
                            span,
                        });
                    }
                }
                Ok(kind) => tokens.push(Tok { kind, span }),
                Err(_) => diagnostics.push(Diagnostic::error(
                    "NAPALM-L001",
                    "invalid token or unterminated string/comment",
                    span,
                )),
            }
        }
        tokens.push(Tok {
            kind: Kind::Eof,
            span: Span::empty(end),
        });
        Self {
            interfaces: Vec::new(),
            type_aliases: Vec::new(),
            source,
            tokens,
            arena,
            token_splits: SmallVec::new(),
            pos: 0,
            depth: 0,
            diagnostics,
            exports: Vec::new(),
            immutable: HashSet::default(),
            parameter_patterns: HashMap::default(),
        }
    }
    fn peek(&self) -> Tok {
        self.tokens[self.pos]
    }
    fn text(&self) -> &'a str {
        &self.source[self.peek().span.as_range()]
    }
    fn at(&self, s: &str) -> bool {
        self.text() == s
    }
    fn bump(&mut self) -> Tok {
        let t = self.peek();
        if t.kind != Kind::Eof {
            self.pos += 1;
        }
        t
    }
    fn eat(&mut self, s: &str) -> bool {
        if self.at(s) {
            self.bump();
            true
        } else {
            false
        }
    }
    fn lines(&mut self) {
        while self.peek().kind == Kind::Line {
            self.bump();
        }
    }
    fn separators(&mut self) {
        while self.peek().kind == Kind::Line || self.at(";") {
            self.bump();
        }
    }
    fn error<T>(&mut self, message: impl Into<String>) -> Result<T> {
        self.diagnostics
            .push(Diagnostic::error("NAPALM-P001", message, self.peek().span));
        Err(())
    }
    fn expect(&mut self, text: &str) -> Result<Tok> {
        if self.at(text) {
            Ok(self.bump())
        } else {
            self.error(format!("expected `{text}`, found `{}`", self.text()))
        }
    }
    fn ident(&mut self) -> Result<Ident> {
        if self.peek().kind != Kind::Word {
            return self.error("expected an identifier");
        }
        let t = self.bump();
        Ok(Spanned::new(
            self.source[t.span.as_range()].to_owned(),
            t.span,
        ))
    }
    fn end(&self, start: usize) -> Span {
        Span::new(start, self.tokens[self.pos.saturating_sub(1)].span.end)
    }
    fn boundary(&mut self) -> Result<()> {
        if self.eat(";")
            || self.peek().kind == Kind::Line
            || self.at("}")
            || self.peek().kind == Kind::Eof
        {
            Ok(())
        } else {
            self.error("expected `;` or a line break after statement")
        }
    }
    fn statements(&mut self, close: Option<&str>) -> Vec<Stmt> {
        let mut out = Vec::new();
        self.separators();
        while self.peek().kind != Kind::Eof && !close.is_some_and(|c| self.at(c)) {
            let before = self.pos;
            match self.declaration(close.is_none()) {
                Ok(statements) => out.extend(statements),
                Err(()) => {
                    while self.peek().kind != Kind::Eof
                        && self.peek().kind != Kind::Line
                        && !self.at(";")
                        && !self.at("}")
                    {
                        self.bump();
                    }
                }
            }
            if before == self.pos {
                self.bump();
            }
            self.separators();
        }
        out
    }
    fn declaration(&mut self, module: bool) -> Result<Vec<Stmt>> {
        self.lines();
        let start = self.peek().span.start;
        let exported = self.eat("export");
        if exported && !module {
            return self.error("exports are only allowed at module scope");
        }

        // Type-only exports disappear completely. They are accepted so ordinary
        // TypeScript declaration files and mixed source modules need no rewriting.
        if exported && self.eat("type") {
            if self.eat("{") {
                self.lines();
                while !self.at("}") {
                    self.ident()?;
                    if self.eat("as") {
                        self.ident()?;
                    }
                    self.lines();
                    if !self.eat(",") {
                        break;
                    }
                    self.lines();
                }
                self.expect("}")?;
                if self.eat("from") {
                    self.lines();
                    let module = self.string_expr()?;
                    if !matches!(module.kind, ExprKind::String(_)) {
                        return self.error("module specifiers cannot interpolate expressions");
                    }
                }
                self.boundary()?;
                return Ok(Vec::new());
            }
            let name = self.ident()?;
            self.type_parameters()?;
            self.expect("=")?;
            self.lines();
            let type_start = self.peek().span.start;
            self.ty()?;
            let type_end = self
                .tokens
                .get(self.pos.saturating_sub(1))
                .map_or(type_start, |token| token.span.end);
            let span = Span::new(type_start, type_end);
            let annotation = classify_variable_type(&self.source[span.as_range()], span);
            self.boundary()?;
            if self.interfaces.iter().any(|i| i.name.value == name.value)
                || self.type_aliases.iter().any(|alias| alias.name.value == name.value)
            {
                return self.error("duplicate interface/type declaration");
            }
            self.type_aliases.push(TypeAliasDecl { name, annotation });
            return Ok(Vec::new());
        }

        if exported && self.eat("{") {
            let export_start = self.exports.len();
            self.lines();
            while !self.at("}") {
                let type_only = self.eat("type");
                let local = self.ident()?;
                let public = if self.eat("as") {
                    self.ident()?
                } else {
                    local.clone()
                };
                if !type_only {
                    self.exports.push((local, public));
                }
                self.lines();
                if !self.eat(",") {
                    break;
                }
                self.lines();
            }
            self.expect("}")?;
            if self.eat("from") {
                let mut import = self.import()?;
                for (local, _) in &mut self.exports[export_start..] {
                    let imported = local.clone();
                    local.value = format!("<reexport:{}>", local.span.start);
                    self.immutable.insert(local.span);
                    import.bindings.push(ImportBinding {
                        imported,
                        local: local.clone(),
                        span: local.span,
                    });
                }
                self.boundary()?;
                return Ok(if import.bindings.is_empty() {
                    Vec::new()
                } else {
                    vec![Stmt {
                        kind: StmtKind::Import(import),
                        span: self.end(start),
                    }]
                });
            }
            self.boundary()?;
            return Ok(Vec::new());
        }

        let default_export = exported && self.eat("default");
        if default_export
            && !self.at("class")
            && !self.at("abstract")
            && !self.at("interface")
            && !self.at("function")
            && !(self.at("async") && matches!(self.next_text(), "function" | "concurrent"))
            && !(self.at("concurrent") && matches!(self.next_text(), "function" | "async"))
        {
            let name = Spanned::new("<default>".into(), self.peek().span);
            let initializer = self.expr(0)?;
            self.boundary()?;
            self.exports
                .push((name.clone(), Spanned::new("default".into(), name.span)));
            return Ok(vec![Stmt {
                kind: StmtKind::Variable {
                    name,
                    annotation: None,
                    initializer: Some(initializer),
                },
                span: self.end(start),
            }]);
        }

        if self.eat("type") {
            if !module {
                return self.error("type aliases must be declared at module scope");
            }
            let name = self.ident()?;
            self.type_parameters()?;
            self.expect("=")?;
            self.lines();
            let type_start = self.peek().span.start;
            self.ty()?;
            let type_end = self
                .tokens
                .get(self.pos.saturating_sub(1))
                .map_or(type_start, |token| token.span.end);
            let span = Span::new(type_start, type_end);
            let annotation = classify_variable_type(&self.source[span.as_range()], span);
            self.boundary()?;
            if self.interfaces.iter().any(|i| i.name.value == name.value)
                || self.type_aliases.iter().any(|alias| alias.name.value == name.value)
            {
                return self.error("duplicate interface/type declaration");
            }
            self.type_aliases.push(TypeAliasDecl { name, annotation });
            return Ok(Vec::new());
        }
        if self.eat("interface") {
            if !module { return self.error("interfaces must be declared at module scope"); }
            let name = self.ident()?;
            self.type_parameters()?;
            let mut parents = Vec::new();
            if self.eat("extends") {
                loop {
                    parents.push(self.ident()?);
                    if !self.eat(",") {
                        break;
                    }
                }
            }
            let fields = self.interface_fields()?;
            if self.interfaces.iter().any(|i| i.name.value == name.value)
                || self.type_aliases.iter().any(|alias| alias.name.value == name.value)
            {
                return self.error("duplicate interface/type declaration");
            }
            self.interfaces.push(InterfaceDecl { name, parents, fields });
            return Ok(Vec::new());
        }

        if self.eat("import") {
            if !module || exported {
                return self.error("imports are only allowed at module scope");
            }
            let type_only = self.eat("type");
            self.lines();
            let side_effect_only = self.peek().kind == Kind::String;
            let import = self.import()?;
            self.boundary()?;
            return Ok(if type_only || (!side_effect_only && import.bindings.is_empty()) {
                Vec::new()
            } else {
                vec![Stmt {
                    kind: StmtKind::Import(import),
                    span: self.end(start),
                }]
            });
        }

        // `declare` is purely a surface/type-system construct in Napalm. We parse
        // enough of the declaration to retain useful diagnostics, then erase it.
        if self.eat("declare") {
            if self.eat("function") {
                self.ident()?;
                self.type_parameters()?;
                self.params()?;
                self.annotation()?;
                self.boundary()?;
                return Ok(Vec::new());
            }
            if self.at("let") || self.at("const") || self.at("var") {
                self.bump();
                loop {
                    self.ident()?;
                    self.eat("!");
                    self.annotation()?;
                    if self.eat("=") {
                        self.expr(0)?;
                    }
                    if !self.eat(",") {
                        break;
                    }
                }
                self.boundary()?;
                return Ok(Vec::new());
            }
            if self.eat("class") {
                self.ident()?;
                self.type_parameters()?;
                if self.eat("extends") {
                    self.ty()?;
                }
                if self.eat("implements") {
                    loop {
                        self.ty()?;
                        if !self.eat(",") {
                            break;
                        }
                    }
                }
                self.lines();
                self.skip_balanced("{", "}")?;
                return Ok(Vec::new());
            }
            if self.eat("namespace") || self.eat("module") {
                if self.peek().kind == Kind::String {
                    self.bump();
                } else {
                    self.ident()?;
                }
                self.lines();
                self.skip_balanced("{", "}")?;
                return Ok(Vec::new());
            }
            return self.error("unsupported `declare` declaration");
        }

        if self.at("enum") || self.at("namespace") {
            return self.error(format!(
                "`{}` is recognized TypeScript syntax but is not lowered yet",
                self.text()
            ));
        }

        let mut out = if self.at("let") || self.at("const") || self.at("var") {
            let immutable = self.eat("const");
            if !immutable {
                self.bump();
            }
            let declarations = self.variables(immutable)?;
            self.boundary()?;
            declarations
        } else if self.at("function")
            || (self.at("async") && matches!(self.next_text(), "function" | "concurrent"))
            || (self.at("concurrent") && matches!(self.next_text(), "function" | "async"))
        {
            let mut async_ = false;
            let mut concurrent_ = false;
            loop {
                if !async_ && self.eat("async") {
                    async_ = true;
                } else if !concurrent_ && self.eat("concurrent") {
                    concurrent_ = true;
                } else {
                    break;
                }
            }
            if async_ && concurrent_ {
                return self.error("a `concurrent` function is already asynchronous at its call boundary; use `launch` + `await` instead of `async concurrent`");
            }
            self.expect("function")?;
            let name = if default_export && self.at("(") {
                Spanned::new("<default>".into(), self.peek().span)
            } else {
                self.ident()?
            };
            self.type_parameters()?;
            let params = self.params()?;
            self.annotation()?;
            // TypeScript overload signatures have no runtime body. Keep the final
            // implementation only; the overloads are erased like other types.
            if !self.at("{") {
                self.boundary()?;
                return Ok(Vec::new());
            }
            let body = self.function_body(params, start)?;
            let initializer = Expr::new(
                if concurrent_ {
                    ExprKind::ConcurrentClosure(body)
                } else if async_ {
                    ExprKind::AsyncClosure(body)
                } else {
                    ExprKind::Closure(body)
                },
                self.end(start),
            );
            vec![Stmt {
                kind: StmtKind::Variable {
                    name,
                    annotation: None,
                    initializer: Some(initializer),
                },
                span: self.end(start),
            }]
        } else {
            let abstract_class = self.eat("abstract");
            let foreign = self.eat("foreign");
            if self.eat("class") {
                self.class(start, foreign, default_export)?
            } else if foreign {
                return self.error("expected `class` after `foreign`");
            } else if abstract_class {
                return self.error("expected `class` after `abstract`");
            } else {
                if exported {
                    return self.error("expected an exported declaration");
                }
                vec![self.statement()?]
            }
        };
        if exported {
            for s in &out {
                let name = match &s.kind {
                    StmtKind::Variable { name, .. } => Some(name),
                    StmtKind::Class(c) => Some(&c.name),
                    _ => None,
                };
                if let Some(name) = name {
                    if name.value.starts_with("<destructure:") || name.value.starts_with("<pattern") {
                        continue;
                    }
                    self.exports.push((
                        name.clone(),
                        if default_export {
                            Spanned::new("default".into(), name.span)
                        } else {
                            name.clone()
                        },
                    ));
                }
            }
        }
        Ok(std::mem::take(&mut out))
    }
    fn next_text(&self) -> &'a str {
        self.nth_text(1)
    }

    fn nth_text(&self, offset: usize) -> &'a str {
        self.tokens
            .get(self.pos + offset)
            .map(|t| &self.source[t.span.as_range()])
            .unwrap_or("")
    }

    /// Parse and erase a TypeScript type-argument list. This is intentionally
    /// speculative at expression sites because `<` is also a runtime operator.
    fn type_arguments(&mut self) -> Result<()> {
        self.expect("<")?;
        self.lines();
        if self.at(">") {
            return self.error("a type-argument list cannot be empty");
        }
        loop {
            self.ty()?;
            self.lines();
            if !self.eat(",") {
                break;
            }
            self.lines();
        }
        self.type_close()
    }

    /// If the tokens at the current position form `<T, ...>` immediately followed
    /// by a call, consume them. Roll back completely on failure so `a < b > (c)`
    /// can still be parsed as ordinary operators when it is not a type-argument
    /// list according to our TypeScript-style lookahead.
    fn try_call_type_arguments(&mut self) -> bool {
        if !self.at("<") {
            return false;
        }
        let saved_pos = self.pos;
        let saved_splits = self.token_splits.len();
        let saved_diagnostics = self.diagnostics.len();
        let saved_depth = self.depth;
        if self.type_arguments().is_ok() {
            let after = self.pos;
            self.lines();
            let is_call = self.at("(");
            if is_call {
                return true;
            }
            self.pos = after;
        }
        self.pos = saved_pos;
        self.restore_token_splits(saved_splits);
        self.diagnostics.truncate(saved_diagnostics);
        self.depth = saved_depth;
        false
    }

    fn skip_balanced(&mut self, open: &str, close: &str) -> Result<()> {
        self.expect(open)?;
        let mut depth = 1usize;
        while self.peek().kind != Kind::Eof {
            if self.at(open) {
                depth += 1;
            } else if self.at(close) {
                depth -= 1;
                self.bump();
                if depth == 0 {
                    return Ok(());
                }
                continue;
            }
            self.bump();
        }
        self.error(format!("unterminated `{open}` ... `{close}` block"))
    }
    fn variables(&mut self, immutable: bool) -> Result<Vec<Stmt>> {
        let mut out = Vec::new();
        loop {
            self.lines();
            let pattern = self.binding_pattern()?;
            let annotation = self.variable_annotation()?;
            let initializer = if self.eat("=") {
                self.lines();
                Some(self.expr(0)?)
            } else {
                None
            };

            if pattern.is_ident() {
                let BindingPattern::Ident(name) = pattern else { unreachable!() };
                if immutable {
                    self.immutable.insert(name.span);
                }
                if immutable && initializer.is_none() {
                    return self.error("a const declaration needs an initializer");
                }
                out.push(Stmt {
                    span: self.end(name.span.start),
                    kind: StmtKind::Variable { name, annotation, initializer },
                });
            } else {
                let Some(initializer) = initializer else {
                    return self.error("a destructuring declaration needs an initializer");
                };
                let span = pattern.span();
                let temporary = Spanned::new(format!("<destructure:{}>", span.start), span);
                let source = Expr::new(ExprKind::Name(temporary.clone()), span);

                // Evaluate the RHS exactly once. The synthetic binding is not
                // source-addressable and cannot collide with source identifiers.
                out.push(Stmt {
                    span: span.join(initializer.span),
                    kind: StmtKind::Variable {
                        name: temporary,
                        annotation,
                        initializer: Some(initializer),
                    },
                });
                self.expand_binding_pattern(pattern, source, immutable, &mut out)?;
            }

            if !self.eat(",") {
                break;
            }
            self.lines();
        }
        Ok(out)
    }

    fn binding_element(&mut self) -> Result<BindingPattern> {
        let pattern = self.binding_pattern()?;
        self.binding_default(pattern)
    }

    fn binding_default(&mut self, pattern: BindingPattern) -> Result<BindingPattern> {
        if self.eat("=") {
            self.lines();
            Ok(BindingPattern::Default { pattern: Box::new(pattern), value: self.expr(1)? })
        } else { Ok(pattern) }
    }

    fn binding_pattern(&mut self) -> Result<BindingPattern> {
        if self.depth >= MAX_PARSE_DEPTH { return self.error("binding pattern is nested too deeply"); }
        self.depth += 1;
        let result = self.binding_pattern_inner();
        self.depth -= 1;
        result
    }

    fn binding_pattern_inner(&mut self) -> Result<BindingPattern> {
        let start = self.peek().span.start;
        if self.eat("[") {
            self.lines();
            let mut items = Vec::new();
            let mut rest = None;
            while !self.at("]") {
                if self.eat(",") { items.push(None); self.lines(); continue; }
                if self.eat("...") {
                    self.lines();
                    rest = Some(Box::new(self.binding_pattern()?));
                    self.lines();
                    if !self.at("]") { return self.error("rest element must be last, without a default or trailing comma"); }
                    break;
                }
                items.push(Some(self.binding_element()?));
                self.lines();
                if !self.eat(",") { break; }
                self.lines();
            }
            self.expect("]")?;
            return Ok(BindingPattern::Array { items, rest, span: self.end(start) });
        }
        if self.eat("{") {
            self.lines();
            let mut items = Vec::new();
            let mut rest = None;
            while !self.at("}") {
                if self.eat("...") {
                    self.lines();
                    rest = Some(Box::new(BindingPattern::Ident(self.ident()?)));
                    self.lines();
                    if !self.at("}") { return self.error("rest property must be last, without a default or trailing comma"); }
                    break;
                }
                let (key, shorthand) = if self.eat("[") {
                    self.lines();
                    let key = self.expr(0)?;
                    self.lines();
                    self.expect("]")?;
                    (key, None)
                } else if self.peek().kind == Kind::Word {
                    let name = self.ident()?;
                    (Expr::new(ExprKind::String(name.value.as_bytes().to_vec()), name.span), Some(name))
                } else if self.peek().kind == Kind::String {
                    let key = self.string_expr()?;
                    if !matches!(key.kind, ExprKind::String(_)) { return self.error("destructuring keys cannot interpolate"); }
                    (key, None)
                } else if self.peek().kind == Kind::Number {
                    (self.prefix()?, None)
                } else { return self.error("expected a property name in object destructuring"); };
                self.lines();
                let pattern = if self.eat(":") {
                    self.lines();
                    self.binding_element()?
                } else if let Some(name) = shorthand {
                    self.binding_default(BindingPattern::Ident(name))?
                } else { return self.error("destructuring property requires `:` and a binding"); };
                items.push((key, pattern));
                self.lines();
                if !self.eat(",") { break; }
                self.lines();
            }
            self.expect("}")?;
            return Ok(BindingPattern::Object { items, rest, span: self.end(start) });
        }
        Ok(BindingPattern::Ident(self.ident()?))
    }

    fn pattern_temporary(&mut self, tag: &str, span: Span, value: Expr, out: &mut Vec<Stmt>) -> Expr {
        let name = Spanned::new(format!("<{tag}:{}>", span.start), Span::empty(span.start));
        out.push(Stmt { span, kind: StmtKind::Variable { name: name.clone(), annotation: None, initializer: Some(value) } });
        Expr::new(ExprKind::Name(name), span)
    }

    fn expand_binding_pattern(&mut self, pattern: BindingPattern, source: Expr, immutable: bool, out: &mut Vec<Stmt>) -> Result<()> {
        match pattern {
            BindingPattern::Ident(name) => {
                if immutable { self.immutable.insert(name.span); }
                out.push(Stmt { span: name.span.join(source.span), kind: StmtKind::Variable { name, annotation: None, initializer: Some(source) } });
            }
            BindingPattern::Default { pattern, value } => {
                let span = source.span.join(value.span);
                let default = Expr::new(ExprKind::Logical { left: Box::new(source), operator: LogicalOp::Nullish, right: Box::new(value) }, span);
                self.expand_binding_pattern(*pattern, default, immutable, out)?;
            }
            BindingPattern::Array { items, rest, span } => {
                let checked = Expr::new(ExprKind::DestructureCheck { source: Box::new(source), is_array: true }, span);
                let source = self.pattern_temporary("pattern", span, checked, out);
                let count = items.len();
                for (index, item) in items.into_iter().enumerate() {
                    if let Some(item) = item {
                        let access = Expr::new(ExprKind::DestructureGet { source: Box::new(source.clone()), key: Box::new(Expr::new(ExprKind::Number(index as f64), span)) }, span);
                        self.expand_binding_pattern(item, access, immutable, out)?;
                    }
                }
                if let Some(rest) = rest {
                    let value = Expr::new(ExprKind::DestructureRest { source: Box::new(source), key: Box::new(Expr::new(ExprKind::Number(count as f64), span)) }, span);
                    self.expand_binding_pattern(*rest, value, immutable, out)?;
                }
            }
            BindingPattern::Object { items, rest, span } => {
                let checked = Expr::new(ExprKind::DestructureCheck { source: Box::new(source), is_array: false }, span);
                let source = self.pattern_temporary("pattern", span, checked, out);
                let mut excluded = Vec::new();
                for (key, item) in items {
                    let key = self.pattern_temporary("pattern-key", key.span, key, out);
                    excluded.push(key.clone());
                    let access = Expr::new(ExprKind::DestructureGet { source: Box::new(source.clone()), key: Box::new(key) }, span);
                    self.expand_binding_pattern(item, access, immutable, out)?;
                }
                if let Some(rest) = rest {
                    let value = Expr::new(ExprKind::DestructureRest { source: Box::new(source), key: Box::new(Expr::new(ExprKind::List(excluded), span)) }, span);
                    self.expand_binding_pattern(*rest, value, immutable, out)?;
                }
            }
        }
        Ok(())
    }

    fn import(&mut self) -> Result<ImportDecl> {
        self.lines();
        let mut bindings = Vec::new();
        let default = self.peek().kind == Kind::Word;
        if default {
            let local = self.ident()?;
            self.immutable.insert(local.span);
            bindings.push(ImportBinding {
                imported: Spanned::new("default".into(), local.span),
                span: local.span,
                local,
            });
            if self.eat(",") {
                self.lines();
            }
        }
        if self.eat("{") {
            self.lines();
            while !self.at("}") {
                let type_only = self.eat("type");
                let imported = self.ident()?;
                let local = if self.eat("as") {
                    self.ident()?
                } else {
                    imported.clone()
                };
                if !type_only {
                    self.immutable.insert(local.span);
                    bindings.push(ImportBinding {
                        span: imported.span.join(local.span),
                        imported,
                        local,
                    });
                }
                self.lines();
                if !self.eat(",") {
                    break;
                }
                self.lines();
            }
            self.expect("}")?;
            self.lines();
            self.expect("from")?;
            self.lines();
        } else if default {
            self.expect("from")?;
            self.lines();
        }
        if self.peek().kind != Kind::String || self.at("`") {
            return self.error("expected a quoted module specifier");
        }
        let value = self.string_expr()?;
        let ExprKind::String(module) = value.kind else {
            return self.error("module specifiers cannot interpolate expressions");
        };
        Ok(ImportDecl { module, bindings })
    }
    fn block(&mut self) -> Result<Stmt> {
        let start = self.expect("{")?.span.start;
        if self.depth >= MAX_PARSE_DEPTH {
            return self.error("syntax is nested too deeply");
        }
        self.depth += 1;
        let statements = self.statements(Some("}"));
        self.depth -= 1;
        self.expect("}")?;
        Ok(Stmt {
            kind: StmtKind::Block(statements),
            span: self.end(start),
        })
    }
    fn statement(&mut self) -> Result<Stmt> {
        if self.depth >= MAX_PARSE_DEPTH {
            return self.error("statement is nested too deeply");
        }
        self.depth += 1;
        let result = self.statement_inner();
        self.depth -= 1;
        result
    }
    fn statement_inner(&mut self) -> Result<Stmt> {
        self.lines();
        let start = self.peek().span.start;
        if self.at("{") {
            return self.block();
        }
        if self.at("switch") {
            return self.error("`switch` is recognized JavaScript/TypeScript syntax but is not lowered yet");
        }
        let kind = if self.eat("if") {
            let condition = self.condition()?;
            let then_branch = Box::new(self.statement()?);
            self.lines();
            let else_branch = if self.eat("else") {
                Some(Box::new(self.statement()?))
            } else {
                None
            };
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            }
        } else if self.eat("while") {
            let condition = self.condition()?;
            StmtKind::While {
                condition,
                body: Box::new(self.statement()?),
                increment: None,
            }
        } else if self.eat("do") {
            let body = Box::new(self.statement()?);
            self.lines();
            self.expect("while")?;
            let condition = self.condition()?;
            self.eat(";");
            StmtKind::DoWhile { body, condition }
        } else if self.eat("for") {
            return self.for_loop(start);
        } else if self.eat("try") {
            self.lines();
            let body = Box::new(self.block()?);
            self.lines();
            let catch = if self.eat("catch") {
                let cstart = self.peek().span.start;
                let binding = if self.eat("(") {
                    let n = self.ident()?;
                    self.annotation()?;
                    self.expect(")")?;
                    Some(n)
                } else {
                    None
                };
                self.lines();
                let body = Box::new(self.block()?);
                Some(CatchClause {
                    binding,
                    body,
                    span: self.end(cstart),
                })
            } else {
                None
            };
            self.lines();
            let finally_branch = if self.eat("finally") {
                self.lines();
                Some(Box::new(self.block()?))
            } else {
                None
            };
            if catch.is_none() && finally_branch.is_none() {
                return self.error("try requires catch or finally");
            }
            StmtKind::Try {
                body,
                catch,
                finally_branch,
            }
        } else if self.at("defer") || self.at("errdefer") {
            let error_only = self.eat("errdefer");
            if !error_only {
                self.bump();
            }
            let error_binding = if error_only && self.eat("(") {
                let n = self.ident()?;
                self.expect(")")?;
                Some(n)
            } else {
                None
            };
            self.lines();
            StmtKind::Defer {
                body: Box::new(self.block()?),
                error_only,
                error_binding,
            }
        } else {
            let kind = if self.eat("return") {
                let value = if self.at(";")
                    || self.at("}")
                    || matches!(self.peek().kind, Kind::Line | Kind::Eof)
                {
                    None
                } else {
                    Some(self.expr(0)?)
                };
                StmtKind::Return(value)
            } else if self.eat("throw") {
                if self.peek().kind == Kind::Line {
                    return self.error("a throw expression must start on the same line");
                }
                StmtKind::Throw(self.expr(0)?)
            } else if self.eat("break") {
                StmtKind::Break
            } else if self.eat("continue") {
                StmtKind::Continue
            } else if self.eat(";") {
                return Ok(Stmt {
                    kind: StmtKind::Block(Vec::new()),
                    span: self.end(start),
                });
            } else {
                StmtKind::Expression(self.expr(0)?)
            };
            self.boundary()?;
            kind
        };
        Ok(Stmt {
            kind,
            span: self.end(start),
        })
    }
    fn condition(&mut self) -> Result<Expr> {
        self.lines();
        self.expect("(")?;
        self.lines();
        let value = self.expr(0)?;
        self.lines();
        self.expect(")")?;
        Ok(value)
    }
    fn for_loop(&mut self, start: usize) -> Result<Stmt> {
        self.lines();
        self.expect("(")?;
        self.lines();
        let mut init = Vec::new();
        if self.at("let") || self.at("const") || self.at("var") {
            let immutable = self.eat("const");
            if !immutable {
                self.bump();
            }
            let save = self.pos;
            let pattern = self.binding_pattern()?;
            self.annotation()?;
            let for_in = if self.eat("of") {
                false
            } else if self.eat("in") {
                true
            } else {
                self.pos = save;
                init = self.variables(immutable)?;
                // Continue with the ordinary C-style `for` parser below.
                self.expect(";")?;
                self.lines();
                let condition = if self.at(";") {
                    Expr::new(ExprKind::Bool(true), self.peek().span)
                } else {
                    self.expr(0)?
                };
                self.expect(";")?;
                self.lines();
                let increment = if self.at(")") {
                    None
                } else {
                    Some(self.expr(0)?)
                };
                self.lines();
                self.expect(")")?;
                let body = Box::new(self.statement()?);
                init.push(Stmt {
                    kind: StmtKind::While {
                        condition,
                        body,
                        increment,
                    },
                    span: self.end(start),
                });
                return Ok(Stmt {
                    kind: StmtKind::Block(init),
                    span: self.end(start),
                });
            };

            {
                self.lines();
                let mut iterable = self.expr(0)?;
                if for_in {
                    let span = iterable.span;
                    iterable = Expr::new(
                        ExprKind::Invoke(InvokeExpr {
                            receiver: Some(Box::new(iterable)),
                            name: Spanned::new("keys".into(), span),
                            args: Vec::new(),
                            syntax: InvokeSyntax::Getter,
                        }),
                        span,
                    );
                }
                self.lines();
                self.expect(")")?;
                let body = self.statement()?;

                let (name, body) = match pattern {
                    BindingPattern::Ident(name) => {
                        if immutable {
                            self.immutable.insert(name.span);
                        }
                        (name, body)
                    }
                    pattern => {
                        let pattern_span = pattern.span();
                        let temporary = Spanned::new(
                            format!("<for-destructure:{}>", pattern_span.start),
                            pattern_span,
                        );
                        let source = Expr::new(ExprKind::Name(temporary.clone()), pattern_span);
                        let mut prefix = Vec::new();
                        self.expand_binding_pattern(pattern, source, immutable, &mut prefix)?;
                        prefix.push(body);
                        (
                            temporary,
                            Stmt {
                                span: self.end(start),
                                kind: StmtKind::Block(prefix),
                            },
                        )
                    }
                };

                return Ok(Stmt {
                    kind: StmtKind::For {
                        name,
                        iterable,
                        body: Box::new(body),
                    },
                    span: self.end(start),
                });
            }
        } else if !self.at(";") {
            let e = self.expr(0)?;
            init.push(Stmt {
                span: e.span,
                kind: StmtKind::Expression(e),
            });
        }
        self.expect(";")?;
        self.lines();
        let condition = if self.at(";") {
            Expr::new(ExprKind::Bool(true), self.peek().span)
        } else {
            self.expr(0)?
        };
        self.expect(";")?;
        self.lines();
        let increment = if self.at(")") {
            None
        } else {
            Some(self.expr(0)?)
        };
        self.lines();
        self.expect(")")?;
        let body = Box::new(self.statement()?);
        init.push(Stmt {
            kind: StmtKind::While {
                condition,
                body,
                increment,
            },
            span: self.end(start),
        });
        Ok(Stmt {
            kind: StmtKind::Block(init),
            span: self.end(start),
        })
    }

    fn params(&mut self) -> Result<Vec<Ident>> {
        let (params, _) = self.params_with_properties(false)?;
        Ok(params)
    }

    /// Parse TypeScript parameters. Constructor parameter-properties are returned
    /// separately so class lowering can synthesize the same fixed-slot storage as
    /// an explicit `x: T` class field while preserving the existing backend AST.
    fn params_with_properties(&mut self, constructor: bool) -> Result<(Vec<Ident>, Vec<(Ident, Option<TypeAnnotation>)>)> {
        self.expect("(")?;
        self.lines();
        let mut params = Vec::new();
        let mut properties = Vec::new();
        while !self.at(")") {
            let mut property = false;
            loop {
                if self.eat("public") || self.eat("private") || self.eat("protected") {
                    if !constructor {
                        return self.error("parameter visibility modifiers are only valid in constructors");
                    }
                    property = true;
                } else if self.eat("readonly") {
                    if !constructor {
                        return self.error("`readonly` parameter properties are only valid in constructors");
                    }
                    property = true;
                } else if self.eat("override") {
                    if !constructor {
                        return self.error("`override` parameter properties are only valid in constructors");
                    }
                    property = true;
                } else {
                    break;
                }
            }
            if self.eat("...") {
                return self.error("rest parameters are not implemented yet");
            }
            let pattern = self.binding_pattern()?;
            let param = match pattern {
                BindingPattern::Ident(name) => name,
                pattern => {
                    if property { return self.error("parameter properties require a simple identifier"); }
                    let span = pattern.span();
                    self.parameter_patterns.insert(span, pattern);
                    Spanned::new(format!("<parameter:{}>", span.start), span)
                }
            };
            // Optionality is a type-system marker. The current VM still has fixed
            // arity, so callers must supply the argument until default/optional
            // argument lowering is added deliberately.
            self.eat("?");
            let annotation = self.variable_annotation()?;
            if self.eat("=") {
                return self.error("default parameters are not implemented yet");
            }
            if property {
                properties.push((param.clone(), annotation));
            }
            params.push(param);
            self.lines();
            if !self.eat(",") {
                break;
            }
            self.lines();
        }
        self.expect(")")?;
        if params.len() > MAX_PARAMETERS {
            return self.error("at most 16 parameters are supported");
        }
        Ok((params, properties))
    }
    fn parameter_bindings(&mut self, params: &[Ident]) -> Result<Vec<Stmt>> {
        let mut prefix = Vec::new();
        for param in params {
            if let Some(pattern) = self.parameter_patterns.remove(&param.span) {
                let source = Expr::new(ExprKind::Name(param.clone()), param.span);
                self.expand_binding_pattern(pattern, source, false, &mut prefix)?;
            }
        }
        Ok(prefix)
    }

    fn function_body(&mut self, params: Vec<Ident>, start: usize) -> Result<FunctionBody> {
        self.lines();
        let block = self.block()?;
        let StmtKind::Block(body_statements) = block.kind else {
            unreachable!()
        };
        let mut statements = self.parameter_bindings(&params)?;
        statements.extend(body_statements);
        Ok(FunctionBody {
            params,
            kind: FunctionBodyKind::Statements(statements),
            span: self.end(start),
        })
    }

    // An explicit type grammar avoids swallowing malformed declarations while
    // erasing annotations, aliases, interfaces and generic parameter lists.
    fn annotation(&mut self) -> Result<()> {
        if self.eat(":") {
            self.ty()?;
        }
        Ok(())
    }

    fn variable_annotation(&mut self) -> Result<Option<TypeAnnotation>> {
        if !self.eat(":") {
            return Ok(None);
        }

        self.lines();
        let start = self.peek().span.start;
        self.ty()?;
        let end = self
            .tokens
            .get(self.pos.saturating_sub(1))
            .map_or(start, |token| token.span.end);
        let span = Span::new(start, end);
        Ok(Some(classify_variable_type(&self.source[span.as_range()], span)))
    }
    fn type_parameters(&mut self) -> Result<()> {
        if self.eat("<") {
            loop {
                self.lines();
                self.ident()?;
                if self.eat("extends") {
                    self.ty()?;
                }
                if self.eat("=") {
                    self.ty()?;
                }
                if !self.eat(",") {
                    break;
                }
            }
            self.type_close()?;
        }
        Ok(())
    }
    fn restore_token_splits(&mut self, checkpoint: usize) {
        while self.token_splits.len() > checkpoint {
            let (position, original) = self.token_splits.pop().unwrap();
            self.tokens.remove(position + 1);
            self.tokens[position] = original;
        }
    }

    fn type_close(&mut self) -> Result<()> {
        self.lines();
        if self.at(">>") {
            let original = self.peek();
            let span = original.span;
            self.token_splits.push((self.pos, original));
            self.tokens[self.pos].span = Span::new(span.start, span.start + 1);
            self.tokens.insert(
                self.pos + 1,
                Tok {
                    kind: Kind::Symbol,
                    span: Span::new(span.start + 1, span.end),
                },
            );
        }
        self.expect(">")?;
        Ok(())
    }
    fn ty(&mut self) -> Result<()> {
        if self.depth >= MAX_PARSE_DEPTH {
            return self.error("type is nested too deeply");
        }
        self.depth += 1;
        let result = self.type_inner();
        self.depth -= 1;
        result
    }
    fn type_inner(&mut self) -> Result<()> {
        self.lines();
        if self.eat("readonly") || self.eat("keyof") || self.eat("typeof") || self.eat("infer") {
            self.ty()?;
        } else if self.eat("...") {
            self.ty()?;
        } else if self.at("{") {
            self.type_object()?;
        } else if self.eat("[") {
            self.lines();
            while !self.at("]") {
                self.ty()?;
                if !self.eat(",") {
                    break;
                }
            }
            self.lines();
            self.expect("]")?;
        } else if self.eat("(") {
            self.lines();
            while !self.at(")") {
                if self.peek().kind == Kind::Word && matches!(self.next_text(), ":" | "?") {
                    self.bump();
                    if self.eat("?") {
                        self.expect(":")?;
                    } else {
                        self.expect(":")?;
                    }
                }
                self.ty()?;
                self.lines();
                if !self.eat(",") {
                    break;
                }
                self.lines();
            }
            self.expect(")")?;
            if self.eat("=>") {
                self.ty()?;
            }
        } else if matches!(self.peek().kind, Kind::Word | Kind::String | Kind::Number) {
            self.bump();
            while self.eat(".") {
                self.ident()?;
            }
            if self.eat("<") {
                loop {
                    self.ty()?;
                    if !self.eat(",") {
                        break;
                    }
                }
                self.type_close()?;
            }
        } else {
            return self.error("expected a type");
        }
        while self.eat("[") {
            if !self.eat("]") {
                self.ty()?;
                self.expect("]")?;
            }
        }
        // Optional tuple/member marker: compile-time only.
        self.eat("?");
        if self.eat("extends") {
            self.ty()?;
            self.expect("?")?;
            self.ty()?;
            self.expect(":")?;
            self.ty()?;
        } else if self.eat("|") || self.eat("&") {
            self.ty()?;
        }
        Ok(())
    }
    fn interface_fields(&mut self) -> Result<Vec<InterfaceField>> {
        self.lines();
        self.expect("{")?;
        self.separators();
        let mut fields = Vec::new();
        while !self.at("}") {
            let readonly = self.eat("readonly");
            let field_name = self.ident()?;
            let optional = self.eat("?");
            let method = self.at("(");
            if method { self.params()?; }
            let Some(mut annotation) = self.variable_annotation()? else {
                return self.error("interface fields require a type");
            };
            if method { annotation.kind = TypeAnnotationKind::Closure; }
            if fields.iter().any(|f: &InterfaceField| f.name.value == field_name.value) {
                return self.error("duplicate interface field");
            }
            fields.push(InterfaceField { name: field_name, annotation, optional, readonly });
            if !self.eat(",") { self.boundary()?; }
            self.separators();
        }
        self.expect("}")?;
        Ok(fields)
    }

    fn type_object(&mut self) -> Result<()> {
        self.lines();
        self.expect("{")?;
        self.separators();
        while !self.at("}") {
            self.eat("readonly");
            self.ident()?;
            self.eat("?");
            if self.at("(") {
                self.params()?;
            }
            self.expect(":")?;
            self.ty()?;
            if !self.eat(",") {
                self.boundary()?;
            }
            self.separators();
        }
        self.expect("}")?;
        Ok(())
    }

    fn expr(&mut self, min: u8) -> Result<Expr> {
        if self.depth >= MAX_PARSE_DEPTH {
            return self.error("expression is nested too deeply");
        }
        self.depth += 1;
        let result = self.expression(min);
        self.depth -= 1;
        result
    }
    fn expression(&mut self, min: u8) -> Result<Expr> {
        let mut left = self.prefix()?;
        loop {
            // Newlines may continue member access and binary expressions. Return,
            // throw and postfix ++/-- enforce their own line-terminator rules.
            let save = self.pos;
            self.lines();
            if self.pos != save
                && !(self.at(".")
                    || self.at("?.")
                    || self.at("??")
                    || self.at("&&")
                    || self.at("||")
                    || infix(self.text()).is_some())
            {
                self.pos = save;
            }
            let start = left.span.start;
            if min <= 18 && self.eat("?.") {
                self.lines();
                if self.at("(") {
                    let args = self.args()?;
                    left = self.append_chain_call(left, args, true, self.end(start));
                } else if self.eat("[") {
                    self.lines();
                    let key = self.expr(0)?;
                    self.lines();
                    self.expect("]")?;
                    left = self.append_chain_subscript(left, key, true, self.end(start));
                } else {
                    let name = self.ident()?;
                    left = self.append_chain_property(left, name, true, self.end(start));
                }
                continue;
            }
            if min <= 18 && self.eat(".") {
                self.lines();
                let name = self.ident()?;
                left = self.append_chain_property(left, name, false, self.end(start));
                continue;
            }
            if min <= 18 && self.eat("[") {
                self.lines();
                let key = self.expr(0)?;
                self.lines();
                self.expect("]")?;
                left = self.append_chain_subscript(left, key, false, self.end(start));
                continue;
            }
            if min <= 18 && self.try_call_type_arguments() {
                // Type arguments are compile-time only. The following iteration
                // consumes the actual argument list and emits the normal call AST.
                continue;
            }
            if min <= 18 && self.at("(") {
                let args = self.args()?;
                left = self.append_chain_call(left, args, false, self.end(start));
                continue;
            }
            if min <= 18 && self.eat("!") {
                // TypeScript non-null assertion. Napalm's current type annotations
                // are erased, so there is intentionally no runtime check here.
                continue;
            }
            if min <= 17 && (self.at("++") || self.at("--")) {
                let op = if self.eat("++") {
                    BinaryOp::Add
                } else {
                    self.bump();
                    BinaryOp::Subtract
                };
                self.assignment_target(&left)?;
                left = Expr::new(
                    ExprKind::Update {
                        target: Box::new(left),
                        operator: op,
                        value: Box::new(Expr::new(ExprKind::Number(1.0), self.end(start))),
                        postfix: true,
                    },
                    self.end(start),
                );
                continue;
            }
            if min <= 10 && self.eat("as") {
                self.ty()?;
                continue;
            }
            if min <= 10 && self.eat("satisfies") {
                self.ty()?;
                continue;
            }
            if matches!(self.text(), ">>>") {
                return self.error(format!("`{}` is recognized TypeScript syntax but is not lowered yet", self.text()));
            }
            if min <= 1 && matches!(self.text(), "&&=" | "||=" | "??=") {
                let operator = match self.text() {
                    "&&=" => LogicalOp::And,
                    "||=" => LogicalOp::Or,
                    _ => LogicalOp::Nullish,
                };
                self.bump();
                self.assignment_target(&left)?;
                self.lines();
                let value = Box::new(self.expr(1)?);
                left = Expr::new(
                    ExprKind::LogicalAssign {
                        target: Box::new(left),
                        operator,
                        value,
                    },
                    self.end(start),
                );
                continue;
            }
            if min <= 1 && self.eat("**=") {
                self.assignment_target(&left)?;
                self.lines();
                let value = Box::new(self.expr(1)?);
                left = Expr::new(
                    ExprKind::Update {
                        target: Box::new(left),
                        operator: BinaryOp::Power,
                        value,
                        postfix: false,
                    },
                    self.end(start),
                );
                continue;
            }
            if min <= 1 && matches!(self.text(), "=" | "+=" | "-=" | "*=" | "/=" | "%=") {
                let op = self.text();
                self.bump();
                if op == "=" { self.destructuring_target(&left)?; }
                else { self.assignment_target(&left)?; }
                self.lines();
                let value = Box::new(self.expr(1)?);
                let kind = if op == "=" {
                    ExprKind::Assign {
                        target: Box::new(left),
                        value,
                    }
                } else {
                    ExprKind::Update {
                        target: Box::new(left),
                        operator: infix(&op[..1]).unwrap().1,
                        value,
                        postfix: false,
                    }
                };
                left = Expr::new(kind, self.end(start));
                continue;
            }
            if min <= 2 && self.eat("?") {
                self.lines();
                let then_expr = Box::new(self.expr(0)?);
                self.lines();
                self.expect(":")?;
                self.lines();
                let else_expr = Box::new(self.expr(1)?);
                left = Expr::new(
                    ExprKind::Conditional {
                        condition: Box::new(left),
                        then_expr,
                        else_expr,
                    },
                    self.end(start),
                );
                continue;
            }
            if self.at("**") {
                let bp = 13;
                if bp < min {
                    break;
                }
                self.bump();
                self.lines();
                // Exponentiation is right-associative: a ** b ** c == a ** (b ** c).
                let right = Box::new(self.expr(bp)?);
                left = Expr::new(
                    ExprKind::Binary {
                        left: Box::new(left),
                        operator: BinaryOp::Power,
                        right,
                    },
                    self.end(start),
                );
                continue;
            }
            if self.at("??") {
                let bp = 3;
                if bp < min {
                    break;
                }
                self.bump();
                self.lines();
                let right = Box::new(self.expr(bp + 1)?);
                left = Expr::new(
                    ExprKind::Logical {
                        left: Box::new(left),
                        operator: LogicalOp::Nullish,
                        right,
                    },
                    self.end(start),
                );
                continue;
            }
            if self.at("&&") || self.at("||") {
                let (bp, operator) = if self.at("&&") {
                    (4, LogicalOp::And)
                } else {
                    (3, LogicalOp::Or)
                };
                if bp < min {
                    break;
                }
                self.bump();
                self.lines();
                let right = Box::new(self.expr(bp + 1)?);
                left = Expr::new(
                    ExprKind::Logical {
                        left: Box::new(left),
                        operator,
                        right,
                    },
                    self.end(start),
                );
                continue;
            }
            let Some((bp, operator)) = infix(self.text()) else {
                break;
            };
            if bp < min {
                break;
            }
            self.bump();
            self.lines();
            let right = Box::new(self.expr(bp + 1)?);
            left = Expr::new(
                ExprKind::Binary {
                    left: Box::new(left),
                    operator,
                    right,
                },
                self.end(start),
            );
        }
        Ok(left)
    }
    fn append_chain_property(
        &mut self,
        left: Expr,
        name: Ident,
        optional: bool,
        span: Span,
    ) -> Expr {
        match left.kind {
            ExprKind::OptionalChain { base, mut segments } => {
                segments.push(OptionalChainSegment::Property {
                    name,
                    optional,
                    call_args: None,
                });
                Expr::new(ExprKind::OptionalChain { base, segments }, span)
            }
            kind if optional => {
                let base = Box::new(Expr::new(kind, left.span));
                Expr::new(
                    ExprKind::OptionalChain {
                        base,
                        segments: vec![OptionalChainSegment::Property {
                            name,
                            optional: true,
                            call_args: None,
                        }],
                    },
                    span,
                )
            }
            kind => Expr::new(
                ExprKind::Invoke(InvokeExpr {
                    receiver: Some(Box::new(Expr::new(kind, left.span))),
                    name,
                    args: Vec::new(),
                    syntax: InvokeSyntax::Getter,
                }),
                span,
            ),
        }
    }

    fn append_chain_subscript(
        &mut self,
        left: Expr,
        key: Expr,
        optional: bool,
        span: Span,
    ) -> Expr {
        match left.kind {
            ExprKind::OptionalChain { base, mut segments } => {
                segments.push(OptionalChainSegment::Subscript { key, optional });
                Expr::new(ExprKind::OptionalChain { base, segments }, span)
            }
            kind if optional => Expr::new(
                ExprKind::OptionalChain {
                    base: Box::new(Expr::new(kind, left.span)),
                    segments: vec![OptionalChainSegment::Subscript {
                        key,
                        optional: true,
                    }],
                },
                span,
            ),
            kind => Expr::new(
                ExprKind::Subscript {
                    receiver: Box::new(Expr::new(kind, left.span)),
                    args: vec![key],
                },
                span,
            ),
        }
    }

    fn append_chain_call(
        &mut self,
        left: Expr,
        args: Vec<Expr>,
        optional: bool,
        span: Span,
    ) -> Expr {
        match left.kind {
            ExprKind::OptionalChain { base, mut segments } => {
                // Preserve the fast/direct method representation for the common
                // `obj?.method(...)` and `obj?.method(...).field` cases.
                if !optional {
                    if let Some(OptionalChainSegment::Property { call_args, .. }) =
                        segments.last_mut()
                    {
                        if call_args.is_none() {
                            *call_args = Some(args);
                            return Expr::new(ExprKind::OptionalChain { base, segments }, span);
                        }
                    }
                }
                segments.push(OptionalChainSegment::Call { args, optional });
                Expr::new(ExprKind::OptionalChain { base, segments }, span)
            }
            kind if optional => Expr::new(
                ExprKind::OptionalChain {
                    base: Box::new(Expr::new(kind, left.span)),
                    segments: vec![OptionalChainSegment::Call {
                        args,
                        optional: true,
                    }],
                },
                span,
            ),
            ExprKind::Invoke(mut call) if call.syntax == InvokeSyntax::Getter => {
                call.args = args;
                call.syntax = InvokeSyntax::Method;
                Expr::new(ExprKind::Invoke(call), span)
            }
            kind => invoke(Expr::new(kind, left.span), "call", args, span),
        }
    }

    fn destructuring_target(&mut self, e: &Expr) -> Result<()> {
        match &e.kind {
            ExprKind::List(items) => {
                for (i, item) in items.iter().enumerate() {
                    if matches!(item.kind, ExprKind::Null) && item.span.start == item.span.end { continue; }
                    if let ExprKind::Spread(target) = &item.kind {
                        if i + 1 != items.len() || matches!(target.kind, ExprKind::Assign { .. })
                            || self.rest_has_trailing_comma(item.span.end, e.span.end) {
                            return self.error("rest element must be last and cannot have a default");
                        }
                        self.destructuring_target(target)?;
                    } else { self.destructuring_target(item)?; }
                }
                Ok(())
            }
            ExprKind::Map(entries) => {
                for (i, entry) in entries.iter().enumerate() {
                    if entry.spread {
                        if i + 1 != entries.len() || self.rest_has_trailing_comma(entry.span.end, e.span.end) {
                            return self.error("rest property must be last, without a trailing comma");
                        }
                        self.assignment_target(&entry.value)?;
                    } else { self.destructuring_target(&entry.value)?; }
                }
                Ok(())
            }
            ExprKind::Assign { target, .. } => self.destructuring_target(target),
            _ => self.assignment_target(e),
        }
    }

    fn rest_has_trailing_comma(&self, start: usize, end: usize) -> bool {
        let mut lexer = Kind::lexer(&self.source[start..end]);
        while let Some(token) = lexer.next() {
            if token == Ok(Kind::Symbol) && lexer.slice() == "," { return true; }
        }
        false
    }

    fn assignment_target(&mut self, e: &Expr) -> Result<()> {
        if matches!(&e.kind, ExprKind::Name(_) | ExprKind::Subscript { .. })
            || matches!(&e.kind, ExprKind::Invoke(i) if i.syntax == InvokeSyntax::Getter)
        {
            Ok(())
        } else {
            self.error("invalid assignment target")
        }
    }
    fn args(&mut self) -> Result<Vec<Expr>> {
        self.expect("(")?;
        self.lines();
        let mut out = Vec::new();
        while !self.at(")") {
            let arg_start = self.peek().span.start;
            let arg = if self.eat("...") {
                self.lines();
                let value = self.expr(1)?;
                Expr::new(ExprKind::Spread(Box::new(value)), self.end(arg_start))
            } else {
                self.expr(0)?
            };
            out.push(arg);
            self.lines();
            if !self.eat(",") {
                break;
            }
            self.lines();
        }
        self.expect(")")?;
        if out.len() > MAX_PARAMETERS {
            return self.error("at most 16 arguments are supported");
        }
        Ok(out)
    }
    fn arrow_ahead(&self) -> bool {
        if !self.at("(") {
            if self.peek().kind != Kind::Word {
                return false;
            }
            if self.next_text() == "=>" {
                return true;
            }
            // `x: T => ...` and `x?: T => ...`. Scan only to the current line so
            // an unrelated later arrow cannot turn a normal expression into one.
            if matches!(self.next_text(), ":" | "?") {
                for i in self.pos + 1..self.tokens.len() {
                    let token = self.tokens[i];
                    if token.kind == Kind::Line || token.kind == Kind::Eof {
                        break;
                    }
                    let text = &self.source[token.span.as_range()];
                    if text == "=>" {
                        return true;
                    }
                    if matches!(text, ";" | "=" | "{") {
                        break;
                    }
                }
            }
            return false;
        }
        let mut depth = 0;
        for i in self.pos..self.tokens.len() {
            let text = &self.source[self.tokens[i].span.as_range()];
            match text {
                "(" => depth += 1,
                ")" => {
                    depth -= 1;
                    if depth == 0 {
                        return self.tokens.get(i + 1).is_some_and(|t| {
                            matches!(&self.source[t.span.as_range()], "=>" | ":")
                        });
                    }
                }
                _ => {}
            }
        }
        false
    }
    fn closure(&mut self, start: usize, async_: bool, function: bool) -> Result<Expr> {
        if self.at("<") {
            self.type_parameters()?;
        }
        let params = if self.at("(") {
            self.params()?
        } else {
            let param = self.ident()?;
            self.eat("?");
            vec![param]
        };
        self.annotation()?;
        let body = if function {
            self.function_body(params, start)?
        } else {
            self.expect("=>")?;
            self.lines();
            if self.at("{") {
                self.function_body(params, start)?
            } else {
                let value = self.expr(1)?;
                let mut statements = self.parameter_bindings(&params)?;
                let kind = if statements.is_empty() { FunctionBodyKind::Expression(Box::new(value)) }
                else {
                    statements.push(Stmt { span: value.span, kind: StmtKind::Return(Some(value)) });
                    FunctionBodyKind::Statements(statements)
                };
                FunctionBody { params, kind, span: self.end(start) }
            }
        };
        Ok(Expr::new(
            if async_ {
                ExprKind::AsyncClosure(body)
            } else {
                ExprKind::Closure(body)
            },
            self.end(start),
        ))
    }
    fn prefix(&mut self) -> Result<Expr> {
        let start = self.peek().span.start;
        if self.at("<") {
            let saved_pos = self.pos;
            let saved_splits = self.token_splits.len();
            let saved_diagnostics = self.diagnostics.len();
            let saved_depth = self.depth;
            if self.type_parameters().is_ok() && self.arrow_ahead() {
                return self.closure(start, false, false);
            }
            self.pos = saved_pos;
            self.restore_token_splits(saved_splits);
            self.diagnostics.truncate(saved_diagnostics);
            self.depth = saved_depth;

            // Non-JSX TypeScript angle-bracket assertion: `<T>value`.
            self.expect("<")?;
            self.ty()?;
            self.type_close()?;
            self.lines();
            return self.expr(16);
        }
        if self.eat("launch") {
            self.lines();
            let call = self.expr(16)?;
            return Ok(Expr::new(ExprKind::Launch(Box::new(call)), self.end(start)));
        }
        if self.eat("concurrent") {
            if self.eat("async") {
                return self.error("`concurrent async` is redundant; concurrent functions are launched asynchronously and return a Promise");
            }
            let function = self.eat("function");
            let closure = self.closure(start, false, function)?;
            let span = closure.span;
            return match closure.kind {
                ExprKind::Closure(body) => Ok(Expr::new(ExprKind::ConcurrentClosure(body), span)),
                _ => unreachable!("non-async closure parser returned a non-closure expression"),
            };
        }
        if self.eat("async") {
            let function = self.eat("function");
            return self.closure(start, true, function);
        }
        if self.eat("function") {
            return self.closure(start, false, true);
        }
        if self.arrow_ahead() {
            return self.closure(start, false, false);
        }
        if matches!(self.text(), "typeof" | "delete" | "void" | "...") {
            return self.error(format!(
                "`{}` is recognized TypeScript syntax but is not lowered yet",
                self.text()
            ));
        }
        if self.eat("await") {
            let value = self.expr(16)?;
            return Ok(Expr::new(ExprKind::Await(Box::new(value)), self.end(start)));
        }
        if self.at("++") || self.at("--") {
            let operator = if self.eat("++") {
                BinaryOp::Add
            } else {
                self.bump();
                BinaryOp::Subtract
            };
            let target = self.expr(18)?;
            self.assignment_target(&target)?;
            return Ok(Expr::new(
                ExprKind::Update {
                    target: Box::new(target),
                    operator,
                    value: Box::new(Expr::new(ExprKind::Number(1.0), self.end(start))),
                    postfix: false,
                },
                self.end(start),
            ));
        }
        if self.at("-") || self.at("!") || self.at("~") || self.at("+") {
            let op = self.text();
            self.bump();
            self.lines();
            let value = self.expr(16)?;
            if op == "+" {
                return Ok(value);
            }
            let operator = match op {
                "-" => UnaryOp::Negate,
                "!" => UnaryOp::Not,
                _ => UnaryOp::Complement,
            };
            return Ok(Expr::new(
                ExprKind::Unary {
                    operator,
                    operand: Box::new(value),
                },
                self.end(start),
            ));
        }
        if self.eat("new") {
            let name = self.ident()?;
            let mut receiver = Expr::new(ExprKind::Name(name), self.end(start));
            while self.eat(".") {
                let name = self.ident()?;
                receiver = Expr::new(
                    ExprKind::Invoke(InvokeExpr {
                        receiver: Some(Box::new(receiver)),
                        name,
                        args: Vec::new(),
                        syntax: InvokeSyntax::Getter,
                    }),
                    self.end(start),
                );
            }
            if self.at("<") {
                self.type_arguments()?;
            }
            let args = self.args()?;
            return Ok(invoke(receiver, "new", args, self.end(start)));
        }
        if self.eat("super") {
            let method = if self.eat(".") {
                Some(self.ident()?)
            } else {
                None
            };
            let syntax = if self.at("(") {
                InvokeSyntax::Method
            } else {
                InvokeSyntax::Getter
            };
            let args = if syntax == InvokeSyntax::Method {
                self.args()?
            } else {
                Vec::new()
            };
            return Ok(Expr::new(
                ExprKind::Super(SuperExpr {
                    method,
                    args,
                    syntax,
                }),
                self.end(start),
            ));
        }
        if self.eat("(") {
            self.lines();
            let value = self.expr(0)?;
            self.lines();
            self.expect(")")?;
            return Ok(Expr::new(
                ExprKind::Grouped(Box::new(value)),
                self.end(start),
            ));
        }
        if self.eat("[") {
            self.lines();
            let mut items = Vec::new();
            while !self.at("]") {
                let item_start = self.peek().span.start;
                if self.eat(",") {
                    items.push(Expr::new(ExprKind::Null, Span::empty(item_start)));
                    self.lines();
                    continue;
                }
                let item = if self.eat("...") {
                    self.lines();
                    let value = self.expr(1)?;
                    Expr::new(ExprKind::Spread(Box::new(value)), self.end(item_start))
                } else {
                    self.expr(0)?
                };
                items.push(item);
                self.lines();
                if !self.eat(",") {
                    break;
                }
                self.lines();
            }
            self.expect("]")?;
            return Ok(Expr::new(ExprKind::List(items), self.end(start)));
        }
        if self.eat("{") {
            self.lines();
            let mut entries = Vec::new();
            while !self.at("}") {
                if self.eat("...") {
                    let spread_start = self.tokens[self.pos.saturating_sub(1)].span.start;
                    self.lines();
                    let value = self.expr(1)?;
                    let span = Span::new(spread_start, value.span.end);
                    entries.push(MapEntry {
                        key: Expr::new(ExprKind::Null, span),
                        value,
                        span,
                        spread: true,
                    });
                } else {
                    let key = if self.eat("[") {
                        let key = self.expr(0)?;
                        self.expect("]")?;
                        key
                    } else if self.peek().kind == Kind::Word {
                        let n = self.ident()?;
                        Expr::new(ExprKind::String(n.value.into_bytes()), n.span)
                    } else if self.peek().kind == Kind::String {
                        self.string_expr()?
                    } else {
                        self.prefix()?
                    };
                    let value = if self.eat(":") {
                        self.lines();
                        self.expr(0)?
                    } else if let ExprKind::String(bytes) = &key.kind {
                        let target = Expr::new(ExprKind::Name(Spanned::new(
                            String::from_utf8_lossy(bytes).into_owned(), key.span)), key.span);
                        if self.eat("=") {
                            self.lines();
                            let value = self.expr(1)?;
                            let span = target.span.join(value.span);
                            Expr::new(ExprKind::Assign { target: Box::new(target), value: Box::new(value) }, span)
                        } else { target }
                    } else {
                        return self.error("expected `:` after object key");
                    };
                    entries.push(MapEntry {
                        span: key.span.join(value.span),
                        key,
                        value,
                        spread: false,
                    });
                }
                self.lines();
                if !self.eat(",") {
                    break;
                }
                self.lines();
            }
            self.expect("}")?;
            return Ok(Expr::new(ExprKind::Map(entries), self.end(start)));
        }
        if self.peek().kind == Kind::String {
            return self.string_expr();
        }
        if self.peek().kind == Kind::Number {
            let text = self.text();
            self.bump();
            let n = if text.starts_with("0x") || text.starts_with("0X") {
                u64::from_str_radix(&text[2..], 16).ok().map(|n| n as f64)
            } else if text.starts_with("0b") || text.starts_with("0B") {
                u64::from_str_radix(&text[2..], 2).ok().map(|n| n as f64)
            } else if text.starts_with("0o") || text.starts_with("0O") {
                u64::from_str_radix(&text[2..], 8).ok().map(|n| n as f64)
            } else {
                text.parse::<f64>().ok()
            };
            return n
                .map(|n| Expr::new(ExprKind::Number(n), self.end(start)))
                .ok_or_else(|| {
                    self.diagnostics.push(Diagnostic::error(
                        "NAPALM-P002",
                        "invalid number",
                        self.end(start),
                    ));
                });
        }
        let name = self.ident()?;
        let kind = match name.value.as_str() {
            "true" => ExprKind::Bool(true),
            "false" => ExprKind::Bool(false),
            "null" => ExprKind::Null,
            "this" => ExprKind::This,
            "construct" | "is" | "in" | "of" | "let" | "const" | "return" | "throw" | "class"
            | "concurrent" | "launch"
            | "import" | "export" | "else" | "undefined" => {
                return self.error("unexpected keyword in expression");
            }
            _ => ExprKind::Name(name),
        };
        Ok(Expr::new(kind, self.end(start)))
    }
    fn string_expr(&mut self) -> Result<Expr> {
        let token = self.bump();
        let bytes = self.source.as_bytes();
        let template = bytes[token.span.start] == b'`';
        let mut i = token.span.start + 1;
        let end = token.span.end - 1;
        let mut segment = i;
        let mut parts = Vec::new();
        while i < end {
            if bytes[i] == b'\\' {
                i += 2;
                continue;
            }
            if template && bytes[i..].starts_with(b"${") {
                parts.push(StringPart::Bytes(self.decode(segment, i)?));
                let close = interpolation_end(bytes, i + 2, 1).ok_or(())?;
                let mut nested = Parser::new(self.source, i + 2, close, self.arena);
                nested.lines();
                let value = nested.expr(0);
                nested.lines();
                if nested.peek().kind != Kind::Eof {
                    let _: Result<()> = nested.error("unexpected tokens in template expression");
                }
                self.immutable.extend(nested.immutable);
                self.diagnostics.extend(nested.diagnostics);
                parts.push(StringPart::Expression(value?));
                i = close + 1;
                segment = i;
            } else {
                i += 1;
            }
        }
        let tail = self.decode(segment, end)?;
        let kind = if parts.is_empty() {
            ExprKind::String(tail)
        } else {
            parts.push(StringPart::Bytes(tail));
            ExprKind::InterpolatedString(parts)
        };
        Ok(Expr::new(kind, token.span))
    }
    fn decode(&mut self, start: usize, end: usize) -> Result<Vec<u8>> {
        let mut out = String::new();
        let mut chars = self.source[start..end].chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            let Some(c) = chars.next() else {
                return self.error("unterminated escape");
            };
            match c {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '0' => out.push('\0'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'v' => out.push('\u{b}'),
                '\n' => {}
                '\r' => {
                    if chars.as_str().starts_with('\n') {
                        chars.next();
                    }
                }
                'x' | 'u' => {
                    let digits: String = if c == 'u' && chars.as_str().starts_with('{') {
                        chars.next();
                        let mut digits = String::new();
                        loop {
                            match chars.next() {
                                Some('}') => break,
                                Some(c) => digits.push(c),
                                None => return self.error("unterminated Unicode escape"),
                            }
                        }
                        digits
                    } else {
                        chars.by_ref().take(if c == 'x' { 2 } else { 4 }).collect()
                    };
                    let Some(code) = u32::from_str_radix(&digits, 16)
                        .ok()
                        .and_then(char::from_u32)
                    else {
                        return self.error("invalid Unicode escape");
                    };
                    out.push(code);
                }
                '\\' | '\'' | '"' | '`' | '$' => out.push(c),
                _ => return self.error(format!("unknown escape `\\{c}`")),
            }
        }
        Ok(out.into_bytes())
    }

    fn heritage_expr(&mut self) -> Result<Expr> {
        if self.peek().kind != Kind::Word {
            return self.expr(17);
        }
        let start = self.peek().span.start;
        let name = self.ident()?;
        let mut value = Expr::new(ExprKind::Name(name), self.end(start));
        while self.eat(".") {
            let name = self.ident()?;
            value = Expr::new(
                ExprKind::Invoke(InvokeExpr {
                    receiver: Some(Box::new(value)),
                    name,
                    args: Vec::new(),
                    syntax: InvokeSyntax::Getter,
                }),
                self.end(start),
            );
        }
        if self.at("<") {
            self.type_arguments()?;
        }
        if self.at("(") {
            let args = self.args()?;
            value = invoke(value, "call", args, self.end(start));
        }
        Ok(value)
    }

    fn class(&mut self, start: usize, is_foreign: bool, default_export: bool) -> Result<Vec<Stmt>> {
        let name = if default_export && (self.at("{") || self.at("extends")) {
            Spanned::new("<default>".into(), self.peek().span)
        } else {
            self.ident()?
        };
        self.type_parameters()?;
        let superclass = if self.eat("extends") {
            Some(self.heritage_expr()?)
        } else {
            None
        };
        if self.eat("implements") {
            loop {
                self.ty()?;
                if !self.eat(",") {
                    break;
                }
            }
        }
        self.lines();
        self.expect("{")?;
        self.separators();
        let mut methods = Vec::new();
        let mut fields = Vec::new();
        let mut initializers = Vec::new();
        let mut parameter_initializers = Vec::new();
        let mut static_initializers = Vec::new();
        let mut constructor = None;
        while !self.at("}") && self.peek().kind != Kind::Eof {
            let member_start = self.peek().span.start;
            let mut is_static = false;
            let mut foreign = false;
            let mut async_ = false;
            let mut type_only_member = false;
            loop {
                if self.eat("static") {
                    is_static = true;
                } else if self.eat("foreign") {
                    foreign = true;
                } else if self.eat("async") {
                    async_ = true;
                } else if self.eat("abstract") || self.eat("declare") {
                    type_only_member = true;
                } else if self.eat("public")
                    || self.eat("private")
                    || self.eat("protected")
                    || self.eat("readonly")
                    || self.eat("override")
                {
                } else {
                    break;
                }
            }

            // ES/TypeScript static initialization blocks map naturally to the
            // synthetic class initializer already used for static field values.
            if is_static && self.at("{") {
                if foreign || async_ || type_only_member {
                    return self.error("a static block cannot be foreign, async, abstract or declare");
                }
                let block = self.block()?;
                let StmtKind::Block(statements) = block.kind else { unreachable!() };
                static_initializers.extend(statements);
                self.separators();
                continue;
            }
            let access = if (self.at("get") || self.at("set")) && self.next_text() != "(" {
                let t = self.text();
                self.bump();
                Some(t)
            } else {
                None
            };
            let member = self.ident()?;
            let is_constructor = member.value == "constructor";
            let _optional_or_definite = self.eat("?") || self.eat("!");
            self.type_parameters()?;
            if self.at("(") {
                let (params, parameter_properties) = self.params_with_properties(is_constructor)?;
                let body_params = params.clone();
                self.annotation()?;
                let signature = match access {
                    Some("get") if params.is_empty() => MethodSignatureAst::Getter { name: member },
                    Some("set") if params.len() == 1 => MethodSignatureAst::Setter {
                        name: member,
                        param: params[0].clone(),
                    },
                    Some(_) => {
                        return self.error("a getter takes no parameters; a setter takes one");
                    }
                    None => MethodSignatureAst::Method {
                        name: if is_constructor {
                            Spanned::new("new".into(), member.span)
                        } else {
                            member
                        },
                        params,
                    },
                };
                if is_constructor && (is_static || async_ || access.is_some()) {
                    return self.error("a constructor cannot be static, async or an accessor");
                }
                if is_constructor && constructor.is_some() {
                    return self.error("a class has only one constructor implementation");
                }
                if !parameter_properties.is_empty() && !is_constructor {
                    return self.error("parameter properties are only valid on constructors");
                }
                if type_only_member || (!foreign && !self.at("{")) {
                    if is_constructor {
                        return self.error("a constructor must have an implementation");
                    }
                    self.boundary()?;
                    self.separators();
                    continue;
                }
                if is_constructor {
                    for (property, annotation) in &parameter_properties {
                        let value = Expr::new(ExprKind::Name(property.clone()), property.span);
                        let (getter, setter, assignment) =
                            synthesize_class_property(property, false, value);
                        fields.push(ClassFieldDecl {
                            storage: class_storage_ident(property, false),
                            annotation: annotation.clone(),
                            initializer: None,
                            is_static: false,
                        });
                        methods.push(getter);
                        methods.push(setter);
                        parameter_initializers.push(assignment);
                    }
                }
                let body = if foreign {
                    self.boundary()?;
                    None
                } else {
                    Some(self.function_body(body_params, member_start)?)
                };
                if is_constructor {
                    constructor = Some(methods.len());
                }
                methods.push(MethodDecl {
                    signature,
                    is_async: async_,
                    is_static,
                    is_foreign: foreign,
                    is_constructor,
                    attributes: Vec::new(),
                    body,
                    span: self.end(member_start),
                });
            } else {
                if access.is_some() || async_ || foreign || is_constructor {
                    return self.error("expected a method parameter list");
                }
                let annotation = self.variable_annotation()?;
                let initializer = if self.eat("=") {
                    self.lines();
                    Some(self.expr(0)?)
                } else {
                    None
                };
                self.boundary()?;
                if type_only_member {
                    self.separators();
                    continue;
                }
                fields.push(ClassFieldDecl {
                    storage: class_storage_ident(&member, is_static),
                    annotation,
                    initializer: initializer.clone(),
                    is_static,
                });
                let initializer = initializer.unwrap_or_else(|| Expr::new(ExprKind::Null, member.span));
                let (getter, setter, assignment) =
                    synthesize_class_property(&member, is_static, initializer);
                methods.push(getter);
                methods.push(setter);
                if is_static {
                    static_initializers.push(assignment);
                } else {
                    initializers.push(assignment);
                }
            }
            self.separators();
        }
        self.expect("}")?;
        if !initializers.is_empty() || !parameter_initializers.is_empty() {
            if constructor.is_none() {
                constructor = Some(methods.len());
                let statements = if superclass.is_some() {
                    vec![Stmt {
                        kind: StmtKind::Expression(Expr::new(
                            ExprKind::Super(SuperExpr {
                                method: None,
                                args: Vec::new(),
                                syntax: InvokeSyntax::Method,
                            }),
                            name.span,
                        )),
                        span: name.span,
                    }]
                } else {
                    Vec::new()
                };
                methods.push(MethodDecl {
                    signature: MethodSignatureAst::Method {
                        name: Spanned::new("new".into(), name.span),
                        params: Vec::new(),
                    },
                    is_constructor: true,
                    is_static: false,
                    is_foreign: false,
                    is_async: false,
                    attributes: Vec::new(),
                    body: Some(FunctionBody {
                        params: Vec::new(),
                        kind: FunctionBodyKind::Statements(statements),
                        span: self.end(start),
                    }),
                    span: self.end(start),
                });
            }
            let method: &mut MethodDecl = &mut methods[constructor.unwrap()];
            if let Some(FunctionBody {
                kind: FunctionBodyKind::Statements(statements),
                ..
            }) = &mut method.body
            {
                let offset = if superclass.is_some() {
                    match statements.iter().position(|s| matches!(&s.kind, StmtKind::Expression(Expr { kind: ExprKind::Super(_), .. }))) {
                        Some(i) => i+1,
                        None => return self.error("a derived constructor with fields must call super() directly before initializing fields"),
                    }
                } else {
                    0
                };
                let mut constructor_initializers = Vec::with_capacity(
                    parameter_initializers.len() + initializers.len(),
                );
                constructor_initializers.append(&mut parameter_initializers);
                constructor_initializers.append(&mut initializers);
                statements.splice(offset..offset, constructor_initializers);
            }
        }
        let has_static_init = !static_initializers.is_empty();
        if has_static_init {
            methods.push(MethodDecl {
                signature: MethodSignatureAst::Method {
                    name: Spanned::new("<initialize-fields>".into(), name.span),
                    params: Vec::new(),
                },
                is_constructor: false,
                is_static: true,
                is_async: false,
                is_foreign: false,
                attributes: Vec::new(),
                body: Some(FunctionBody {
                    params: Vec::new(),
                    kind: FunctionBodyKind::Statements(static_initializers),
                    span: self.end(start),
                }),
                span: self.end(start),
            });
        }
        let class = ClassDecl {
            name: name.clone(),
            superclass,
            is_foreign,
            attributes: Vec::new(),
            fields,
            methods,
        };
        let mut out = vec![Stmt {
            kind: StmtKind::Class(class),
            span: self.end(start),
        }];
        if has_static_init {
            out.push(Stmt {
                kind: StmtKind::Expression(invoke(
                    Expr::new(ExprKind::Name(name), self.end(start)),
                    "<initialize-fields>",
                    Vec::new(),
                    self.end(start),
                )),
                span: self.end(start),
            });
        }
        Ok(out)
    }
}


/// Convert a TypeScript-style class property into the existing Napalm-compatible
/// AST representation: fixed storage plus ordinary getter/setter methods. This
/// keeps all field layout, GC and JIT machinery unchanged while the surface
/// syntax becomes Napalm/TypeScript-like.
fn class_storage_ident(property: &Ident, is_static: bool) -> Ident {
    Spanned::new(
        if is_static {
            format!("__{}", property.value)
        } else {
            format!("_{}", property.value)
        },
        property.span,
    )
}

fn synthesize_class_property(
    member: &Ident,
    is_static: bool,
    initializer: Expr,
) -> (MethodDecl, MethodDecl, Stmt) {
    let storage = Spanned::new(
        format!("{}{}", if is_static { "__" } else { "_" }, member.value),
        member.span,
    );
    let field = Expr::new(
        if is_static {
            ExprKind::StaticField(storage)
        } else {
            ExprKind::Field(storage)
        },
        member.span,
    );
    let param = Spanned::new("<value>".into(), member.span);
    let read = FunctionBody {
        params: Vec::new(),
        kind: FunctionBodyKind::Expression(Box::new(field.clone())),
        span: member.span,
    };
    let write = FunctionBody {
        params: Vec::new(),
        kind: FunctionBodyKind::Expression(Box::new(Expr::new(
            ExprKind::Assign {
                target: Box::new(field.clone()),
                value: Box::new(Expr::new(ExprKind::Name(param.clone()), member.span)),
            },
            member.span,
        ))),
        span: member.span,
    };
    let getter = MethodDecl {
        signature: MethodSignatureAst::Getter {
            name: member.clone(),
        },
        body: Some(read),
        is_static,
        is_async: false,
        is_foreign: false,
        is_constructor: false,
        attributes: Vec::new(),
        span: member.span,
    };
    let setter = MethodDecl {
        signature: MethodSignatureAst::Setter {
            name: member.clone(),
            param,
        },
        body: Some(write),
        is_static,
        is_async: false,
        is_foreign: false,
        is_constructor: false,
        attributes: Vec::new(),
        span: member.span,
    };
    let assignment = Stmt {
        kind: StmtKind::Expression(Expr::new(
            ExprKind::Assign {
                target: Box::new(field),
                value: Box::new(initializer),
            },
            member.span,
        )),
        span: member.span,
    };
    (getter, setter, assignment)
}

fn invoke(receiver: Expr, name: &str, args: Vec<Expr>, span: Span) -> Expr {
    Expr::new(
        ExprKind::Invoke(InvokeExpr {
            receiver: Some(Box::new(receiver)),
            name: Spanned::new(name.into(), span),
            args,
            syntax: InvokeSyntax::Method,
        }),
        span,
    )
}
fn infix(op: &str) -> Option<(u8, BinaryOp)> {
    use BinaryOp::*;
    Some(match op {
        "|" => (5, BitOr),
        "^" => (6, BitXor),
        "&" => (7, BitAnd),
        "==" | "===" => (8, Equal),
        "!=" | "!==" => (8, NotEqual),
        "<" => (9, Less),
        "<=" => (9, LessEqual),
        ">" => (9, Greater),
        ">=" => (9, GreaterEqual),
        "instanceof" => (9, Is),
        "<<" => (10, ShiftLeft),
        ">>" => (10, ShiftRight),
        "+" => (11, Add),
        "-" => (11, Subtract),
        "*" => (12, Multiply),
        "/" => (12, Divide),
        "%" => (12, Modulo),
        _ => return None,
    })
}

// Const is a binding rule, not a type check. Resolve names lexically, including
// nested closures and shadowing, before any AST lowering or optimizations.
fn validate_bindings(
    module: &ModuleAst,
    immutable: &HashSet<Span>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    struct Validator<'a> {
        immutable: &'a HashSet<Span>,
        scopes: Vec<HashMap<String, bool>>,
        diagnostics: &'a mut Vec<Diagnostic>,
    }
    impl Validator<'_> {
        fn declare(&mut self, name: &Ident) {
            let scope = self.scopes.last_mut().unwrap();
            if scope
                .insert(name.value.clone(), self.immutable.contains(&name.span))
                .is_some()
            {
                self.diagnostics.push(Diagnostic::error(
                    "NAPALM-B001",
                    format!("duplicate binding '{}'", name.value),
                    name.span,
                ));
            }
        }
        fn statements(&mut self, statements: &[Stmt]) {
            for s in statements {
                match &s.kind {
                    StmtKind::Variable { name, .. } => self.declare(name),
                    StmtKind::Class(c) => self.declare(&c.name),
                    StmtKind::Import(i) => {
                        for b in &i.bindings {
                            self.declare(&b.local);
                        }
                    }
                    _ => {}
                }
            }
            for s in statements {
                self.stmt(s);
            }
        }
        fn body(&mut self, body: &FunctionBody, params: &[Ident]) {
            self.scopes.push(HashMap::default());
            for p in params {
                self.declare(p);
            }
            match &body.kind {
                FunctionBodyKind::Expression(e) => self.expr(e),
                FunctionBodyKind::Statements(s) => self.statements(s),
            }
            self.scopes.pop();
        }
        fn stmt(&mut self, s: &Stmt) {
            match &s.kind {
                StmtKind::Variable { initializer, .. } => {
                    if let Some(e) = initializer {
                        self.expr(e);
                    }
                }
                StmtKind::Class(c) => {
                    if let Some(e) = &c.superclass {
                        self.expr(e);
                    }
                    for m in &c.methods {
                        if let Some(body) = &m.body {
                            let params = match &m.signature {
                                MethodSignatureAst::Method { params, .. }
                                | MethodSignatureAst::SubscriptGetter { params } => params.clone(),
                                MethodSignatureAst::Setter { param, .. }
                                | MethodSignatureAst::InfixOperator { param, .. } => {
                                    vec![param.clone()]
                                }
                                MethodSignatureAst::SubscriptSetter { params, value } => {
                                    let mut p = params.clone();
                                    p.push(value.clone());
                                    p
                                }
                                _ => Vec::new(),
                            };
                            self.body(body, &params);
                        }
                    }
                }
                StmtKind::Block(s) => {
                    self.scopes.push(HashMap::default());
                    self.statements(s);
                    self.scopes.pop();
                }
                StmtKind::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    self.expr(condition);
                    self.stmt(then_branch);
                    if let Some(s) = else_branch {
                        self.stmt(s);
                    }
                }
                StmtKind::While {
                    condition,
                    body,
                    increment,
                } => {
                    self.expr(condition);
                    self.stmt(body);
                    if let Some(e) = increment {
                        self.expr(e);
                    }
                }
                StmtKind::DoWhile { body, condition } => {
                    self.stmt(body);
                    self.expr(condition);
                }
                StmtKind::For {
                    name,
                    iterable,
                    body,
                } => {
                    self.expr(iterable);
                    self.scopes.push(HashMap::default());
                    self.declare(name);
                    self.stmt(body);
                    self.scopes.pop();
                }
                StmtKind::Defer {
                    body,
                    error_binding,
                    ..
                } => {
                    self.scopes.push(HashMap::default());
                    if let Some(n) = error_binding {
                        self.declare(n);
                    }
                    self.stmt(body);
                    self.scopes.pop();
                }
                StmtKind::Try {
                    body,
                    catch,
                    finally_branch,
                } => {
                    self.stmt(body);
                    if let Some(c) = catch {
                        self.scopes.push(HashMap::default());
                        if let Some(n) = &c.binding {
                            self.declare(n);
                        }
                        self.stmt(&c.body);
                        self.scopes.pop();
                    }
                    if let Some(s) = finally_branch {
                        self.stmt(s);
                    }
                }
                StmtKind::Expression(e) | StmtKind::Throw(e) => self.expr(e),
                StmtKind::Return(Some(e)) => self.expr(e),
                _ => {}
            }
        }
        fn assignment(&mut self, target: &Expr) {
            match &target.kind {
                ExprKind::Name(n) => {
                    if self.scopes.iter().rev().find_map(|scope| scope.get(&n.value)).copied() == Some(true) {
                        self.diagnostics.push(Diagnostic::error("NAPALM-B002",
                            format!("cannot assign to immutable binding '{}'", n.value), n.span));
                    }
                }
                ExprKind::List(items) => for item in items { self.assignment(item); },
                ExprKind::Map(entries) => for entry in entries { self.expr(&entry.key); self.assignment(&entry.value); },
                ExprKind::Spread(inner) => self.assignment(inner),
                ExprKind::Assign { target, value } => { self.assignment(target); self.expr(value); }
                _ => self.expr(target),
            }
        }
        fn expr(&mut self, e: &Expr) {
            match &e.kind {
                ExprKind::Assign { target, value }
                | ExprKind::Update { target, value, .. }
                | ExprKind::LogicalAssign { target, value, .. } => {
                    self.assignment(target);
                    self.expr(value);
                }
                ExprKind::Closure(b) | ExprKind::AsyncClosure(b) | ExprKind::ConcurrentClosure(b) => self.body(b, &b.params),
                ExprKind::Invoke(i) => {
                    if let Some(e) = &i.receiver {
                        self.expr(e);
                    }
                    for a in &i.args {
                        self.expr(a);
                    }
                }
                ExprKind::Subscript { receiver, args } => {
                    self.expr(receiver);
                    for a in args {
                        self.expr(a);
                    }
                }
                ExprKind::Super(s) => {
                    for a in &s.args {
                        self.expr(a);
                    }
                }
                ExprKind::List(items) => {
                    for e in items {
                        self.expr(e);
                    }
                }
                ExprKind::Map(entries) => {
                    for e in entries {
                        if matches!(&e.value.kind, ExprKind::Assign { target, .. } if target.span == e.key.span) {
                            self.diagnostics.push(Diagnostic::error("NAPALM-P001",
                                "shorthand defaults are only valid in destructuring patterns", e.value.span));
                        }
                        self.expr(&e.key);
                        self.expr(&e.value);
                    }
                }
                ExprKind::Unary { operand, .. }
                | ExprKind::Grouped(operand)
                | ExprKind::Await(operand)
                | ExprKind::Launch(operand)
                | ExprKind::DestructureCheck { source: operand, .. }
                | ExprKind::Spread(operand) => self.expr(operand),
                ExprKind::DestructureGet { source: left, key: right }
            | ExprKind::DestructureRest { source: left, key: right }
            | ExprKind::Binary { left, right, .. } | ExprKind::Logical { left, right, .. } => {
                    self.expr(left);
                    self.expr(right);
                }
                ExprKind::Conditional {
                    condition,
                    then_expr,
                    else_expr,
                } => {
                    self.expr(condition);
                    self.expr(then_expr);
                    self.expr(else_expr);
                }
                ExprKind::InterpolatedString(parts) => {
                    for p in parts {
                        if let StringPart::Expression(e) = p {
                            self.expr(e);
                        }
                    }
                }
                ExprKind::OptionalChain { base, segments } => {
                    self.expr(base);
                    for segment in segments {
                        match segment {
                            OptionalChainSegment::Property { call_args, .. } => {
                                if let Some(args) = call_args {
                                    for arg in args { self.expr(arg); }
                                }
                            }
                            OptionalChainSegment::Subscript { key, .. } => self.expr(key),
                            OptionalChainSegment::Call { args, .. } => {
                                for arg in args { self.expr(arg); }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut v = Validator {
        immutable,
        scopes: vec![HashMap::default()],
        diagnostics,
    };
    v.statements(&module.statements);
    let mut exported = HashSet::default();
    for (local, public) in module.exports.as_ref().unwrap() {
        if !v.scopes[0].contains_key(&local.value) {
            v.diagnostics.push(Diagnostic::error(
                "NAPALM-B003",
                format!("cannot export undefined binding '{}'", local.value),
                local.span,
            ));
        }
        if !exported.insert(&public.value) {
            v.diagnostics.push(Diagnostic::error(
                "NAPALM-B004",
                format!("duplicate export '{}'", public.value),
                public.span,
            ));
        }
    }
}

#[cfg(test)]
mod napalm_syntax_tests {
    use super::*;

    #[test]
    fn speculative_types_restore_split_shift_tokens() {
        let source = "<Outer<Inner>> + 1";
        let arena = Bump::new();
        let mut parser = Parser::new(source, 0, source.len(), &arena);
        let original: Vec<_> = parser.tokens.iter().map(|t| (t.kind, t.span)).collect();
        // Parsing the nested type splits >>, but the following + rejects a call.
        for _ in 0..3 {
            assert!(!parser.try_call_type_arguments());
            assert_eq!(parser.pos, 0);
            assert!(parser.diagnostics.is_empty());
            assert_eq!(parser.tokens.iter().map(|t| (t.kind, t.span)).collect::<Vec<_>>(), original);
        }

        let module = assert_parses(r#"
            const id = <T>(x: T) => x;
            const first = id<Array<number>>([1]);
            const second = <Array<number>>first;
            const text = `nested ${id<Array<number>>(second)}`;
            const shifted = 8 >> 1;
        "#);
        assert_eq!(module.statements.len(), 5);
    }

    fn assert_parses(source: &str) -> ModuleAst {
        let parsed = parse("syntax.np", source);
        assert!(
            !parsed.has_errors(),
            "Napalm frontend diagnostics: {:#?}",
            parsed.diagnostics
        );
        parsed.module
    }

    #[test]
    fn parses_erased_types_generics_and_modern_class_syntax() {
        let module = assert_parses(r#"
            import type { ExternalType } from "types";
            import { type OtherType, runtimeValue } from "runtime";

            export type Id<T> = T;
            export type { ExternalType as PublicExternal };

            interface Runnable<T> {
                run(value: T): T;
            }

            declare function ambient<T>(value: T): T;
            declare const AMBIENT_VERSION: string;

            abstract class Base<T> {
                abstract run(value: T): T;
                declare cached!: T;

                static {
                    let initialized: bool = true;
                    System.print(initialized);
                }
            }

            class Box<T> extends Base<T> implements Runnable<T> {
                value!: T;
                maybe?: T;

                constructor(public item: T, private readonly id: number) {
                    super();
                    this.value = item;
                }

                run<U>(value: T): T {
                    return value;
                }

                static {
                    System.print("Box initialized");
                }
            }

            function identity<T>(value: T): T;
            function identity<T>(value: T): T {
                return value;
            }

            const genericArrow = <T>(value: T): T => value;
            const typedArrow = (value: number): number => value;
            const a = identity<number>(1);
            const b = new Box<number>(a, 7);
            const c = b.item! satisfies number;
            const d = <number>a;
        "#);

        assert!(!module.statements.is_empty());
    }

    #[test]
    fn parses_js_sugar_with_single_evaluation_shapes() {
        let module = assert_parses(r#"
            const source = [1, 2, 3];
            const xs = [0, ...source, 4];
            const base = {"a": 1, "b": 2};
            const merged = {...base, c: 3};

            const [first, second] = xs;
            const {a, b: renamed} = merged;

            let maybe = null;
            let value = maybe?.child?.run(first)?.result ?? renamed;
            let called = maybe?.(first);
            let powered = 2 ** 3 ** 2;

            let assigned = null;
            assigned ??= renamed;
            assigned &&= powered;
            assigned ||= 1;

            for (const [left, right] of [[1, 2], [3, 4]]) {
                System.print(left + right);
            }

            for (const key in merged) {
                System.print(key);
            }

            let i = 0;
            do {
                i += 1;
            } while (i < 2);

            function acceptsSpreadSyntax(a, b) {
                return a + b;
            }
            acceptsSpreadSyntax(...xs);
        "#);

        assert!(!module.statements.is_empty());
    }

    #[test]
    fn retains_variable_type_annotations_in_the_engine_ast() {
        let module = assert_parses(r#"
            let count: number = 1;
            let label: string = "napalm";
            let enabled: boolean = true;
            let values: number[] = [1, 2, 3];
            let escapeHatch: any = count;
            let inferred = 42;
        "#);

        let expected = [
            TypeAnnotationKind::Num,
            TypeAnnotationKind::String,
            TypeAnnotationKind::Bool,
            TypeAnnotationKind::List(Some(Box::new(TypeAnnotation::new(
                TypeAnnotationKind::Num,
                Span::default(),
            )))),
            TypeAnnotationKind::Any,
        ];

        for (statement, expected_kind) in module.statements.iter().take(5).zip(expected) {
            let StmtKind::Variable { annotation: Some(annotation), .. } = &statement.kind else {
                panic!("expected retained variable type annotation");
            };
            match (&annotation.kind, expected_kind) {
                (TypeAnnotationKind::List(Some(actual)), TypeAnnotationKind::List(Some(expected))) => {
                    assert_eq!(actual.kind, expected.kind);
                }
                (actual, expected) => assert_eq!(actual, &expected),
            }
        }

        let StmtKind::Variable { annotation, .. } = &module.statements[5].kind else {
            panic!("expected inferred variable declaration");
        };
        assert!(annotation.is_none(), "inferred types are intentionally computed by lowering");
    }

    #[test]
    fn retains_interfaces_closed_type_aliases_and_class_field_types() {
        let module = assert_parses(r#"
            interface OpenPoint {
                x: number;
                payload: any;
            }

            type ClosedPoint = {
                x: number;
                payload: any;
            };

            class Point {
                x: number = 1;
                payload: any = null;
                static count: number = 0;
            }
        "#);

        assert_eq!(module.interfaces.len(), 1);
        assert_eq!(module.interfaces[0].name.value, "OpenPoint");
        assert!(matches!(
            &module.interfaces[0].fields[1].annotation.kind,
            TypeAnnotationKind::Any
        ));

        assert_eq!(module.type_aliases.len(), 1);
        assert_eq!(module.type_aliases[0].name.value, "ClosedPoint");
        let TypeAnnotationKind::Object(fields) = &module.type_aliases[0].annotation.kind else {
            panic!("closed object type alias should retain its structural fields");
        };
        assert_eq!(fields.len(), 2);
        assert!(matches!(&fields[0].annotation.kind, TypeAnnotationKind::Num));
        assert!(matches!(&fields[1].annotation.kind, TypeAnnotationKind::Any));

        let class = module
            .statements
            .iter()
            .find_map(|statement| match &statement.kind {
                StmtKind::Class(class) => Some(class),
                _ => None,
            })
            .expect("Point class");
        assert_eq!(class.fields.len(), 3);
        assert!(matches!(
            class.fields[0].annotation.as_ref().map(|annotation| &annotation.kind),
            Some(TypeAnnotationKind::Num)
        ));
        assert!(matches!(
            class.fields[1].annotation.as_ref().map(|annotation| &annotation.kind),
            Some(TypeAnnotationKind::Any)
        ));
        assert!(class.fields[2].is_static);
    }

    #[test]
    fn recognizes_runtime_typescript_features_that_still_need_lowering() {
        for source in [
            "let x = value >>> shift;",
            "let x = typeof value;",
        ] {
            let parsed = parse("future-syntax.np", source);
            assert!(parsed.has_errors(), "feature unexpectedly lowered: {source}");
            assert!(parsed
                .diagnostics
                .iter()
                .any(|d| d.message.contains("recognized TypeScript syntax")));
        }

        let rest = parse("future-syntax.np", "function f(...args) { return args; }");
        assert!(rest.has_errors());
        assert!(rest
            .diagnostics
            .iter()
            .any(|d| d.message.contains("rest parameters are not implemented yet")));
    }
}

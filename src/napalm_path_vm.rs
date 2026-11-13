//! Cross-platform lexical paths and RFC 3986/8089 file URI helpers.
//!
//! This module does not resolve symlinks, query the filesystem, or turn a URI
//! path into a local path implicitly. `Fs.realPath` remains the filesystem-aware
//! canonicalization API.

use super::*;
use std::env;

const MAX_PATH_INPUT: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Flavor {
    Posix,
    Windows,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathRoot {
    Relative,
    Posix,
    WinRooted,
    Drive { letter: char, absolute: bool },
    Unc { server: String, share: String },
}

#[derive(Clone, Debug)]
struct LexPath {
    flavor: Flavor,
    root: PathRoot,
    components: Vec<String>,
}

impl LexPath {
    fn is_absolute(&self) -> bool {
        match &self.root {
            PathRoot::Relative => false,
            PathRoot::Drive { absolute, .. } => *absolute,
            _ => true,
        }
    }

    fn separator(&self) -> char {
        match self.flavor { Flavor::Posix => '/', Flavor::Windows => '\\' }
    }

    fn render(&self) -> String {
        let separator = self.separator();
        let sep = separator.to_string();
        let components = self.components.join(&sep);
        match &self.root {
            PathRoot::Relative => if components.is_empty() { ".".into() } else { components },
            PathRoot::Posix => format!("/{components}"),
            PathRoot::WinRooted => format!("\\{components}"),
            PathRoot::Drive { letter, absolute } => {
                if *absolute {
                    format!("{}:\\{components}", letter.to_ascii_uppercase())
                } else {
                    format!("{}:{components}", letter.to_ascii_uppercase())
                }
            }
            PathRoot::Unc { server, share } => {
                if components.is_empty() {
                    format!("\\\\{server}\\{share}")
                } else {
                    format!("\\\\{server}\\{share}\\{components}")
                }
            }
        }
    }

    fn dirname(&self) -> String {
        if self.components.is_empty() {
            return match &self.root {
                PathRoot::Relative => ".".to_owned(),
                _ => self.render(),
            };
        }
        let mut parent = self.clone();
        parent.components.pop();
        parent.render()
    }

    fn basename(&self) -> &str {
        self.components.last().map(String::as_str).unwrap_or_else(|| {
            if matches!(&self.root, PathRoot::Relative) { "." } else { "" }
        })
    }

    fn comparable_root(&self, other: &LexPath) -> bool {
        match (&self.root, &other.root) {
            (PathRoot::Posix, PathRoot::Posix) => true,
            (PathRoot::WinRooted, PathRoot::WinRooted) => true,
            (PathRoot::Drive { letter: a, absolute: true },
             PathRoot::Drive { letter: b, absolute: true }) => a.eq_ignore_ascii_case(b),
            (PathRoot::Unc { server: a, share: c }, PathRoot::Unc { server: b, share: d }) =>
                a.eq_ignore_ascii_case(b) && c.eq_ignore_ascii_case(d),
            _ => false,
        }
    }
}

fn windows_separator(ch: char) -> bool { ch == '/' || ch == '\\' }

fn windows_path(text: &str) -> bool {
    let bytes = text.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || text.starts_with("\\\\")
        || (cfg!(windows) && text.starts_with("//") && !text.starts_with("///"))
        || text.starts_with('\\')
        || text.contains('\\')
}

fn parse_path(text: &str) -> Result<LexPath, String> {
    if text.len() > MAX_PATH_INPUT {
        return Err("path exceeds maximum supported length".into());
    }
    if text.contains('\0') {
        return Err("path contains a NUL character".into());
    }
    // Windows canonicalize() commonly returns an extended-length namespace.
    // Normalize only ordinary drive and UNC variants; device paths are unsafe
    // to interpret using ordinary lexical filesystem rules.
    let normalized_prefix;
    let text = if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        normalized_prefix = format!(r"\\{rest}");
        normalized_prefix.as_str()
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        if rest.len() < 3 || !rest.as_bytes()[0].is_ascii_alphabetic()
            || rest.as_bytes()[1] != b':' || rest.as_bytes()[2] != b'\\' {
            return Err("unsupported Windows extended-length device path".into());
        }
        rest
    } else if text.starts_with(r"\\.\") {
        return Err("Windows device paths are not supported".into());
    } else { text };
    let bytes = text.as_bytes();
    let flavor = if windows_path(text) { Flavor::Windows } else { Flavor::Posix };
    let (root, rest) = if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let after_drive = &text[2..];
        let absolute = after_drive.starts_with(windows_separator);
        let remainder = if absolute { after_drive.trim_start_matches(windows_separator) } else { after_drive };
        (PathRoot::Drive { letter: bytes[0] as char, absolute }, remainder)
    } else if text.starts_with("\\\\") || (flavor == Flavor::Windows && text.starts_with("//")) {
        // UNC syntax requires both a server and a share name. Reject incomplete
        // network roots instead of treating them as relative or ordinary POSIX paths.
        let rest = text.trim_start_matches(windows_separator);
        let mut pieces = rest.split(windows_separator).filter(|part| !part.is_empty());
        let server = pieces.next().ok_or("UNC path is missing a server name")?;
        let share = pieces.next().ok_or("UNC path is missing a share name")?;
        if server == "." || server == ".." || share == "." || share == ".." {
            return Err("invalid UNC server or share".into());
        }
        let offset = rest.find(server).unwrap_or(0);
        let server_end = offset + server.len();
        let remaining = rest[server_end..].trim_start_matches(windows_separator);
        let share_end = share.len();
        (PathRoot::Unc { server: server.to_owned(), share: share.to_owned() },
            remaining[share_end..].trim_start_matches(windows_separator))
    } else if flavor == Flavor::Windows && text.starts_with(windows_separator) {
        (PathRoot::WinRooted, text.trim_start_matches(windows_separator))
    } else if text.starts_with('/') {
        (PathRoot::Posix, text.trim_start_matches('/'))
    } else {
        (PathRoot::Relative, text)
    };
    let mut path = LexPath { flavor, root, components: Vec::new() };
    let absolute = path.is_absolute();
    for component in rest.split(|ch| match flavor {
        Flavor::Windows => windows_separator(ch),
        Flavor::Posix => ch == '/',
    }) {
        match component {
            "" | "." => {},
            ".." => {
                if path.components.last().is_some_and(|last| last != "..") {
                    path.components.pop();
                } else if !absolute {
                    path.components.push("..".into());
                }
            }
            value => path.components.push(value.into()),
        }
    }
    Ok(path)
}

fn path_join(parts: &[String]) -> Result<String, String> {
    let nonempty: Vec<&str> = parts.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
    if nonempty.is_empty() { return Ok(".".into()); }
    let windows = nonempty.iter().any(|part| windows_path(part));
    let separator = if windows { '\\' } else { '/' };
    let mut combined = String::new();
    for (index, part) in nonempty.iter().enumerate() {
        if index != 0 && !combined.ends_with(|ch| matches!(ch, '/' | '\\' | ':')) {
            combined.push(separator);
        }
        combined.push_str(if index == 0 { *part } else { part.trim_start_matches(
            |ch| if windows { windows_separator(ch) } else { ch == '/' }
        ) });
    }
    Ok(parse_path(&combined)?.render())
}

fn current_directory() -> Result<String, String> {
    let cwd = env::current_dir().map_err(|err| format!("cannot resolve current directory: {err}"))?;
    cwd.into_os_string().into_string().map_err(|_| "current directory is not valid UTF-8".into())
}

fn path_resolve(parts: &[String]) -> Result<String, String> {
    let mut path = current_directory()?;
    for part in parts {
        if part.is_empty() { continue; }
        let incoming = parse_path(part)?;
        if matches!(&incoming.root, PathRoot::WinRooted) {
            let cwd = parse_path(&path)?;
            match cwd.root {
                PathRoot::Drive { letter, absolute: true } => {
                    path = format!("{}:{}", letter.to_ascii_uppercase(), incoming.render());
                }
                _ => return Err("cannot resolve Windows root-relative path without an active drive".into()),
            }
        } else if incoming.is_absolute() {
            path = incoming.render();
        } else if let PathRoot::Drive { letter, absolute: false } = incoming.root {
            // A drive-relative path means C:'s working directory, which is not
            // generally the process working directory and may be unavailable.
            let base = parse_path(&path)?;
            match base.root {
                PathRoot::Drive { letter: current, absolute: true } if current.eq_ignore_ascii_case(&letter) => {
                    path = path_join(&[path, part[2..].into()])?;
                }
                _ => return Err(format!("cannot resolve drive-relative path {part:?} against the current directory")),
            }
        } else {
            path = path_join(&[path, part.clone()])?;
        }
    }
    let result = parse_path(&path)?;
    if !result.is_absolute() { return Err("resolved path is not absolute".into()); }
    Ok(result.render())
}

fn path_relative(from: &str, to: &str) -> Result<String, String> {
    let left = parse_path(&path_resolve(&[from.into()])?)?;
    let right = parse_path(&path_resolve(&[to.into()])?)?;
    if !left.comparable_root(&right) { return Ok(right.render()); }
    let mut common = 0;
    while common < left.components.len() && common < right.components.len() {
        let same = match right.flavor {
            Flavor::Windows => left.components[common].eq_ignore_ascii_case(&right.components[common]),
            Flavor::Posix => left.components[common] == right.components[common],
        };
        if !same { break; }
        common += 1;
    }
    let mut output: Vec<String> = vec!["..".into(); left.components.len() - common];
    output.extend(right.components[common..].iter().cloned());
    if output.is_empty() { Ok(".".into()) }
    else { Ok(output.join(&right.separator().to_string())) }
}

fn path_extension(path: &LexPath) -> String {
    let basename = path.basename();
    if let Some(index) = basename.rfind('.') {
        if index != 0 && basename != ".." { return basename[index..].to_owned(); }
    }
    String::new()
}

#[derive(Clone, Debug)]
struct ParsedUri<'a> {
    text: &'a str,
    scheme: &'a str,
    authority: &'a str,
    has_authority: bool,
    has_query: bool,
    has_fragment: bool,
    path: &'a str,
    query: &'a str,
    fragment: &'a str,
}

fn validate_percent(text: &str) -> Result<(), String> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() || !bytes[i + 1].is_ascii_hexdigit() || !bytes[i + 2].is_ascii_hexdigit() {
                return Err("URI contains invalid percent encoding".into());
            }
            i += 3;
        } else {
            if bytes[i] < 0x20 || bytes[i] == 0x7f || bytes[i] == b' ' {
                return Err("URI contains unescaped whitespace or control characters".into());
            }
            i += 1;
        }
    }
    Ok(())
}

fn parse_uri(text: &str) -> Result<ParsedUri<'_>, String> {
    if text.len() > MAX_PATH_INPUT { return Err("URI exceeds maximum supported length".into()); }
    validate_percent(text)?;
    let colon = text.find(':').ok_or("URI is missing a scheme")?;
    let scheme = &text[..colon];
    if scheme.is_empty() || !scheme.as_bytes()[0].is_ascii_alphabetic() ||
        !scheme.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.')) {
        return Err("URI has an invalid scheme".into());
    }
    let mut after = &text[colon + 1..];
    let mut fragment = "";
    let has_fragment = after.contains('#');
    if let Some(i) = after.find('#') { fragment = &after[i + 1..]; after = &after[..i]; }
    let mut query = "";
    let has_query = after.contains('?');
    if let Some(i) = after.find('?') { query = &after[i + 1..]; after = &after[..i]; }
    let has_authority = after.starts_with("//");
    let (authority, path) = if has_authority {
        let rest = &after[2..];
        let end = rest.find('/').unwrap_or(rest.len());
        (&rest[..end], &rest[end..])
    } else {
        ("", after)
    };
    if has_authority && !path.is_empty() && !path.starts_with('/') {
        return Err("URI authority must be followed by an absolute path".into());
    }
    Ok(ParsedUri { text, scheme, authority, has_authority, has_query, has_fragment, path, query, fragment })
}

fn encoded_uri_path(bytes: &[u8], keep_drive_colon: bool) -> String {
    let mut out = String::with_capacity(bytes.len());
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for &byte in bytes {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/')
            || (keep_drive_colon && byte == b':') {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 15) as usize] as char);
        }
    }
    out
}

fn decode_uri_path(text: &str) -> Result<String, String> {
    let mut result = Vec::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let first = (bytes[i + 1] as char).to_digit(16).ok_or("invalid URI hex digit")?;
            let second = (bytes[i + 2] as char).to_digit(16).ok_or("invalid URI hex digit")?;
            let decoded = ((first << 4) | second) as u8;
            if decoded == b'/' || decoded == b'\\' || decoded == 0 {
                return Err("encoded path separator or NUL in file URI is not supported".into());
            }
            result.push(decoded);
            i += 3;
        } else {
            if bytes[i] == 0 || bytes[i] == b'\\' { return Err("invalid file URI path".into()); }
            result.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(result).map_err(|_| "file URI path is not UTF-8".into())
}

fn uri_from_file_path(path: &str) -> Result<String, String> {
    let parsed = parse_path(path)?;
    if !parsed.is_absolute() { return Err("Uri.fromFilePath expects an absolute filesystem path".into()); }
    match &parsed.root {
        PathRoot::Unc { server, share } => {
            if !server.is_ascii() || server.chars().any(|ch| matches!(ch, '/' | '\\' | '@' | ':' | '?' | '#')) {
                return Err("UNC server name is not valid as a URI authority".into());
            }
            let tail = if parsed.components.is_empty() { String::new() } else {
                format!("/{}", parsed.components.join("/"))
            };
            Ok(format!("file://{server}{}", encoded_uri_path(format!("/{share}{tail}").as_bytes(), false)))
        }
        PathRoot::Drive { letter, absolute: true } => {
            let suffix = parsed.components.join("/");
            let path = if suffix.is_empty() { format!("/{}:/", letter.to_ascii_uppercase()) }
                       else { format!("/{}:/{suffix}", letter.to_ascii_uppercase()) };
            Ok(format!("file://{}", encoded_uri_path(path.as_bytes(), true)))
        }
        PathRoot::Posix => Ok(format!("file://{}", encoded_uri_path(parsed.render().as_bytes(), false))),
        PathRoot::WinRooted => Err("a rooted Windows path without a drive cannot be converted to a file URI".into()),
        _ => Err("Uri.fromFilePath expects an absolute filesystem path".into()),
    }
}

fn uri_to_file_path(text: &str) -> Result<String, String> {
    let uri = parse_uri(text)?;
    if !uri.scheme.eq_ignore_ascii_case("file") {
        return Err("Uri.toFilePath requires a file: URI".into());
    }
    if uri.has_query || uri.has_fragment {
        return Err("file URI with query or fragment cannot be converted to a filesystem path".into());
    }
    if !uri.path.starts_with('/') {
        return Err("file URI must contain an absolute path".into());
    }
    let path = decode_uri_path(uri.path)?;
    if !uri.authority.is_empty() && !uri.authority.eq_ignore_ascii_case("localhost") {
        if !uri.authority.is_ascii() || uri.authority.chars().any(|ch| matches!(ch, '/' | '\\' | '@' | ':' | '%' | '?' | '#')) {
            return Err("file URI has an unsupported remote authority".into());
        }
        let mut pieces = path.trim_start_matches('/').split('/');
        let share = pieces.next().unwrap_or("");
        if share.is_empty() { return Err("UNC file URI has no share name".into()); }
        let remainder = pieces.collect::<Vec<_>>().join("\\");
        let unc = if remainder.is_empty() { format!("\\\\{}\\{share}", uri.authority) }
                  else { format!("\\\\{}\\{share}\\{remainder}", uri.authority) };
        return Ok(parse_path(&unc)?.render());
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 3 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() && bytes[2] == b':' {
        if bytes.get(3).is_some_and(|c| *c != b'/') { return Err("invalid Windows drive URI".into()); }
        let windows = path[1..].replace('/', "\\");
        return Ok(parse_path(&windows)?.render());
    }
    Ok(parse_path(&path)?.render())
}

impl Interpreter<'_> {
    fn path_text_arg(&self, word: RegisterWord, label: &str) -> Result<String, RuntimeError> {
        let value = Value::from_word(word);
        let Some(HeapObject::String(string)) = self.heap.get(value) else {
            return Err(RuntimeError::new(format!("TypeError: {label} must be String")));
        };
        let text = std::str::from_utf8(string.as_ref()).map_err(|_| RuntimeError::new(
            format!("TypeError: {label} is not valid UTF-8")))?;
        if text.len() > MAX_PATH_INPUT { return Err(RuntimeError::new(format!("{label} is too long"))); }
        Ok(text.to_owned())
    }

    fn new_path_string(&mut self, text: String) -> Value {
        self.heap.alloc(HeapObject::String(NapalmString::new(Arc::<[u8]>::from(text.into_bytes()))))
    }

    fn path_dispatch(&mut self, native: PathNative, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let input = |index: usize, label: &str| -> Result<String, RuntimeError> {
            self.path_text_arg(*args.get(index).ok_or_else(|| RuntimeError::new("missing path argument"))?, label)
        };
        let value = match native {
            PathNative::Join | PathNative::Resolve => {
                let parts = args.iter().enumerate().map(|(index, &value)|
                    self.path_text_arg(value, &format!("path component {index}")))
                    .collect::<Result<Vec<_>, _>>()?;
                if native == PathNative::Join { path_join(&parts) } else { path_resolve(&parts) }
            }
            PathNative::Normalize => parse_path(&input(0, "path")?).map(|path| path.render()),
            PathNative::Dirname => parse_path(&input(0, "path")?).map(|path| path.dirname()),
            PathNative::Basename => parse_path(&input(0, "path")?).map(|path| path.basename().to_owned()),
            PathNative::Extension => parse_path(&input(0, "path")?).map(|path| path_extension(&path)),
            PathNative::IsAbsolute => {
                return Ok(Value::from_bool(parse_path(&input(0, "path")?).map_err(RuntimeError::new)?.is_absolute()));
            }
            PathNative::Relative => path_relative(&input(0, "from")?, &input(1, "to")?),
        }.map_err(|error| RuntimeError::new(format!("Path: {error}")))?;
        Ok(self.new_path_string(value))
    }

    fn new_uri_record(&mut self, uri: &ParsedUri<'_>) -> Result<Value, RuntimeError> {
        let output = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
        let root = self.add_root(output);
        let result = (|| {
            for (name, text) in [
                ("text", uri.text), ("scheme", uri.scheme), ("authority", uri.authority),
                ("path", uri.path), ("query", uri.query), ("fragment", uri.fragment),
            ] {
                let value = self.new_path_string(text.to_owned());
                self.heap.remember_if_reference(output, value);
                self.heap.map_set_encoded(output,
                    MapKey::String(MapStringKey::new(Arc::<[u8]>::from(name.as_bytes()))),
                    value, None)?;
            }
            for (name, present) in [
                ("hasAuthority", uri.has_authority),
                ("hasQuery", uri.has_query),
                ("hasFragment", uri.has_fragment),
            ] {
                self.heap.map_set_encoded(output,
                    MapKey::String(MapStringKey::new(Arc::<[u8]>::from(name.as_bytes()))),
                    Value::from_bool(present), None)?;
            }
            Ok(())
        })();
        self.release_root(root);
        result?;
        let external = self.heap.get(output).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(output, external);
        self.heap.refresh_jit_view(output);
        Ok(output)
    }

    fn uri_dispatch(&mut self, native: UriNative, args: &[RegisterWord]) -> Result<Value, RuntimeError> {
        let input = *args.first().ok_or_else(|| RuntimeError::new("Uri method requires one argument"))?;
        let text = if native == UriNative::ToFilePath && matches!(self.heap.get(Value::from_word(input)), Some(HeapObject::Map(_))) {
            let key = MapKey::String(MapStringKey::new(Arc::<[u8]>::from(&b"text"[..])));
            let raw = self.heap.map_get_encoded(Value::from_word(input), &key)?;
            self.path_text_arg(raw.word(), "uri.text")?
        } else {
            self.path_text_arg(input, "Uri argument")?
        };
        match native {
            UriNative::Parse => {
                let parsed = parse_uri(&text).map_err(|error| RuntimeError::new(format!("Uri.parse: {error}")))?;
                self.new_uri_record(&parsed)
            }
            UriNative::FromFilePath => Ok(self.new_path_string(
                uri_from_file_path(&text).map_err(|error| RuntimeError::new(format!("Uri.fromFilePath: {error}")))?)),
            UriNative::ToFilePath => Ok(self.new_path_string(
                uri_to_file_path(&text).map_err(|error| RuntimeError::new(format!("Uri.toFilePath: {error}")))?)),
        }
    }

    pub(super) fn execute_path_native(&mut self, op: &PreparedOp, native: PathNative, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.path_dispatch(native, args) {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }

    pub(super) fn execute_uri_native(&mut self, op: &PreparedOp, native: UriNative, args: &[RegisterWord]) {
        self.mark_gc_point(op);
        self.maybe_collect();
        match self.uri_dispatch(native, args) {
            Ok(value) => unsafe { self.set_reg(op.a, value.word()); self.advance(op); },
            Err(error) => self.fail_at(op, error.message),
        }
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn drive_root_and_relative_are_distinct() {
        assert_eq!(parse_path(r"C:\a\..\b").unwrap().render(), r"C:\b");
        assert_eq!(parse_path(r"C:a\..\b").unwrap().render(), r"C:b");
        assert!(parse_path(r"C:\a").unwrap().is_absolute());
        assert!(!parse_path(r"C:a").unwrap().is_absolute());
    }

    #[test]
    fn unc_paths_preserve_server_and_share() {
        assert_eq!(parse_path(r"\\server\share\foo\..\bar").unwrap().render(), r"\\server\share\bar");
        assert_eq!(parse_path(r"\\server\share\..\..").unwrap().render(), r"\\server\share");
    }

    #[test]
    fn posix_paths_do_not_escape_root() {
        assert_eq!(parse_path("/a/../b/../../x").unwrap().render(), "/x");
        assert_eq!(parse_path("a/../../x").unwrap().render(), "../x");
        assert_eq!(parse_path(".gitignore").unwrap().basename(), ".gitignore");
        assert_eq!(path_extension(&parse_path(".gitignore").unwrap()), "");
        assert_eq!(path_extension(&parse_path("file.ts").unwrap()), ".ts");
    }

    #[test]
    fn uri_roundtrip_handles_utf8_and_reserved_characters() {
        let win = r"C:\My Docs\a#é?.napalm";
        let uri = uri_from_file_path(win).unwrap();
        assert_eq!(uri, "file:///C:/My%20Docs/a%23%C3%A9%3F.napalm");
        assert_eq!(uri_to_file_path(&uri).unwrap(), win);
        let unc = r"\\server\share\space name";
        let uri = uri_from_file_path(unc).unwrap();
        assert_eq!(uri, "file://server/share/space%20name");
        assert_eq!(uri_to_file_path(&uri).unwrap(), unc);
    }

    #[test]
    fn encoded_separator_is_rejected() {
        assert!(uri_to_file_path("file:///tmp/%2Fetc").is_err());
        assert!(uri_to_file_path("file:///tmp/%5Cetc").is_err());
        assert!(parse_uri("file:///bad%xy").is_err());
        assert!(uri_to_file_path("https://example.com/path").is_err());
    }

    #[test]
    fn lexical_path_join_and_relative() {
        assert_eq!(path_join(&["a".into(), "b".into(), "..".into(), "c".into()]).unwrap(), "a/c");
        assert_eq!(path_relative("/a/b", "/a/c/d").unwrap(), "../c/d");
        assert_eq!(path_relative(r"C:\Foo\bar", r"c:\foo\baz").unwrap(), r"..\baz");
    }
}

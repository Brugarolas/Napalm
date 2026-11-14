# Napalm Language Server — milestone 1

A native Napalm implementation of LSP 3.17 JSON-RPC transport and in-memory document synchronization. No Node.js, TypeScript, Rust extension, or external dependency is used **at runtime**. Node.js is only needed for the optional integration-test harness.

## Contents

- `main.napalm` — production entry point; uses `stdin` / `stdout` exclusively for LSP.
- `src/transport.napalm` — binary-safe Content-Length protocol and error handling.
- `src/server.napalm` — JSON-RPC lifecycle, server capabilities, request dispatch.
- `src/documents.napalm` — unsaved editor buffers, versions, open/change/close.
- `src/positions.napalm` — Unicode-scalar offsets ⇔ UTF-16 LSP positions; incremental edit application.
- `scripts/documents_smoke.napalm` — pure Napalm document and Unicode tests.
- `tests/*.test.mjs` — executable protocol tests for partial frames, UTF-8 byte lengths, lifecycle, and incremental edits.

## Requirements

Napalm must have the previously integrated `Process` streams, `Bytes`, `ByteBuffer`, `JSON`, and Unicode `String` utilities. In particular, `Process.stdin.readLine()` and `readExact()` must work together on the same stdin stream. The supplied source assumes those integrations are installed and passing their original smoke tests.

From the repository root:

```powershell
cargo build --release
.\target\release\napalm.exe .\lsp\scripts\documents_smoke.napalm
.\target\release\napalm.exe .\lsp\main.napalm
```

The final command blocks waiting for valid LSP frames from an editor. **Never run it as a normal interactive REPL:** raw JSON or plain-text lines are not valid LSP input. To run the real protocol tests:

```powershell
node --test .\lsp\tests\*.test.mjs
```

Or run `npm test` from `lsp/`. Set `NAPALM_BIN` to an absolute path if your executable lives elsewhere.

## Current capabilities

The server advertises **only** `positionEncoding: "utf-16"` and `textDocumentSync: {openClose: true, change: 2}`. It implements:

- `initialize`, `initialized`, `shutdown`, `exit`;
- `textDocument/didOpen`, `textDocument/didChange`, `textDocument/didClose`;
- correct JSON-RPC errors for parse errors, invalid requests, requests received before initialization, and unknown methods;
- bounded UTF-8 Content-Length input/output with multiple frames per connection, case-insensitive headers, and optional UTF-8 charset;
- incremental edits, applied in order to a document's evolving text, with CR, LF, CRLF and supplementary Unicode characters;
- editor buffer version checks and rejection of edits that split UTF-16 surrogate pairs.

It **does not yet** advertise or implement `publishDiagnostics`, completion, hover, go-to-definition, rename, symbols, or project indexing. Those require connecting the Rust frontend/parser and semantic IR to an analysis API; they must not be guessed from regexes. `Fs.watch()` is reserved for workspace changes originating outside the editor, not unsaved text.

## Invariants

1. **Never call `System.print` or `System.write`** from the running language server: stdout belongs entirely to the editor protocol. All diagnostic logging goes to `Process.stderr`.
2. Every `Content-Length` is a **UTF-8 byte count**, never a scalar or UTF-16 count.
3. Positions are **UTF-16 code-unit columns** by default; Napalm's `String.slice` uses Unicode-scalar offsets, so conversions must pass through `positions.napalm`.
4. `didChange` does not read the file from disk. The editor's unsaved text is authoritative.
5. A `shutdown` request returns `null`; the client must still send `exit`. Exiting without prior shutdown has a nonzero process exit code.

## Next milestone

Implement a **native Rust analysis bridge** exposing the existing tokenizer, parser, lowerer, source spans and module dependency graph to Napalm (without reimplementing them in another language). Start with parser diagnostics and correct UTF-16 mapped ranges, followed by document symbols, hover, completion, and go-to-definition. Keep the JSON-RPC loop itself in Napalm.

## Validation

The JavaScript framing unit tests can run without the Napalm binary. Integration tests run the native executable if it is present; they skip otherwise. No Rust compiler or Napalm executable was available in the authoring environment, so the Napalm runtime tests have not been executed here.

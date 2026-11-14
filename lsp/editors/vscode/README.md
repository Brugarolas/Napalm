# Napalm: editor syntax and file associations (milestone 1)

This package contributes **declarative** VS Code language features. There is no
extension host code, JavaScript runtime activation, or LSP dependency. The
Napalm LSP can later be launched as a separate compiled native executable.

## Install for local development

From this directory, open VS Code and press **F5** (Extension Development Host),
then open a `.napalm` source file. Alternatively, link/copy this directory into
your VS Code extensions directory and restart the editor.

To build a VSIX after installing Node.js:

```sh
npm install
npx vsce package --allow-missing-repository
code --install-extension napalm-language-0.1.0.vsix
```

`publisher` is `napalm-local` as a **local packaging placeholder**. Replace it
with your real Visual Studio Marketplace publisher, set the final license, and
remove `private` when you prepare public distribution.

## Existing artwork

No artwork is included: use your **own** existing SVG or PNG image assets. To
copy three icons and set the language's fallback icons, run:

```sh
node scripts/configure-icons.mjs \
  --napalm "C:/art/napalm.svg" \
  --fire "C:/art/fire.svg" \
  --burning "C:/art/burning.svg"
```

The helper updates `contributes.languages[].icon` and copies the files into
`icons/`. You can optionally pass `--napalm-light`, etc. to use different files
for light themes. SVG/PNG source assets are accepted.

VS Code file-icon themes may override the fallback icon for a language or
extension. A custom file-icon theme can force specific extensions to use
specific icons, but switching to that theme also affects other file icons. This
package deliberately doesn't replace the user's existing icon theme.

Windows Explorer icon associations are **separate** from VS Code icons; see
`windows/README.md` for instructions. Windows generally wants ICO assets.

## What is highlighted?

`.napalm` uses `source.napalm` TextMate grammar: comments, strings, nested
template interpolation, numbers, operators (`|>`), declarations, keywords,
builtin names, and decorators. `.burning` and `.fire` are registered as
independent file types for icon association **without** a syntax grammar. The
source format of `.fire` was not confirmed from the current public manual.

This is lexical highlighting; it is not the language's parser, error checker,
or a complete semantic-token implementation. The LSP can later add semantic
tokens without losing immediate highlighting while the server starts.

## Testing

```sh
npm test
```

Tests validate the declarative extension's registration and grammar structure.
Opening the sample in VS Code remains a useful visual smoke test; actual
TextMate tokenization is owned by the editor.

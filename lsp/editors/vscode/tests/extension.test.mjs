import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = fileURLToPath(new URL('../', import.meta.url));
const load = (path) => JSON.parse(readFileSync(join(root, path), 'utf8'));
const manifest = load('package.json');
const grammar = load('syntaxes/napalm.tmLanguage.json');
const configuration = load('language-configuration.json');

test('source and artifact extensions have separate language IDs', () => {
  const [source, burning, fire] = manifest.contributes.languages;
  assert.deepEqual(source.extensions, ['.napalm']);
  assert.deepEqual(burning.extensions, ['.burning']);
  assert.deepEqual(fire.extensions, ['.fire']);
  assert.equal(manifest.contributes.grammars.length, 1);
  assert.equal(manifest.contributes.grammars[0].language, source.id);
  assert.equal(manifest.contributes.grammars[0].scopeName, 'source.napalm');
});

test('all grammar references are resolvable and regular expressions parse', () => {
  const repository = grammar.repository;
  const visited = new Set();
  function check(rule) {
    if (Array.isArray(rule)) { rule.forEach(check); return; }
    if (!rule || typeof rule !== 'object') return;
    if (typeof rule.include === 'string' && rule.include.startsWith('#')) {
      const name = rule.include.slice(1);
      assert.ok(Object.hasOwn(repository, name), `Missing grammar rule ${name}`);
      visited.add(name);
    }
    for (const field of ['match', 'begin', 'end']) {
      if (typeof rule[field] === 'string') {
        assert.doesNotThrow(() => new RegExp(rule[field]), `${field} regex: ${rule[field]}`);
      }
    }
    if (rule.patterns) check(rule.patterns);
  }
  check(grammar.patterns);
  Object.values(repository).forEach(check);
  assert.ok(visited.has('strings') && visited.has('template-interpolation'));
});

test('language configuration supports comments and bracket pairs', () => {
  assert.equal(configuration.comments.lineComment, '//');
  assert.deepEqual(configuration.comments.blockComment, ['/*', '*/']);
  assert.ok(configuration.brackets.some(([open, close]) => open === '{' && close === '}'));
});

test('grammar recognizes key Napalm syntax', () => {
  const rules = grammar.repository;
  const patterns = (name) => rules[name].patterns.map(({ match }) => match).filter(Boolean);
  assert.ok(patterns('operators').some((pattern) => new RegExp(pattern).test('|>')));
  assert.ok(patterns('keywords').some((pattern) => new RegExp(pattern).test('concurrent')));
  assert.ok(patterns('keywords').some((pattern) => new RegExp(pattern).test('computed')));
  assert.ok(patterns('numbers').some((pattern) => new RegExp(pattern).test('0xDEAD_BEEF')));
  assert.ok(patterns('numbers').some((pattern) => new RegExp(pattern).test('1_000_000')));
  assert.ok(patterns('constants').some((pattern) => new RegExp(pattern).test('null')));
});

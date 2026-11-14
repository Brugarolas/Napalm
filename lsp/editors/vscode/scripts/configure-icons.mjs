#!/usr/bin/env node
// Install author-provided icons without supplying or modifying image artwork.
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { extname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const manifestPath = join(root, 'package.json');
const iconsDir = join(root, 'icons');
const keys = new Set([
  'napalm', 'napalm-light', 'napalm-dark',
  'fire', 'fire-light', 'fire-dark',
  'burning', 'burning-light', 'burning-dark',
]);

function usage() {
  return [
    'Usage: node scripts/configure-icons.mjs --napalm PATH [--fire PATH] [--burning PATH]',
    'Optional per-theme assets: --napalm-light PATH --napalm-dark PATH, etc.',
    'Only .svg or .png files are supported for VS Code language icons.',
  ].join('\n');
}

function parseOptions(argv) {
  if (argv.length === 0 || argv.includes('--help')) {
    process.stdout.write(`${usage()}\n`);
    process.exit(argv.includes('--help') ? 0 : 2);
  }
  const options = new Map();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key?.startsWith('--') || !keys.has(key.slice(2)) || !value || value.startsWith('--')) {
      throw new Error(`Invalid argument: ${key ?? '(missing)'}\n${usage()}`);
    }
    if (options.has(key.slice(2))) {
      throw new Error(`Duplicate option ${key}`);
    }
    options.set(key.slice(2), value);
  }
  return options;
}

function validatedPath(input) {
  const location = isAbsolute(input) ? input : resolve(process.cwd(), input);
  const extension = extname(location).toLowerCase();
  if (!['.svg', '.png'].includes(extension)) {
    throw new Error(`Expected .svg or .png for ${location}`);
  }
  if (!existsSync(location)) {
    throw new Error(`Icon not found: ${location}`);
  }
  return { location, extension };
}

function main() {
  const options = parseOptions(process.argv.slice(2));
  const prepared = [];
  for (const id of ['napalm', 'fire', 'burning']) {
    const fallback = options.get(id);
    const lightInput = options.get(`${id}-light`) ?? fallback ?? options.get(`${id}-dark`);
    const darkInput = options.get(`${id}-dark`) ?? fallback ?? options.get(`${id}-light`);
    if (!lightInput) continue;
    const light = validatedPath(lightInput);
    const dark = validatedPath(darkInput);
    prepared.push({ id, light, dark });
  }
  if (prepared.length === 0) throw new Error(`No icon paths specified.\n${usage()}`);

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const languages = manifest.contributes?.languages;
  if (!Array.isArray(languages)) throw new Error('Invalid extension manifest: missing languages');
  for (const entry of prepared) {
    const target = languages.find((language) => language.id === (entry.id === 'napalm' ? 'napalm' : `napalm-${entry.id}`));
    if (!target) throw new Error(`Missing language declaration for ${entry.id}`);
  }
  mkdirSync(iconsDir, { recursive: true });
  for (const entry of prepared) {
    const target = languages.find((language) => language.id === (entry.id === 'napalm' ? 'napalm' : `napalm-${entry.id}`));
    const lightName = `${entry.id}-light${entry.light.extension}`;
    const darkName = `${entry.id}-dark${entry.dark.extension}`;
    copyFileSync(entry.light.location, join(iconsDir, lightName));
    copyFileSync(entry.dark.location, join(iconsDir, darkName));
    target.icon = { light: `./icons/${lightName}`, dark: `./icons/${darkName}` };
    process.stdout.write(`Configured ${entry.id}: ${lightName}, ${darkName}\n`);
  }
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
}

try {
  main();
} catch (error) {
  process.stderr.write(`Error: ${error.message}\n`);
  process.exitCode = 1;
}

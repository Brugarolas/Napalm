import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, resolve, isAbsolute} from 'node:path';
import {encodeFrame, FrameDecoder} from './protocol.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const executable = process.env.NAPALM_BIN ?? resolve(
  projectRoot,
  process.platform === 'win32' ? 'target/release/napalm.exe' : 'target/release/napalm',
);
const canExecute = isAbsolute(executable) ? existsSync(executable) : true;

async function runServer(frames, {expectedExit = 0, timeoutMs = 30_000} = {}) {
  const child = spawn(executable, ['lsp/main.napalm'], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });
  const decoder = new FrameDecoder();
  const responses = [];
  const stderr = [];
  let protocolError = null;
  child.stdout.on('data', (chunk) => {
    if (protocolError !== null) return;
    try {
      responses.push(...decoder.push(chunk));
    } catch (error) {
      protocolError = error;
      child.kill();
    }
  });
  child.stderr.on('data', (chunk) => stderr.push(chunk));
  const done = new Promise((resolveDone, rejectDone) => {
    child.on('error', rejectDone);
    child.on('close', (code, signal) => resolveDone({code, signal}));
  });
  const timeout = setTimeout(() => child.kill(), timeoutMs);
  try {
    for (const frame of frames) child.stdin.write(frame);
    child.stdin.end();
    const result = await done;
    if (protocolError !== null) throw protocolError;
    decoder.finish();
    assert.equal(result.signal, null, 'Server was killed or timed out');
    assert.equal(result.code, expectedExit, Buffer.concat(stderr).toString('utf8'));
    return {responses, stderr: Buffer.concat(stderr).toString('utf8')};
  } finally {
    clearTimeout(timeout);
    if (child.exitCode === null) child.kill();
  }
}

const onlyIfPresent = {skip: !canExecute && 'Build Napalm or set NAPALM_BIN to run live LSP tests'};

test('Initialize, incremental sync, unknown request, shutdown, exit', onlyIfPresent, async () => {
  const uri = 'file:///project/source.napalm';
  const packets = [
    {jsonrpc: '2.0', id: 1, method: 'initialize', params: {capabilities: {}}},
    {jsonrpc: '2.0', method: 'initialized', params: {}},
    {jsonrpc: '2.0', method: 'textDocument/didOpen', params: {
      textDocument: {uri, languageId: 'napalm', version: 1, text: 'A😀é\r\nvalue'},
    }},
    {jsonrpc: '2.0', method: 'textDocument/didChange', params: {
      textDocument: {uri, version: 2},
      contentChanges: [{range: {
        start: {line: 0, character: 1}, end: {line: 0, character: 3},
      }, rangeLength: 2, text: 'Q'}],
    }},
    {jsonrpc: '2.0', method: 'textDocument/didClose', params: {textDocument: {uri}}},
    {jsonrpc: '2.0', id: 'no-method', method: 'textDocument/hover', params: {}},
    {jsonrpc: '2.0', id: 3, method: 'shutdown', params: null},
    {jsonrpc: '2.0', method: 'exit'},
  ];
  const {responses} = await runServer(packets.map((item) => encodeFrame(item)));
  assert.deepEqual(responses.map((item) => item.id), [1, 'no-method', 3]);
  assert.equal(responses[0].result.capabilities.positionEncoding, 'utf-16');
  assert.equal(responses[0].result.capabilities.textDocumentSync.change, 2);
  assert.equal(responses[1].error.code, -32601);
  assert.equal(responses[2].result, null);
});

test('Malformed JSON yields parse error, then the server recovers', onlyIfPresent, async () => {
  const invalidJson = Buffer.from('{bad json!', 'utf8');
  const invalidFrame = Buffer.concat([
    Buffer.from(`Content-Length: ${invalidJson.length}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n`),
    invalidJson,
  ]);
  const packets = [
    invalidFrame,
    encodeFrame({jsonrpc: '2.0', id: 42, method: 'initialize', params: {capabilities: {}}}),
    encodeFrame({jsonrpc: '2.0', id: 43, method: 'shutdown'}),
    encodeFrame({jsonrpc: '2.0', method: 'exit'}),
  ];
  const {responses} = await runServer(packets);
  assert.equal(responses[0].error.code, -32700);
  assert.equal(responses[1].id, 42);
  assert.equal(responses[2].id, 43);
});

test('Exit without shutdown uses the required nonzero exit code', onlyIfPresent, async () => {
  const packets = [encodeFrame({jsonrpc: '2.0', method: 'exit'})];
  const {responses} = await runServer(packets, {expectedExit: 1});
  assert.deepEqual(responses, []);
});

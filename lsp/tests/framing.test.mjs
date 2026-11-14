import test from 'node:test';
import assert from 'node:assert/strict';
import {FrameDecoder, encodeFrame} from './protocol.mjs';

test('Content-Length counts UTF-8 bytes, not UTF-16 code units', () => {
  const payload = {jsonrpc: '2.0', id: 4, result: 'A😀é'};
  const frame = encodeFrame(payload);
  const boundary = frame.indexOf('\r\n\r\n');
  const declared = Number(frame.subarray(0, boundary).toString('ascii').split(':')[1].trim());
  const body = frame.subarray(boundary + 4);
  assert.equal(declared, body.byteLength);
  assert.ok(declared > body.toString('utf8').length);
  assert.deepEqual(new FrameDecoder().push(frame), [payload]);
});

test('Framer supports arbitrary chunk boundaries and multiple messages', () => {
  const input = [
    {jsonrpc: '2.0', id: 1, result: {capabilities: {}}},
    {jsonrpc: '2.0', id: 2, error: {code: -32601, message: 'missing'}},
    {jsonrpc: '2.0', id: '🌍', result: null},
  ];
  const wire = Buffer.concat(input.map((value) => encodeFrame(value)));
  const decoder = new FrameDecoder();
  const output = [];
  for (let index = 0; index < wire.length; index++) {
    output.push(...decoder.push(wire.subarray(index, index + 1)));
  }
  decoder.finish();
  assert.deepEqual(output, input);
});

test('Framer detects missing and duplicate Content-Length', () => {
  const decoder = new FrameDecoder();
  assert.throws(() => decoder.push(Buffer.from('X-Header: yes\r\n\r\n{}')),
    /Content-Length/);
  const other = new FrameDecoder();
  assert.throws(() => other.push(Buffer.from('Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}')),
    /Content-Length/);
});

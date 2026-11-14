// Dependency-free LSP byte framing helpers used by integration tests.
export const MAX_FRAME_BYTES = 16 * 1024 * 1024;
export const MAX_HEADER_BYTES = 8192;

export function encodeFrame(payload, extraHeaders = []) {
  const data = Buffer.from(JSON.stringify(payload), 'utf8');
  if (data.length > MAX_FRAME_BYTES) throw new RangeError('Oversized LSP frame');
  const header = Buffer.from(
    [`Content-Length: ${data.length}`, ...extraHeaders, '', ''].join('\r\n'),
    'ascii',
  );
  return Buffer.concat([header, data]);
}

export class FrameDecoder {
  #pending = Buffer.alloc(0);
  #expectedLength = null;

  push(chunk) {
    if (!Buffer.isBuffer(chunk)) chunk = Buffer.from(chunk);
    this.#pending = Buffer.concat([this.#pending, chunk]);
    const messages = [];
    while (true) {
      if (this.#expectedLength === null) {
        const headerEnd = this.#pending.indexOf('\r\n\r\n');
        if (headerEnd === -1) {
          if (this.#pending.length > MAX_HEADER_BYTES) throw new Error('Oversized LSP headers');
          break;
        }
        if (headerEnd + 4 > MAX_HEADER_BYTES) throw new Error('Oversized LSP headers');
        const headers = this.#pending.subarray(0, headerEnd).toString('ascii').split('\r\n');
        const lengths = headers
          .filter((header) => /^content-length\s*:/i.test(header))
          .map((header) => header.slice(header.indexOf(':') + 1).trim());
        if (lengths.length !== 1 || !/^\d+$/.test(lengths[0])) {
          throw new Error('Missing or invalid Content-Length');
        }
        const length = Number(lengths[0]);
        if (!Number.isSafeInteger(length) || length > MAX_FRAME_BYTES) {
          throw new Error('Oversized LSP body');
        }
        this.#expectedLength = length;
        this.#pending = this.#pending.subarray(headerEnd + 4);
      }
      if (this.#pending.length < this.#expectedLength) break;
      const body = this.#pending.subarray(0, this.#expectedLength);
      this.#pending = this.#pending.subarray(this.#expectedLength);
      this.#expectedLength = null;
      messages.push(JSON.parse(body.toString('utf8')));
    }
    return messages;
  }

  finish() {
    if (this.#expectedLength !== null || this.#pending.length !== 0) {
      throw new Error('Incomplete LSP output frame');
    }
  }
}

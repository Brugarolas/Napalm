# HTTP services

Run `cargo run -- scripts/server_simple.napalm` or `cargo run -- scripts/server.napalm`.
Both listen on port 3000 and stay alive until Ctrl+C. The complete demo calls its
own local endpoints, including SSE, without depending on external services.

```typescript
const server = listen "127.0.0.1:3000" {
  route "GET" ["/", "/hello"] (req) => {
    return { status: 200, body: "Hello!" };
  };
  route "GET" ["/users/:id"] async (req) => {
    await Timer.delay(10);
    return { body: "User " + req.params.id };
  };
  route "POST" ["/echo"] (req) => {
    return { body: req.body, headers: [["X-Echo", "1"]] };
  };
};
```

`listen` is an expression returning a numeric server handle; using it as a
statement also works. The address can be a string literal, a variable, or an
expression beginning with either. `127.0.0.1:0` chooses a free port;
`http.address(server)` returns its actual `host:port`. `0.0.0.0` listens on all
IPv4 interfaces. Binding errors are catchable with `try/catch`.

The parser lowers this syntax to `http.listen(address, routes)`, where each route
is `[method, paths, handler]`. You can call that API directly with computed route
lists. Closures capture their lexical environment and may be async. Listeners
retain their closures for garbage collection until closed.

`http.close(server)` stops accepting connections and releases the route table;
it returns `true` once and `false` for an already closed/unknown handle. Handlers
already dispatched and open streams can finish. Requests accepted but not yet
dispatched may receive 404 after closing. Timers and other pending work still
keep the process alive. Call `http.streamEnd` to finish an open stream.

## Routing and requests

Routes match in declaration order, case-sensitive by path. Methods are normalized
to uppercase; `"*"` accepts any method. Matching ignores the query string.
`:name` captures one nonempty segment into `req.params.name`; a final `*` captures
the rest into `req.params.wildcard`. Put specific routes before parameter/wildcard
routes. Duplicate method/path pairs are rejected.

Missing paths return 404; a matching path with the wrong method returns 405 with
`Allow`. HEAD and OPTIONS need explicit routes (or a `"*"` method). There is no
automatic CORS or HEAD-to-GET fallback.

| Request field | Value |
| --- | --- |
| `method` | HTTP method |
| `path` | Path without query string |
| `url` | Request target, including query string |
| `query` | Raw query string without `?` |
| `params` | Captured path segments, without percent decoding |
| `headers` | Map of lowercase names to strings; last repeated value wins |
| `body` | Buffered text, decoded as UTF-8 with replacement for invalid bytes |
| `serverId`, `reqId` | Handles for streaming operations |

## Responses

Return `{status: 200, body: "text", headers: ...}` or just a string (status 200).
Omitted status defaults to 200; omitted body defaults to an empty string. Headers
accept objects or `[name, value]` pairs; pairs preserve repeated response headers.
Status must be an integer from 200 to 599. Body and header values must be strings.
Set `Content-Type` explicitly when serving JSON or SSE.

Thrown errors, rejected handlers, malformed responses and invalid headers return
`500 Internal Server Error` without exposing the exception to clients. Other
requests continue running. Returning `0` or `null` is only valid after starting a
stream; otherwise it produces 500. After streaming begins, normal return values
are ignored and an exception ends the stream.

## Streaming and SSE

```typescript
listen "127.0.0.1:3000" {
  route "GET" ["/sse"] (req) => {
    http.startStream(req, 200, [["Content-Type", "text/event-stream"]]);
    setTimeout(() => http.streamWrite(req.serverId, req.reqId, "data: hello\n\n"), 0);
    setTimeout(() => http.streamEnd(req.serverId, req.reqId), 200);
    return 0;
  };
}
```

`startStream` sends status and headers once. `streamWrite` queues a UTF-8 chunk;
`streamEnd` closes the stream and returns whether it was open. Writing to a
closed/disconnected stream throws. Each chunk is limited to 64 KiB, with at most
64 queued chunks per stream. A full queue throws a catchable error: retry later
using a timer or `await Timer.delay(...)`, rather than spinning in the VM. Empty
chunks are ignored. Content-Length, Transfer-Encoding, Connection, Trailer and
Upgrade are managed by the server and cannot be supplied by handlers.

## Clients and timers

```typescript
async function demo() {
  const r = await get("http://127.0.0.1:3000/");
  print(r.status); print(r.body);
  const j = await getJSON("http://127.0.0.1:3000/json");
  print(j.json);
  const p = await postJSON("http://127.0.0.1:3000/echo", "{\"a\":1}", [["X-Foo", "Bar"]]);
  print(p.json.a);
}
const timer = setTimeout(() => { demo(); }, 500);
// timer.cancel();
```

`get(url, headers?)` returns a Promise for `{status, ok, body, headers}`.
`getJSON(url, headers?)` and `postJSON(url, jsonText, headers?)` also provide
`json`, parsed from the response body; invalid response JSON rejects the Promise.
`postJSON` accepts an already serialized JSON string and defaults Content-Type to
`application/json`. Client headers accept objects and pairs. Response header names
are lowercase. HTTP error statuses resolve normally; transport errors reject.
These helpers buffer the entire response, including streams, up to the existing
fetch limit of 64 MiB. Use `fetch` for other methods and its existing Response API.
`fetch` now also accepts header pairs.

`setTimeout(callback, milliseconds)` is shorthand for
`Timer.delay(milliseconds, callback)` and returns the same cancellable Promise.
`print(value)` is shorthand for `System.print(value)`. Examples use Napalm's
`async function` and arrows; existing pipe lambdas also work as callbacks.

## Transport scope and testing

HTTP/1.0 and HTTP/1.1 over TCP, one request per connection; no HTTPS listener,
WebSockets or HTTP/2 server. Hyper and Tokio pass plain Rust data to the VM;
only the VM executes Napalm code. Long synchronous handlers can block other
handlers and timers; use async work for I/O.

Request bodies and buffered response bodies are limited to 8 MiB. At most 256
connections are active across the VM's listeners, with 100 headers per request.
Header/body reads and waiting for a handler response have 30-second timeouts;
connections, including streams, have a maximum lifetime of 5 minutes. Oversized
request bodies receive 413; body read timeouts receive 408 and handler response
timeouts receive 504. Hyper handles malformed protocol input.

Tests use local ephemeral ports and real HTTP requests:

```sh
cargo test napalm_http -- --test-threads=1
```

//! VM side of the HTTP eDSL: validation, routing, GC roots and Promise reactions.
use super::*;

pub(super) struct Route {
    method: String,
    path: String,
    pub handler: Value,
    function: u32,
}

fn id(value: Value) -> Result<u64, RuntimeError> {
    match value.as_num() {
        Some(n) if n.is_finite() && n > 0.0 && n.fract() == 0.0 && n <= 9007199254740991.0 => {
            Ok(n as u64)
        }
        _ => Err(RuntimeError::new("HTTP handle must be a positive integer")),
    }
}
fn status(value: Value) -> Result<u16, RuntimeError> {
    if value.is_null() {
        return Ok(200);
    }
    match value.as_num() {
        Some(n) if (200.0..=599.0).contains(&n) && n.fract() == 0.0 => Ok(n as u16),
        _ => Err(RuntimeError::new(
            "HTTP status must be an integer from 200 to 599",
        )),
    }
}

fn match_path(pattern: &str, path: &str) -> Option<Vec<(String, String)>> {
    let mut params = Vec::new();
    let mut actual = path.split('/');
    for part in pattern.split('/') {
        if part == "*" {
            params.push(("wildcard".into(), actual.collect::<Vec<_>>().join("/")));
            return Some(params);
        }
        let value = actual.next()?;
        if let Some(name) = part.strip_prefix(':') {
            if value.is_empty() {
                return None;
            }
            params.push((name.to_owned(), value.to_owned()));
        } else if part != value {
            return None;
        }
    }
    if actual.next().is_some() {
        None
    } else {
        Some(params)
    }
}

impl Interpreter<'_> {
    pub(super) fn http_string(&mut self, text: &str) -> Value {
        self.heap
            .alloc(HeapObject::String(NapalmString::new(Arc::<[u8]>::from(
                text.as_bytes(),
            ))))
    }
    fn http_put(&mut self, map: Value, name: &str, value: Value) -> Result<(), RuntimeError> {
        let key = MapKey::String(MapStringKey::new(Arc::<[u8]>::from(name.as_bytes())));
        self.heap.remember_if_reference(map, value);
        let structural = self.heap.map_set_encoded(map, key, value, None)?;
        if structural {
            self.heap.refresh_jit_view(map);
        }
        let external = self.heap.get(map).map_or(0, external_bytes_for);
        self.heap.set_external_bytes(map, external);
        Ok(())
    }
    fn http_pairs(&mut self, pairs: &[(String, String)]) -> Result<Value, RuntimeError> {
        let map = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
        let root = self.add_root(map);
        let result = (|| {
            for (name, text) in pairs {
                let value = self.http_string(text);
                self.http_put(map, name, value)?;
            }
            Ok(map)
        })();
        self.release_root(root);
        result
    }
    fn http_routes_from_value(&self, value: Value) -> Result<Vec<Route>, RuntimeError> {
        let Some(HeapObject::List(entries)) = self.heap.get(value) else {
            return Err(RuntimeError::new("HTTP routes must be a List"));
        };
        let mut routes = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &entry in entries {
            let Some(HeapObject::List(parts)) = self.heap.get(entry) else {
                return Err(RuntimeError::new("route must be [method, paths, handler]"));
            };
            if parts.len() != 3 {
                return Err(RuntimeError::new("route must be [method, paths, handler]"));
            }
            let method = self
                .fetch_string(parts[0], "route method")?
                .to_ascii_uppercase();
            if method != "*"
                && (method.is_empty() || reqwest::Method::from_bytes(method.as_bytes()).is_err())
            {
                return Err(RuntimeError::new("invalid route method"));
            }
            let Some(HeapObject::List(paths)) = self.heap.get(parts[1]) else {
                return Err(RuntimeError::new("route paths must be a List"));
            };
            if paths.is_empty() {
                return Err(RuntimeError::new("route requires at least one path"));
            }
            let function = match self.heap.get(parts[2]) {
                Some(HeapObject::Closure { function_index, .. }) => *function_index,
                _ => return Err(RuntimeError::new("route handler must be a function")),
            };
            if self.prepared.functions[function as usize].arity != 1 {
                return Err(RuntimeError::new(
                    "route handler must take one request argument",
                ));
            }
            for &path in paths {
                let path = self.fetch_string(path, "route path")?;
                let segments: Vec<_> = path.split('/').collect();
                let mut names = std::collections::HashSet::new();
                if !path.starts_with('/')
                    || path.contains(['?', '#'])
                    || segments.iter().enumerate().any(|(i, part)| {
                        (*part == "*" && i + 1 != segments.len())
                            || part
                                .strip_prefix(':')
                                .is_some_and(|name| name.is_empty() || !names.insert(name))
                    })
                {
                    return Err(RuntimeError::new(
                        "route path must start with /, have named parameters and only a final * wildcard",
                    ));
                }
                if !seen.insert((method.clone(), path.clone())) {
                    return Err(RuntimeError::new("duplicate HTTP route"));
                }
                routes.push(Route {
                    method: method.clone(),
                    path,
                    handler: parts[2],
                    function,
                });
            }
        }
        Ok(routes)
    }
    pub(super) fn http_native(
        &mut self,
        native: NativeBuiltin,
        args: &[Value],
    ) -> Result<Value, RuntimeError> {
        let arg = |i| args.get(i).copied().unwrap_or(Value::NULL);
        match native {
            NativeBuiltin::HttpListen => {
                let address = self.fetch_string(arg(0), "listen address")?;
                let routes = self.http_routes_from_value(arg(1))?;
                if self.http.is_none() {
                    self.http = Some(
                        http_backend::HttpRuntime::new(self.external_wake_tx.clone())
                            .map_err(RuntimeError::new)?,
                    );
                }
                let server = self
                    .http
                    .as_mut()
                    .unwrap()
                    .listen(&address)
                    .map_err(RuntimeError::new)?;
                self.http_routes.insert(server, routes);
                Ok(Value::from_num(server as f64))
            }
            NativeBuiltin::HttpClose => {
                let server = id(arg(0))?;
                let closed = self.http.as_mut().is_some_and(|h| h.close(server));
                self.http_routes.remove(&server);
                Ok(Value::from_bool(closed))
            }
            NativeBuiltin::HttpAddress => {
                let server = id(arg(0))?;
                let address = self
                    .http
                    .as_ref()
                    .and_then(|h| h.address(server))
                    .map(str::to_owned)
                    .ok_or_else(|| RuntimeError::new("HTTP server is closed or unknown"))?;
                Ok(self.http_string(&address))
            }
            NativeBuiltin::HttpStartStream => {
                let key = (
                    id(self.fetch_option_value(arg(0), "serverId")?)?,
                    id(self.fetch_option_value(arg(0), "reqId")?)?,
                );
                let status = status(arg(1))?;
                let headers = self.fetch_headers_from_value(arg(2))?;
                self.http
                    .as_ref()
                    .ok_or_else(|| RuntimeError::new("HTTP server is not running"))?
                    .start_stream(key, status, headers)
                    .map_err(RuntimeError::new)?;
                Ok(Value::NULL)
            }
            NativeBuiltin::HttpStreamWrite => {
                let key = (id(arg(0))?, id(arg(1))?);
                let body = self.fetch_string(arg(2), "stream chunk")?.into_bytes();
                self.http
                    .as_ref()
                    .ok_or_else(|| RuntimeError::new("HTTP server is not running"))?
                    .write(key, body)
                    .map_err(RuntimeError::new)?;
                Ok(Value::NULL)
            }
            NativeBuiltin::HttpStreamEnd => {
                let key = (id(arg(0))?, id(arg(1))?);
                Ok(Value::from_bool(
                    self.http.as_ref().is_some_and(|h| h.end(key)),
                ))
            }
            NativeBuiltin::SetTimeout => {
                let delay = Self::timer_duration(arg(1), false)?;
                self.schedule_timer_delay(delay, Some(arg(0)))
            }
            NativeBuiltin::HttpGet | NativeBuiltin::HttpGetJson | NativeBuiltin::HttpPostJson => {
                let post = matches!(native, NativeBuiltin::HttpPostJson);
                let json = !matches!(native, NativeBuiltin::HttpGet);
                let request = (|| {
                    let url = self.fetch_string(arg(0), "HTTP URL")?;
                    let mut headers =
                        self.fetch_headers_from_value(arg(if post { 2 } else { 1 }))?;
                    let body = if post {
                        let body = self.fetch_string(arg(1), "postJSON body (JSON string)")?;
                        if !headers
                            .iter()
                            .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
                        {
                            headers.push(("Content-Type".into(), "application/json".into()));
                        }
                        Some(body.into_bytes())
                    } else {
                        None
                    };
                    Ok::<_, RuntimeError>(FetchRequest {
                        url,
                        method: if post { "POST" } else { "GET" }.into(),
                        headers,
                        body,
                        redirect: RedirectMode::Follow,
                    })
                })();
                Ok(match request {
                    Ok(request) => self.start_fetch_mode(request, Some(json)),
                    Err(error) => self.allocate_rejected_promise_message(error.message),
                })
            }
            _ => unreachable!(),
        }
    }
    pub(super) fn http_client_response(
        &mut self,
        response: FetchResponse,
        json: bool,
    ) -> Result<Value, RuntimeError> {
        let map = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
        let root = self.add_root(map);
        let result = (|| {
            self.http_put(map, "status", Value::from_num(response.status as f64))?;
            self.http_put(
                map,
                "ok",
                Value::from_bool((200..300).contains(&response.status)),
            )?;
            let body = self.http_string(&String::from_utf8_lossy(&response.body));
            self.http_put(map, "body", body)?;
            let headers = self.http_pairs(&response.headers)?;
            self.http_put(map, "headers", headers)?;
            if json {
                let value = self.parse_json_response_body(response.body)?;
                self.http_put(map, "json", value)?;
            }
            Ok(map)
        })();
        self.release_root(root);
        result
    }
    pub(super) fn drain_http_requests(&mut self) {
        // Bound each turn so a busy listener cannot starve timers and continuations.
        for _ in 0..256 {
            let Some(request) = self.http.as_ref().and_then(|h| h.try_request()) else {
                break;
            };
            let key = (request.server, request.id);
            if !self.http.as_ref().is_some_and(|h| h.is_waiting(key)) {
                continue;
            }
            let mut allowed = Vec::new();
            let selected = self.http_routes.get(&request.server).and_then(|routes| {
                for route in routes {
                    if let Some(params) = match_path(&route.path, &request.path) {
                        if route.method == request.method || route.method == "*" {
                            return Some((route.handler, route.function, params));
                        }
                        if !allowed.contains(&route.method) {
                            allowed.push(route.method.clone());
                        }
                    }
                }
                None
            });
            let Some((handler, function, params)) = selected else {
                let (status, body, headers) = if allowed.is_empty() {
                    (404, "Not Found", vec![])
                } else {
                    (
                        405,
                        "Method Not Allowed",
                        vec![("Allow".into(), allowed.join(", "))],
                    )
                };
                let _ = self.http.as_ref().unwrap().respond(
                    key,
                    status,
                    headers,
                    body.as_bytes().to_vec(),
                );
                continue;
            };
            let callback_root = self.add_root(handler);
            let req = self.heap.alloc(HeapObject::Map(NapalmMap::default()));
            let req_root = self.add_root(req);
            let spawned = (|| {
                self.http_put(req, "serverId", Value::from_num(key.0 as f64))?;
                self.http_put(req, "reqId", Value::from_num(key.1 as f64))?;
                for (name, text) in [
                    ("method", request.method),
                    ("path", request.path),
                    ("url", request.url),
                    ("query", request.query),
                    ("body", String::from_utf8_lossy(&request.body).into_owned()),
                ] {
                    let value = self.http_string(&text);
                    self.http_put(req, name, value)?;
                }
                let headers = self.http_pairs(&request.headers)?;
                self.http_put(req, "headers", headers)?;
                let params = self.http_pairs(&params)?;
                self.http_put(req, "params", params)?;
                self.spawn_async_task(function, None, &[req.word()], Some(handler), None)
            })();
            self.release_root(req_root);
            self.release_root(callback_root);
            match spawned {
                Ok(promise) => {
                    let _ = self.attach_promise_combinator_waiter(
                        promise,
                        PromiseWaiter::HttpResponse {
                            server: key.0,
                            request: key.1,
                        },
                    );
                }
                Err(_) => {
                    self.http_handler_finished(key, PromiseOutcome::Rejected(Value::NULL));
                }
            }
        }
    }
    pub(super) fn http_handler_finished(&mut self, key: (u64, u64), outcome: PromiseOutcome) {
        if !self.http.as_ref().is_some_and(|h| h.is_waiting(key)) {
            // An exception after streaming starts closes the stream; never send a
            // second status line. Normal return leaves it to its scheduled writers.
            if !matches!(outcome, PromiseOutcome::Fulfilled(_)) {
                if let Some(h) = &self.http {
                    h.end(key);
                }
            }
            return;
        }
        let response = (|| {
            let PromiseOutcome::Fulfilled(value) = outcome else {
                return Err(RuntimeError::new("handler failed"));
            };
            if matches!(self.heap.get(value), Some(HeapObject::String(_))) {
                return Ok((
                    200,
                    vec![],
                    self.fetch_string(value, "response")?.into_bytes(),
                ));
            }
            if !matches!(self.heap.get(value), Some(HeapObject::Map(_))) {
                return Err(RuntimeError::new(
                    "handler must return a response or start a stream",
                ));
            }
            let status = status(self.fetch_option_value(value, "status")?)?;
            let body = self.fetch_option_value(value, "body")?;
            let body = if body.is_null() {
                vec![]
            } else {
                self.fetch_string(body, "response body")?.into_bytes()
            };
            if body.len() > http_backend::MAX_BODY {
                return Err(RuntimeError::new("response body exceeds 8 MiB"));
            }
            let headers =
                self.fetch_headers_from_value(self.fetch_option_value(value, "headers")?)?;
            Ok((status, headers, body))
        })();
        let h = self.http.as_ref().unwrap();
        let result = response.and_then(|(status, headers, body)| {
            h.respond(key, status, headers, body)
                .map_err(RuntimeError::new)
        });
        if result.is_err() {
            let _ = h.respond(key, 500, vec![], b"Internal Server Error".to_vec());
        }
    }
}

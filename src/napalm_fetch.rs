//! Asynchronous HTTP backend for Napalm's `fetch()` builtin.
//!
//! This module deliberately owns no Napalm `Value`s. Requests are converted to
//! plain Rust data before they cross the async boundary, and completions return
//! through an MPSC queue. The interpreter remains the sole owner of managed heap
//! values and Promise state. This separation makes the backend replaceable when
//! Napalm's N:M scheduler grows its own I/O reactor.

use reqwest::header::{HeaderName, HeaderValue, CONTENT_LENGTH, CONTENT_TYPE};
use reqwest::{Client, Method};
use crate::napalm_collections::HashMap;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::Duration;
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use tokio::task::AbortHandle;

pub type FetchId = u64;

/// Hard cap for a fully buffered Fetch body. Streaming can later lift this limit
/// behind a ReadableStream without allowing an untrusted peer to exhaust the VM.
pub const MAX_FETCH_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_REDIRECTS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedirectMode {
    Follow,
    Manual,
    Error,
}

#[derive(Clone, Debug)]
pub struct FetchRequest {
    pub url: String,
    pub method: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub redirect: RedirectMode,
}

#[derive(Clone, Debug)]
pub struct FetchResponse {
    pub status: u16,
    pub status_text: String,
    pub url: String,
    pub redirected: bool,
    pub headers: Vec<(String, String)>,
    pub content_type: Option<String>,
    pub body: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FetchError {
    pub message: String,
}

impl FetchError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FetchError {}

#[derive(Debug)]
pub struct FetchCompletion {
    pub id: FetchId,
    pub result: Result<FetchResponse, FetchError>,
}

pub struct FetchRuntime {
    runtime: Runtime,
    follow_client: Client,
    no_redirect_client: Client,
    completion_tx: Sender<FetchCompletion>,
    completion_rx: Receiver<FetchCompletion>,
    wake_tx: Option<Sender<()>>,
    aborts: HashMap<FetchId, AbortHandle>,
    next_id: FetchId,
}

impl FetchRuntime {
    pub fn new() -> Result<Self, FetchError> {
        Self::build(None)
    }

    pub fn new_with_waker(wake_tx: Sender<()>) -> Result<Self, FetchError> {
        Self::build(Some(wake_tx))
    }

    fn build(wake_tx: Option<Sender<()>>) -> Result<Self, FetchError> {
        let runtime = RuntimeBuilder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("napalm-http")
            .build()
            .map_err(|error| FetchError::new(format!("failed to create HTTP runtime: {error}")))?;

        // Construct async clients while entered into the runtime they will use.
        // This avoids relying on a caller-provided Tokio context during lazy fetch
        // initialization and keeps Interpreter::new() completely runtime-agnostic.
        let runtime_guard = runtime.enter();
        let follow_client = Client::builder()
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .build()
            .map_err(|error| FetchError::new(format!("failed to create HTTP client: {error}")))?;
        let no_redirect_client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| FetchError::new(format!("failed to create HTTP client: {error}")))?;
        drop(runtime_guard);
        let (completion_tx, completion_rx) = mpsc::channel();

        Ok(Self {
            runtime,
            follow_client,
            no_redirect_client,
            completion_tx,
            completion_rx,
            wake_tx,
            aborts: HashMap::default(),
            next_id: 1,
        })
    }

    pub fn spawn(&mut self, request: FetchRequest) -> Result<FetchId, FetchError> {
        let method = Method::from_bytes(request.method.as_bytes())
            .map_err(|error| FetchError::new(format!("invalid HTTP method `{}`: {error}", request.method)))?;

        let client = match request.redirect {
            RedirectMode::Follow => self.follow_client.clone(),
            RedirectMode::Manual | RedirectMode::Error => self.no_redirect_client.clone(),
        };
        let original_url = request.url.clone();
        let mut builder = client.request(method, &request.url);
        for (name, value) in request.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| FetchError::new(format!("invalid HTTP header name `{name}`: {error}")))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|error| FetchError::new(format!("invalid HTTP header value: {error}")))?;
            builder = builder.header(name, value);
        }
        if let Some(body) = request.body {
            builder = builder.body(body);
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let tx = self.completion_tx.clone();
        let wake_tx = self.wake_tx.clone();
        let redirect = request.redirect;

        let task = self.runtime.spawn(async move {
            let result = execute_request(builder, original_url, redirect).await;
            let _ = tx.send(FetchCompletion { id, result });
            if let Some(wake_tx) = wake_tx {
                let _ = wake_tx.send(());
            }
        });
        self.aborts.insert(id, task.abort_handle());
        Ok(id)
    }

    #[inline]
    pub fn has_pending(&self) -> bool {
        !self.aborts.is_empty()
    }

    pub fn cancel(&mut self, id: FetchId) -> bool {
        let Some(abort) = self.aborts.remove(&id) else {
            return false;
        };
        abort.abort();
        true
    }

    pub fn try_completion(&mut self) -> Option<FetchCompletion> {
        match self.completion_rx.try_recv() {
            Ok(completion) => {
                self.aborts.remove(&completion.id);
                Some(completion)
            }
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    /// Wait for one HTTP completion. `None` blocks until a request completes;
    /// `Some(duration)` preserves timer deadlines while the VM has no runnable work.
    pub fn wait_completion(&mut self, timeout: Option<Duration>) -> Result<Option<FetchCompletion>, FetchError> {
        let result = match timeout {
            Some(timeout) => match self.completion_rx.recv_timeout(timeout) {
                Ok(completion) => Some(completion),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(FetchError::new("HTTP completion channel disconnected"));
                }
            },
            None => Some(
                self.completion_rx
                    .recv()
                    .map_err(|_| FetchError::new("HTTP completion channel disconnected"))?,
            ),
        };
        if let Some(completion) = &result {
            self.aborts.remove(&completion.id);
        }
        Ok(result)
    }
}

async fn execute_request(
    builder: reqwest::RequestBuilder,
    original_url: String,
    redirect: RedirectMode,
) -> Result<FetchResponse, FetchError> {
    let mut response = builder
        .send()
        .await
        .map_err(|error| FetchError::new(format!("fetch failed: {error}")))?;

    let status = response.status();
    if redirect == RedirectMode::Error && status.is_redirection() {
        return Err(FetchError::new(format!(
            "fetch redirect rejected with HTTP status {}",
            status.as_u16()
        )));
    }

    if let Some(length) = response.headers().get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        if length > MAX_FETCH_BODY_BYTES as u64 {
            return Err(FetchError::new(format!(
                "fetch response body exceeds {} bytes",
                MAX_FETCH_BODY_BYTES
            )));
        }
    }

    let final_url = response.url().to_string();
    let redirected = reqwest::Url::parse(&original_url)
        .map(|original| original != *response.url())
        .unwrap_or_else(|_| final_url != original_url);
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let headers = collect_headers(response.headers());
    let mut body = Vec::with_capacity(response.content_length().unwrap_or(0).min(MAX_FETCH_BODY_BYTES as u64) as usize);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| FetchError::new(format!("failed to read fetch response body: {error}")))?
    {
        let new_len = body.len().saturating_add(chunk.len());
        if new_len > MAX_FETCH_BODY_BYTES {
            return Err(FetchError::new(format!(
                "fetch response body exceeds {} bytes",
                MAX_FETCH_BODY_BYTES
            )));
        }
        body.extend_from_slice(&chunk);
    }

    Ok(FetchResponse {
        status: status.as_u16(),
        status_text: status.canonical_reason().unwrap_or("").to_owned(),
        redirected,
        url: final_url,
        headers,
        content_type,
        body,
    })
}

fn collect_headers(headers: &reqwest::header::HeaderMap) -> Vec<(String, String)> {
    // Header multiplicity is observable. Combine repeated values using the same
    // comma+space convention exposed by browser Headers for ordinary headers.
    let mut out = Vec::<(String, String)>::new();
    for (name, value) in headers.iter() {
        let value = String::from_utf8_lossy(value.as_bytes());
        if let Some((_, existing)) = out.iter_mut().find(|(candidate, _)| candidate.eq_ignore_ascii_case(name.as_str())) {
            existing.push_str(", ");
            existing.push_str(&value);
        } else {
            out.push((name.as_str().to_ascii_lowercase(), value.into_owned()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn serve_once(response: Vec<u8>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local fetch fixture");
        let address = listener.local_addr().expect("fixture address");
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("accept fetch fixture");
            let _ = socket.set_read_timeout(Some(Duration::from_secs(5)));
            let mut request = [0u8; 4096];
            let _ = socket.read(&mut request);
            socket.write_all(&response).expect("write fetch fixture response");
            socket.flush().expect("flush fetch fixture response");
        });
        (format!("http://{address}/fixture"), handle)
    }

    fn get(url: String, redirect: RedirectMode) -> FetchRequest {
        FetchRequest {
            url,
            method: "GET".to_owned(),
            headers: Vec::new(),
            body: None,
            redirect,
        }
    }

    #[test]
    fn http_error_status_is_a_successful_fetch_response() {
        let body = br#"{"ok":false,"reason":"missing"}"#;
        let response = format!(
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            std::str::from_utf8(body).unwrap()
        );
        let (url, server) = serve_once(response.into_bytes());

        let mut runtime = FetchRuntime::new().expect("create fetch runtime");
        let id = runtime.spawn(get(url, RedirectMode::Follow)).expect("spawn fetch");
        let completion = runtime
            .wait_completion(Some(Duration::from_secs(5)))
            .expect("wait fetch")
            .expect("fetch timed out");
        assert_eq!(completion.id, id);
        let response = completion.result.expect("404 must not reject fetch");
        assert_eq!(response.status, 404);
        assert_eq!(response.content_type.as_deref(), Some("application/json"));
        assert_eq!(response.body, body);
        server.join().unwrap();
    }

    #[test]
    fn redirect_error_mode_rejects_without_following() {
        let response = b"HTTP/1.1 302 Found\r\nLocation: /elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let (url, server) = serve_once(response.to_vec());

        let mut runtime = FetchRuntime::new().expect("create fetch runtime");
        runtime.spawn(get(url, RedirectMode::Error)).expect("spawn fetch");
        let completion = runtime
            .wait_completion(Some(Duration::from_secs(5)))
            .expect("wait fetch")
            .expect("fetch timed out");
        assert!(completion.result.is_err());
        server.join().unwrap();
    }
}

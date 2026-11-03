//! HTTP/1 transport. Only Rust data crosses threads; handlers run on the VM.
use http_body_util::{BodyExt, Full};
use hyper::{
    Request, Response,
    body::{Body, Bytes, Frame, Incoming, SizeHint},
    server::conn::http1,
    service::service_fn,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    runtime::{Builder, Runtime},
    sync::{Semaphore, mpsc as channel, oneshot},
    task::AbortHandle,
};

pub const MAX_BODY: usize = 8 * 1024 * 1024;
pub const MAX_CHUNK: usize = 64 * 1024;
type Key = (u64, u64);
type Slots = Arc<Mutex<BTreeMap<Key, Slot>>>;

pub struct HttpRequest {
    pub server: u64,
    pub id: u64,
    pub method: String,
    pub path: String,
    pub url: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

enum ServerBody {
    Full(Full<Bytes>),
    Stream(channel::Receiver<Bytes>),
}
impl Body for ServerBody {
    type Data = Bytes;
    type Error = Infallible;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match &mut *self {
            Self::Full(body) => Pin::new(body).poll_frame(cx),
            Self::Stream(rx) => rx
                .poll_recv(cx)
                .map(|chunk| chunk.map(|bytes| Ok(Frame::data(bytes)))),
        }
    }
    fn is_end_stream(&self) -> bool {
        match self {
            Self::Full(body) => body.is_end_stream(),
            Self::Stream(_) => false,
        }
    }
    fn size_hint(&self) -> SizeHint {
        match self {
            Self::Full(body) => body.size_hint(),
            Self::Stream(_) => SizeHint::default(),
        }
    }
}
enum Slot {
    Waiting(oneshot::Sender<Response<ServerBody>>),
    Streaming(channel::Sender<Bytes>),
}
struct Server {
    address: String,
    abort: AbortHandle,
}

pub struct HttpRuntime {
    runtime: Runtime,
    servers: BTreeMap<u64, Server>,
    slots: Slots,
    active: Arc<AtomicUsize>,
    permits: Arc<Semaphore>,
    next_server: u64,
    next_request: Arc<AtomicU64>,
    tx: mpsc::Sender<HttpRequest>,
    rx: mpsc::Receiver<HttpRequest>,
    wake: mpsc::Sender<()>,
}

// Also runs on timeout, peer disconnect and task cancellation.
struct ConnectionGuard {
    key: Key,
    slots: Slots,
    active: Arc<AtomicUsize>,
    wake: mpsc::Sender<()>,
}
impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.slots.lock().unwrap().remove(&self.key);
        self.active.fetch_sub(1, Ordering::SeqCst);
        let _ = self.wake.send(());
    }
}

impl HttpRuntime {
    pub fn new(wake: mpsc::Sender<()>) -> Result<Self, String> {
        let runtime = Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("napalm-server")
            .build()
            .map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::channel();
        Ok(Self {
            runtime,
            servers: BTreeMap::new(),
            slots: Arc::new(Mutex::new(BTreeMap::new())),
            active: Arc::new(AtomicUsize::new(0)),
            permits: Arc::new(Semaphore::new(256)),
            next_server: 1,
            next_request: Arc::new(AtomicU64::new(1)),
            tx,
            rx,
            wake,
        })
    }
    pub fn listen(&mut self, address: &str) -> Result<u64, String> {
        // Binding is synchronous so errors are catchable at the listen expression.
        let listener =
            std::net::TcpListener::bind(address).map_err(|e| format!("listen {address}: {e}"))?;
        let address = listener
            .local_addr()
            .map_err(|e| e.to_string())?
            .to_string();
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let _entered = self.runtime.enter();
        let listener = tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
        let id = self.next_server;
        self.next_server += 1;
        let (slots, active, permits, next, tx, wake) = (
            self.slots.clone(),
            self.active.clone(),
            self.permits.clone(),
            self.next_request.clone(),
            self.tx.clone(),
            self.wake.clone(),
        );
        let task = self.runtime.spawn(async move {
            loop {
                let Ok(permit) = permits.clone().acquire_owned().await else {
                    break;
                };
                let (socket, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let key = (id, next.fetch_add(1, Ordering::Relaxed));
                active.fetch_add(1, Ordering::SeqCst);
                let guard = ConnectionGuard {
                    key,
                    slots: slots.clone(),
                    active: active.clone(),
                    wake: wake.clone(),
                };
                let (slots, tx, wake) = (slots.clone(), tx.clone(), wake.clone());
                tokio::spawn(async move {
                    let (_guard, _permit) = (guard, permit);
                    let service = service_fn(move |req| {
                        receive(req, key, slots.clone(), tx.clone(), wake.clone())
                    });
                    let mut builder = http1::Builder::new();
                    builder
                        .keep_alive(false)
                        .max_headers(100)
                        .timer(TokioTimer::new())
                        .header_read_timeout(Duration::from_secs(30));
                    // Finite connection lifetime also reclaims abandoned streams.
                    let _ = tokio::time::timeout(
                        Duration::from_secs(300),
                        builder.serve_connection(TokioIo::new(socket), service),
                    )
                    .await;
                });
            }
        });
        self.servers.insert(
            id,
            Server {
                address,
                abort: task.abort_handle(),
            },
        );
        Ok(id)
    }
    pub fn has_pending(&self) -> bool {
        !self.servers.is_empty() || self.active.load(Ordering::SeqCst) != 0
    }
    pub fn try_request(&self) -> Option<HttpRequest> {
        self.rx.try_recv().ok()
    }
    pub fn address(&self, id: u64) -> Option<&str> {
        self.servers.get(&id).map(|s| s.address.as_str())
    }
    pub fn close(&mut self, id: u64) -> bool {
        if let Some(server) = self.servers.remove(&id) {
            server.abort.abort();
            true
        } else {
            false
        }
    }
    pub fn is_waiting(&self, key: Key) -> bool {
        matches!(self.slots.lock().unwrap().get(&key), Some(Slot::Waiting(_)))
    }
    pub fn respond(
        &self,
        key: Key,
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> Result<(), String> {
        let response = build_response(
            status,
            headers,
            ServerBody::Full(Full::new(Bytes::from(body))),
        )?;
        let mut slots = self.slots.lock().unwrap();
        if !matches!(slots.get(&key), Some(Slot::Waiting(_))) {
            return Err("request is closed or already handled".into());
        }
        let Some(Slot::Waiting(tx)) = slots.remove(&key) else {
            unreachable!()
        };
        tx.send(response).map_err(|_| "client disconnected".into())
    }
    pub fn start_stream(
        &self,
        key: Key,
        status: u16,
        headers: Vec<(String, String)>,
    ) -> Result<(), String> {
        let (tx, rx) = channel::channel(64);
        let response = build_response(status, headers, ServerBody::Stream(rx))?;
        let mut slots = self.slots.lock().unwrap();
        if !matches!(slots.get(&key), Some(Slot::Waiting(_))) {
            return Err("request is closed or already handled".into());
        }
        let Some(Slot::Waiting(reply)) = slots.remove(&key) else {
            unreachable!()
        };
        reply
            .send(response)
            .map_err(|_| "client disconnected".to_string())?;
        slots.insert(key, Slot::Streaming(tx));
        Ok(())
    }
    pub fn write(&self, key: Key, body: Vec<u8>) -> Result<(), String> {
        if body.len() > MAX_CHUNK {
            return Err("stream chunk exceeds 64 KiB".into());
        }
        let slots = self.slots.lock().unwrap();
        let Some(Slot::Streaming(tx)) = slots.get(&key) else {
            return Err("stream is not open".into());
        };
        if body.is_empty() {
            return Ok(());
        }
        tx.try_send(Bytes::from(body))
            .map_err(|e| format!("stream write failed (retry on a full buffer): {e}"))
    }
    pub fn end(&self, key: Key) -> bool {
        let mut slots = self.slots.lock().unwrap();
        if matches!(slots.get(&key), Some(Slot::Streaming(_))) {
            slots.remove(&key);
            true
        } else {
            false
        }
    }
}

fn build_response(
    status: u16,
    headers: Vec<(String, String)>,
    body: ServerBody,
) -> Result<Response<ServerBody>, String> {
    if !(200..=599).contains(&status) {
        return Err("HTTP status must be an integer from 200 to 599".into());
    }
    let mut response = Response::builder().status(status);
    for (name, value) in headers {
        // Framing belongs to Hyper; caller-supplied framing can corrupt streams.
        if [
            "content-length",
            "transfer-encoding",
            "connection",
            "trailer",
            "upgrade",
        ]
        .iter()
        .any(|s| name.eq_ignore_ascii_case(s))
        {
            return Err(format!("HTTP header {name} is managed by the server"));
        }
        response = response.header(name, value);
    }
    response.body(body).map_err(|e| e.to_string())
}
fn error_response(status: u16, message: &str) -> Response<ServerBody> {
    build_response(
        status,
        vec![],
        ServerBody::Full(Full::new(Bytes::copy_from_slice(message.as_bytes()))),
    )
    .unwrap()
}
async fn receive(
    mut request: Request<Incoming>,
    key: Key,
    slots: Slots,
    tx: mpsc::Sender<HttpRequest>,
    wake: mpsc::Sender<()>,
) -> Result<Response<ServerBody>, Infallible> {
    if request
        .body()
        .size_hint()
        .upper()
        .is_some_and(|n| n > MAX_BODY as u64)
    {
        return Ok(error_response(413, "Payload Too Large"));
    }
    let body = tokio::time::timeout(Duration::from_secs(30), async {
        let mut body = Vec::new();
        while let Some(frame) = request.body_mut().frame().await {
            let frame = frame.map_err(|_| 400u16)?;
            if let Ok(data) = frame.into_data() {
                if body.len().saturating_add(data.len()) > MAX_BODY {
                    return Err(413);
                }
                body.extend_from_slice(&data);
            }
        }
        Ok(body)
    })
    .await;
    let body = match body {
        Ok(Ok(body)) => body,
        Ok(Err(code)) => return Ok(error_response(code, "Invalid request body")),
        Err(_) => return Ok(error_response(408, "Request Timeout")),
    };
    let event = HttpRequest {
        server: key.0,
        id: key.1,
        method: request.method().to_string(),
        path: request.uri().path().to_string(),
        url: request.uri().to_string(),
        query: request.uri().query().unwrap_or("").to_string(),
        headers: request
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    String::from_utf8_lossy(v.as_bytes()).into_owned(),
                )
            })
            .collect(),
        body,
    };
    let (reply, rx) = oneshot::channel();
    slots.lock().unwrap().insert(key, Slot::Waiting(reply));
    if tx.send(event).is_err() {
        return Ok(error_response(503, "Server closed"));
    }
    let _ = wake.send(());
    Ok(
        match tokio::time::timeout(Duration::from_secs(30), rx).await {
            Ok(Ok(response)) => response,
            _ => error_response(504, "Handler Timeout"),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;

    fn fixture(request: &str) -> (HttpRuntime, u64, BufReader<TcpStream>) {
        let (wake, _) = mpsc::channel();
        let mut runtime = HttpRuntime::new(wake).unwrap();
        let server = runtime.listen("127.0.0.1:0").unwrap();
        let mut socket = TcpStream::connect(runtime.address(server).unwrap()).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket.write_all(request.as_bytes()).unwrap();
        (runtime, server, BufReader::new(socket))
    }

    #[test]
    fn napalm_http_wire_stream_delivers_before_end_and_preserves_headers() {
        let (mut runtime, server, mut socket) =
            fixture("GET /sse HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let request = runtime.rx.recv_timeout(Duration::from_secs(3)).unwrap();
        let key = (server, request.id);
        runtime
            .start_stream(
                key,
                200,
                vec![
                    ("Set-Cookie".into(), "a=1".into()),
                    ("Set-Cookie".into(), "b=2".into()),
                ],
            )
            .unwrap();
        assert!(runtime.start_stream(key, 200, vec![]).is_err());
        runtime.write(key, b"hello".to_vec()).unwrap();
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            assert!(socket.read_line(&mut line).unwrap() > 0);
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }
        assert!(headers.starts_with("HTTP/1.1 200"));
        assert!(headers.contains("transfer-encoding: chunked"));
        assert!(headers.contains("set-cookie: a=1\r\nset-cookie: b=2"));
        let mut chunk = [0; 10];
        socket.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk, b"5\r\nhello\r\n");
        // We have read the first frame while the stream is still open.
        assert!(runtime.end(key));
        assert!(!runtime.end(key));
        assert!(runtime.write(key, b"late".to_vec()).is_err());
        let mut end = String::new();
        socket.read_to_string(&mut end).unwrap();
        assert_eq!(end, "0\r\n\r\n");
        runtime.close(server);
    }

    #[test]
    fn napalm_http_wire_chunked_request_and_payload_limit() {
        let (mut runtime, server, mut socket) = fixture(
            "POST /echo?q=1 HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n3\r\none\r\n3\r\ntwo\r\n0\r\n\r\n",
        );
        let request = runtime.rx.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(request.body, b"onetwo");
        assert_eq!(request.path, "/echo");
        assert_eq!(request.query, "q=1");
        runtime
            .respond((server, request.id), 200, vec![], request.body)
            .unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        assert!(response.ends_with("onetwo"));
        runtime.close(server);

        let (mut runtime, server, mut socket) = fixture(&format!(
            "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        ));
        let mut status = String::new();
        socket.read_line(&mut status).unwrap();
        assert!(status.starts_with("HTTP/1.1 413"));
        assert!(runtime.try_request().is_none());
        runtime.close(server);
    }
}

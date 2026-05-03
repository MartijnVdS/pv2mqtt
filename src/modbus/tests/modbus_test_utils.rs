use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_modbus::prelude::*;
use tokio_modbus::server::tcp::Server;
use tracing::{debug, error};

pub struct MockService {
    pub registers: Arc<Mutex<Vec<u16>>>,
}

pub struct MockServerHandle {
    pub registers: Arc<Mutex<Vec<u16>>>,
    pub connections: Arc<AtomicU32>,
    pub addr: SocketAddr,
    pub notify: Arc<tokio::sync::Notify>,
}

impl tokio_modbus::server::Service for MockService {
    type Request = Request<'static>;
    type Response = Response;
    type Exception = ExceptionCode;
    type Future = std::future::Ready<Result<Self::Response, Self::Exception>>;

    fn call(&self, req: Self::Request) -> Self::Future {
        match req {
            Request::ReadHoldingRegisters(addr, cnt) => {
                let regs = self.registers.lock().unwrap();
                let end = (addr + cnt) as usize;
                if end <= regs.len() {
                    std::future::ready(Ok(Response::ReadHoldingRegisters(
                        regs[addr as usize..end].to_vec(),
                    )))
                } else {
                    std::future::ready(Err(ExceptionCode::IllegalDataAddress))
                }
            }
            Request::WriteMultipleRegisters(addr, vals) => {
                let mut regs = self.registers.lock().unwrap();
                let end = (addr as usize) + vals.len();
                if end <= regs.len() {
                    for (i, &val) in vals.iter().enumerate() {
                        regs[(addr as usize) + i] = val;
                    }
                    std::future::ready(Ok(Response::WriteMultipleRegisters(
                        addr,
                        vals.len() as u16,
                    )))
                } else {
                    std::future::ready(Err(ExceptionCode::IllegalDataAddress))
                }
            }
            _ => std::future::ready(Err(ExceptionCode::IllegalFunction)),
        }
    }
}

pub async fn start_mock_server(addr: SocketAddr) -> MockServerHandle {
    let registers = Arc::new(Mutex::new(vec![0u16; 60000]));
    let regs_clone = registers.clone();
    let connections = Arc::new(AtomicU32::new(0));
    let conn_clone = connections.clone();
    let notify = Arc::new(tokio::sync::Notify::new());
    let notify_clone = notify.clone();

    let listener = TcpListener::bind(addr)
        .await
        .expect("Failed to bind mock server");
    let local_addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let server = Server::new(listener);

        let on_connected = move |stream: TcpStream, addr: SocketAddr| {
            let regs = regs_clone.clone();
            let notify = notify_clone.clone();
            conn_clone.fetch_add(1, Ordering::SeqCst);
            notify.notify_waiters();
            async move {
                debug!("Connected: {}", addr);
                Ok(Some((MockService { registers: regs }, stream)))
            }
        };

        let on_process_error = |err: io::Error| {
            error!("Process error: {}", err);
        };

        server.serve(&on_connected, on_process_error).await.unwrap();
    });

    MockServerHandle {
        registers,
        connections,
        addr: local_addr,
        notify,
    }
}

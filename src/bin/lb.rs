// TCP to UDS FD-passing load balancer (port 9999).
//
// For each accepted TCP connection:
//   1. Round-robin pick an upstream.
//   2. sendmsg(SCM_RIGHTS) hands the FD over a persistent UDS connection.
//   3. Close the local copy of the FD.
//
// Design notes:
//   * SOCK_STREAM control sockets.
//   * cmsg pre-initialised per sender.
//   * WORKERS threads share the listener via SO_REUSEPORT.
//   * Each worker owns its own persistent UDS FDs (no contention).

#[cfg(not(target_os = "linux"))]
fn main() -> std::io::Result<()> {
    panic!("Linux only");
}

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    use std::io;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    let port: u16 = std::env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(9999);
    let upstreams_raw = std::env::var("FD_UPSTREAMS")
        .or_else(|_| std::env::var("LB_BACKENDS"))
        .unwrap_or_else(|_| "/tmp/sock/api1.sock,/tmp/sock/api2.sock".into());
    let upstreams: Vec<String> = upstreams_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if upstreams.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no upstreams"));
    }
    let workers: usize = std::env::var("WORKERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    // Bind first so health check sees port open while we wait for upstreams.
    let listener_fd = create_tcp_listener(port)?;
    eprintln!("[lb] listening port={} workers={} upstreams={}", port, workers, upstreams.len());

    wait_upstreams(&upstreams);
    eprintln!("[lb] upstreams ready");

    let upstreams = Arc::new(upstreams);
    let next = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let upstreams = upstreams.clone();
        let next = next.clone();
        handles.push(std::thread::spawn(move || worker_loop(listener_fd, &upstreams, &next)));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn create_tcp_listener(port: u16) -> std::io::Result<libc::c_int> {
    use std::io;
    let sock = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if sock < 0 { return Err(io::Error::last_os_error()); }
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(sock, libc::SOL_SOCKET, libc::SO_REUSEADDR,
                         &one as *const _ as *const _, std::mem::size_of::<libc::c_int>() as _);
        libc::setsockopt(sock, libc::SOL_SOCKET, libc::SO_REUSEPORT,
                         &one as *const _ as *const _, std::mem::size_of::<libc::c_int>() as _);
    }
    let mut addr: libc::sockaddr_in = unsafe { std::mem::zeroed() };
    addr.sin_family = libc::AF_INET as libc::sa_family_t;
    addr.sin_addr.s_addr = u32::to_be(libc::INADDR_ANY);
    addr.sin_port = u16::to_be(port);
    let r = unsafe { libc::bind(sock, &addr as *const _ as *const _,
                                std::mem::size_of::<libc::sockaddr_in>() as _) };
    if r != 0 {
        let e = io::Error::last_os_error();
        unsafe { libc::close(sock) };
        return Err(e);
    }
    let r = unsafe { libc::listen(sock, 8192) };
    if r != 0 {
        let e = io::Error::last_os_error();
        unsafe { libc::close(sock) };
        return Err(e);
    }
    Ok(sock)
}

#[cfg(target_os = "linux")]
fn connect_unix(path: &str) -> std::io::Result<libc::c_int> {
    use std::io;
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 { return Err(io::Error::last_os_error()); }
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_bytes();
    if bytes.len() >= addr.sun_path.len() {
        unsafe { libc::close(fd) };
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "path too long"));
    }
    for (i, &b) in bytes.iter().enumerate() {
        addr.sun_path[i] = b as libc::c_char;
    }
    let r = unsafe { libc::connect(fd, &addr as *const _ as *const _,
                                   std::mem::size_of::<libc::sockaddr_un>() as _) };
    if r != 0 {
        let e = io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e);
    }
    Ok(fd)
}

#[cfg(target_os = "linux")]
fn wait_upstreams(upstreams: &[String]) {
    let delay = std::time::Duration::from_millis(50);
    for _ in 0..200 {
        let ready = upstreams.iter().filter(|p| match connect_unix(p) {
            Ok(fd) => { unsafe { libc::close(fd) }; true }
            Err(_) => false,
        }).count();
        if ready == upstreams.len() { return; }
        std::thread::sleep(delay);
    }
    eprintln!("[lb] upstreams not all up after 10s, starting anyway");
}

#[cfg(target_os = "linux")]
struct Sender {
    path: String,
    fd: libc::c_int,
}

#[cfg(target_os = "linux")]
impl Sender {
    fn new(path: String) -> Self { Self { path, fd: -1 } }
    fn send(&mut self, client_fd: libc::c_int) -> bool {
        if self.fd < 0 {
            self.fd = connect_unix(&self.path).unwrap_or(-1);
            if self.fd < 0 { return false; }
        }
        if send_fd_once(self.fd, client_fd) { return true; }
        unsafe { libc::close(self.fd) };
        self.fd = connect_unix(&self.path).unwrap_or(-1);
        if self.fd < 0 { return false; }
        send_fd_once(self.fd, client_fd)
    }
}

#[cfg(target_os = "linux")]
impl Drop for Sender {
    fn drop(&mut self) {
        if self.fd >= 0 { unsafe { libc::close(self.fd) }; }
    }
}

#[cfg(target_os = "linux")]
fn set_tcp_nodelay(fd: libc::c_int) {
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_NODELAY,
                         &one as *const _ as *const _, std::mem::size_of::<libc::c_int>() as _);
    }
}

#[cfg(target_os = "linux")]
fn worker_loop(listener_fd: libc::c_int, upstreams: &[String],
               next: &std::sync::atomic::AtomicUsize) {
    use std::sync::atomic::Ordering;
    let mut senders: Vec<Sender> = upstreams.iter().map(|p| Sender::new(p.clone())).collect();
    loop {
        let client = unsafe { libc::accept4(listener_fd, std::ptr::null_mut(),
                                            std::ptr::null_mut(), libc::SOCK_CLOEXEC) };
        if client < 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) { continue; }
            eprintln!("[lb] accept error: {}", err);
            continue;
        }
        set_tcp_nodelay(client);
        let start = next.fetch_add(1, Ordering::Relaxed);
        let mut sent = false;
        for i in 0..senders.len() {
            let idx = (start + i) % senders.len();
            if senders[idx].send(client) { sent = true; break; }
        }
        if !sent {
            // Best-effort 502
            let resp = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            unsafe { libc::write(client, resp.as_ptr() as *const _, resp.len()); }
        }
        unsafe { libc::close(client) };
    }
}

#[cfg(target_os = "linux")]
fn send_fd_once(control_fd: libc::c_int, client_fd: libc::c_int) -> bool {
    let one: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &one as *const _ as *mut _,
        iov_len: 1,
    };
    // CMSG_SPACE(sizeof(c_int)) is 24 on x86_64 Linux. Use 32 for safety.
    let mut cmsg_buf = [0u8; 32];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut _;
    msg.msg_controllen = cmsg_buf.len() as _;
    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() { return false; }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;
        std::ptr::copy_nonoverlapping(
            &client_fd as *const libc::c_int as *const u8,
            libc::CMSG_DATA(cmsg),
            std::mem::size_of::<libc::c_int>(),
        );
        // Set msg_controllen to the actual cmsg_len after initialisation.
        msg.msg_controllen = (*cmsg).cmsg_len;

        loop {
            let n = libc::sendmsg(control_fd, &msg, libc::MSG_NOSIGNAL);
            if n == 1 { return true; }
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.raw_os_error() == Some(libc::EINTR) { continue; }
                eprintln!("[lb] sendmsg failed: {}", e);
            }
            return false;
        }
    }
}

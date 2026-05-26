// API server. Receives client TCP file descriptors from the LB via
// recvmsg(SCM_RIGHTS) and serves HTTP/1.1 over them directly with epoll.
//
// Threads:
//   * fd-accept: accept LB control connections on the UDS listener.
//   * fd-recv (per control conn): blocking recvmsg loop, pushes incoming
//     client FDs onto an mpsc channel and pokes the eventfd.
//   * epoll: waits on client FDs and the eventfd, parses HTTP, runs the IVF
//     classifier and writes the response.

use std::collections::HashMap;
use std::ffi::CString;
use std::io;
use std::os::fd::RawFd;
use std::sync::mpsc;
use std::sync::Arc;

use libc::{
    epoll_create1, epoll_ctl, epoll_event, epoll_wait, EPOLLIN, EPOLLOUT,
    EPOLL_CTL_ADD, EPOLL_CTL_DEL, EPOLL_CTL_MOD,
};

use crate::http::{parse_request, RequestKind};
use crate::json::parse_payload;
use crate::normalize::vectorize_int16;
use crate::response::{response_for, FALLBACK_LEGIT, READY_OK};
use crate::specialist::SpecialistIndex;

const READ_BUF_SIZE: usize = 8192;
// 64 keeps per-iteration cost bounded under spike: large batches stretch
// the time-to-first-byte of late events in the batch.
const MAX_EVENTS: usize = 64;
const LISTEN_BACKLOG: i32 = 4096;
const WAKE_TOKEN: u64 = u64::MAX;

struct Conn {
    fd: RawFd,
    read_buf: Vec<u8>,
    filled: usize,
    write_buf: Vec<u8>,
    write_pos: usize,
    want_close: bool,
}

impl Conn {
    fn new(fd: RawFd) -> Self {
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFL, 0);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
            // SO_BUSY_POLL spins on the socket for a few microseconds before
            // sleeping in epoll. Helpful at low RPS, harmful at saturation —
            // when the CPU is already busy, the spin just burns quota. Off
            // by default; set RINHA_BUSY_POLL_US=50 to re-enable.
            let busy: libc::c_int = std::env::var("RINHA_BUSY_POLL_US")
                .ok()
                .and_then(|v| v.parse::<libc::c_int>().ok())
                .unwrap_or(0);
            if busy > 0 {
                libc::setsockopt(
                    fd, libc::SOL_SOCKET, libc::SO_BUSY_POLL,
                    &busy as *const _ as *const _,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
            }
            let one: libc::c_int = 1;
            libc::setsockopt(
                fd, libc::IPPROTO_TCP, libc::TCP_NODELAY,
                &one as *const _ as *const _,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            // Force ACK immediately on the next recv/send instead of the
            // kernel's delayed-ACK heuristic. Kernel resets QUICKACK after
            // each event, so we re-arm it in flush_write too.
            libc::setsockopt(
                fd, libc::IPPROTO_TCP, libc::TCP_QUICKACK,
                &one as *const _ as *const _,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
        Conn {
            fd,
            read_buf: vec![0u8; READ_BUF_SIZE],
            filled: 0,
            write_buf: Vec::with_capacity(256),
            write_pos: 0,
            want_close: false,
        }
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd); }
    }
}

pub fn run(sock_path: &str, index: Arc<SpecialistIndex>, _workers: usize) -> io::Result<()> {
    // Don't pin: without cpuset every API container would land on cpu 0,
    // and contention with the LB on the same core dwarfs any cache locality
    // gain. Let the kernel scheduler distribute across the 4 cpus available.
    let _ = std::fs::remove_file(sock_path);
    let listener_fd = bind_uds_listener(sock_path)?;
    eprintln!("listening on {}", sock_path);

    let (fd_tx, fd_rx) = mpsc::channel::<RawFd>();
    let wake_fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if wake_fd < 0 {
        return Err(io::Error::last_os_error());
    }

    let tx = fd_tx.clone();
    std::thread::Builder::new()
        .name("fd-accept".into())
        .spawn(move || accept_loop(listener_fd, tx, wake_fd))?;

    epoll_main_loop(index, fd_rx, wake_fd)
}

// Pin the current thread to the first CPU allowed by the cgroup. Reduces
// cross-core migrations that flush L1/L2 caches mid-IVF-scan.
fn pin_current_thread_to_first_cpu() {
    unsafe {
        let mut allowed: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut allowed);
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) != 0 {
            return;
        }
        let mut pinned: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut pinned);
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &allowed) {
                libc::CPU_SET(cpu, &mut pinned);
                break;
            }
        }
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &pinned);
    }
}

fn bind_uds_listener(path: &str) -> io::Result<RawFd> {
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let cpath = CString::new(path).unwrap();
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let p = cpath.as_bytes_with_nul();
    if p.len() > addr.sun_path.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "path too long"));
    }
    for (i, &b) in p.iter().enumerate() {
        addr.sun_path[i] = b as libc::c_char;
    }
    let path_len = (std::mem::size_of_val(&addr.sun_family) + p.len()) as libc::socklen_t;
    if unsafe { libc::bind(fd, &addr as *const _ as *const _, path_len) } < 0 {
        let e = io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e);
    }
    unsafe { libc::chmod(cpath.as_ptr(), 0o666); }
    if unsafe { libc::listen(fd, LISTEN_BACKLOG) } < 0 {
        let e = io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e);
    }
    Ok(fd)
}

fn accept_loop(listener_fd: RawFd, fd_tx: mpsc::Sender<RawFd>, wake_fd: RawFd) {
    loop {
        let control = unsafe {
            libc::accept4(
                listener_fd, std::ptr::null_mut(), std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        if control < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            eprintln!("accept error: {}", err);
            continue;
        }
        let tx = fd_tx.clone();
        std::thread::Builder::new()
            .name("fd-recv".into())
            .spawn(move || recv_loop(control, tx, wake_fd))
            .ok();
    }
}

fn recv_loop(control_fd: RawFd, fd_tx: mpsc::Sender<RawFd>, wake_fd: RawFd) {
    while let Some(client_fd) = recv_fd(control_fd) {
        if fd_tx.send(client_fd).is_err() {
            unsafe { libc::close(client_fd) };
            break;
        }
        // Poke the eventfd so the epoll thread wakes up and drains the queue.
        let one: u64 = 1;
        unsafe { libc::write(wake_fd, &one as *const _ as *const _, 8); }
    }
    unsafe { libc::close(control_fd) };
}

fn recv_fd(control_fd: RawFd) -> Option<RawFd> {
    let mut payload: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &mut payload as *mut _ as *mut _,
        iov_len: 1,
    };
    let mut cmsg_buf = [0u8; 64];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut _;
    msg.msg_controllen = cmsg_buf.len() as _;
    loop {
        let n = unsafe { libc::recvmsg(control_fd, &mut msg, 0) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return None;
        }
        if n == 0 {
            return None;
        }
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET
                    && (*cmsg).cmsg_type == libc::SCM_RIGHTS
                    && (*cmsg).cmsg_len >= libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _
                {
                    let mut fd: libc::c_int = -1;
                    std::ptr::copy_nonoverlapping(
                        libc::CMSG_DATA(cmsg) as *const u8,
                        &mut fd as *mut libc::c_int as *mut u8,
                        std::mem::size_of::<libc::c_int>(),
                    );
                    return Some(fd);
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
        }
    }
}

fn epoll_main_loop(
    index: Arc<SpecialistIndex>,
    fd_rx: mpsc::Receiver<RawFd>,
    wake_fd: RawFd,
) -> io::Result<()> {
    let epfd = unsafe { epoll_create1(libc::EPOLL_CLOEXEC) };
    if epfd < 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe {
        let mut ev = epoll_event { events: EPOLLIN as u32, u64: WAKE_TOKEN };
        epoll_ctl(epfd, EPOLL_CTL_ADD, wake_fd, &mut ev);
    }
    let mut conns: HashMap<RawFd, Conn> = HashMap::with_capacity(2048);
    let mut events: Vec<epoll_event> = vec![epoll_event { events: 0, u64: 0 }; MAX_EVENTS];
    loop {
        let n = unsafe { epoll_wait(epfd, events.as_mut_ptr(), MAX_EVENTS as i32, -1) };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            eprintln!("epoll_wait error: {}", err);
            break;
        }
        for i in 0..n as usize {
            let ev = events[i];
            let token = ev.u64;
            if token == WAKE_TOKEN {
                let mut drain = [0u8; 8];
                while unsafe { libc::read(wake_fd, drain.as_mut_ptr() as *mut _, 8) } == 8 {}
                while let Ok(client_fd) = fd_rx.try_recv() {
                    register_client(epfd, client_fd, &mut conns);
                }
            } else {
                handle_client_event(&ev, epfd, &mut conns, &index);
            }
        }
    }
    unsafe { libc::close(epfd); }
    Ok(())
}

fn register_client(epfd: RawFd, client_fd: RawFd, conns: &mut HashMap<RawFd, Conn>) {
    let conn = Conn::new(client_fd);
    unsafe {
        let mut e = epoll_event { events: EPOLLIN as u32, u64: client_fd as u64 };
        if epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &mut e) < 0 {
            eprintln!("epoll add client failed: {}", io::Error::last_os_error());
            return;
        }
    }
    conns.insert(client_fd, conn);
}

fn handle_client_event(
    ev: &epoll_event,
    epfd: RawFd,
    conns: &mut HashMap<RawFd, Conn>,
    index: &SpecialistIndex,
) {
    let fd = ev.u64 as RawFd;
    let evs = { let e = ev.events; e } as i32;
    let close_now = {
        let Some(c) = conns.get_mut(&fd) else { return; };
        let mut close = false;
        if evs & EPOLLIN != 0 && !handle_readable(c, index, epfd) {
            close = true;
        }
        if !close && evs & EPOLLOUT != 0 && !flush_write(c, epfd) {
            close = true;
        }
        close
    };
    if close_now {
        drop_conn(conns, fd, epfd);
    }
}

fn handle_readable(c: &mut Conn, index: &SpecialistIndex, epfd: RawFd) -> bool {
    loop {
        if c.filled >= c.read_buf.len() {
            return false;
        }
        let n = unsafe {
            libc::recv(
                c.fd,
                c.read_buf[c.filled..].as_mut_ptr() as *mut _,
                c.read_buf.len() - c.filled,
                0,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EAGAIN) {
                break;
            }
            return false;
        }
        if n == 0 {
            return false;
        }
        c.filled += n as usize;
    }
    let mut start = 0usize;
    loop {
        let parsed = parse_request(&c.read_buf[start..c.filled]);
        match parsed.kind {
            RequestKind::NeedMore => break,
            RequestKind::BadRequest => return false,
            RequestKind::Ready => {
                c.write_buf.extend_from_slice(READY_OK);
                start += parsed.consumed;
                if !parsed.keep_alive {
                    c.want_close = true;
                }
            }
            RequestKind::NotFound => {
                c.write_buf.extend_from_slice(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                c.want_close = true;
                start += parsed.consumed;
                break;
            }
            RequestKind::FraudScore => {
                let resp = match parse_payload(parsed.body) {
                    Ok(p) => {
                        let q = vectorize_int16(&p);
                        let frauds = index.fraud_count(&q);
                        response_for(frauds)
                    }
                    Err(_) => FALLBACK_LEGIT,
                };
                c.write_buf.extend_from_slice(resp);
                start += parsed.consumed;
                if !parsed.keep_alive {
                    c.want_close = true;
                }
            }
        }
        if start >= c.filled {
            break;
        }
    }
    if start > 0 {
        c.read_buf.copy_within(start..c.filled, 0);
        c.filled -= start;
    }
    flush_write(c, epfd)
}

fn flush_write(c: &mut Conn, epfd: RawFd) -> bool {
    while c.write_pos < c.write_buf.len() {
        let n = unsafe {
            libc::send(
                c.fd,
                c.write_buf[c.write_pos..].as_ptr() as *const _,
                c.write_buf.len() - c.write_pos,
                libc::MSG_NOSIGNAL,
            )
        };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EAGAIN) {
                unsafe {
                    let mut ev = epoll_event {
                        events: (EPOLLIN | EPOLLOUT) as u32,
                        u64: c.fd as u64,
                    };
                    epoll_ctl(epfd, EPOLL_CTL_MOD, c.fd, &mut ev);
                }
                return true;
            }
            return false;
        }
        c.write_pos += n as usize;
    }
    if c.write_pos == c.write_buf.len() {
        c.write_buf.clear();
        c.write_pos = 0;
        unsafe {
            let one: libc::c_int = 1;
            libc::setsockopt(
                c.fd, libc::IPPROTO_TCP, libc::TCP_QUICKACK,
                &one as *const _ as *const _,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            let mut ev = epoll_event { events: EPOLLIN as u32, u64: c.fd as u64 };
            epoll_ctl(epfd, EPOLL_CTL_MOD, c.fd, &mut ev);
        }
    }
    if c.want_close && c.write_pos == c.write_buf.len() {
        return false;
    }
    true
}

fn drop_conn(conns: &mut HashMap<RawFd, Conn>, fd: RawFd, epfd: RawFd) {
    if conns.remove(&fd).is_some() {
        unsafe {
            let mut ev = epoll_event { events: 0, u64: 0 };
            let _ = epoll_ctl(epfd, EPOLL_CTL_DEL, fd, &mut ev);
        }
    }
}

use std::path::PathBuf;
use std::sync::Arc;

use rinha2026::server;
use rinha2026::specialist::SpecialistIndex;

fn main() -> std::io::Result<()> {
    // Disable timer coalescing so epoll wake-ups don't get batched up to
    // 50us by the kernel. Costs ~1ns per timer arming; reduces tail latency
    // on the busy-poll path.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_TIMERSLACK, 1u64, 0u64, 0u64, 0u64);
    }

    let index_path = std::env::var("RINHA_INDEX_PATH")
        .unwrap_or_else(|_| "/data/specialist.bin".into());
    let sock_path = std::env::var("RINHA_SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/sock/api.sock".into());

    let t0 = std::time::Instant::now();
    let index = SpecialistIndex::load(&PathBuf::from(&index_path))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    eprintln!(
        "specialist: {} vectors, {} partitions, {} nodes ({}ms)",
        index.total_vectors, index.n_partitions(), index.n_nodes(),
        t0.elapsed().as_millis()
    );

    if let Some(parent) = std::path::Path::new(&sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    server::run(&sock_path, Arc::new(index), 1)
}

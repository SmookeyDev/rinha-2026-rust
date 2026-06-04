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
        // Pin every page now AND any future allocation into RAM. Removes
        // minor page-fault jitter under cgroup memory pressure. Best-effort:
        // silent EPERM/ENOMEM when CAP_IPC_LOCK or RLIMIT_MEMLOCK is missing.
        libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
        // Bump scheduling priority to the negative-nice ceiling. Best-effort
        // (silent EACCES without CAP_SYS_NICE).
        libc::setpriority(libc::PRIO_PROCESS, 0, -20);
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

    // Synthetic-query warmup drives the full predict path so icache,
    // branch predictors and per-thread Scratch are primed before /ready
    // opens. Disabled when RINHA_WARMUP_QUERIES=0.
    let warmup_n: usize = std::env::var("RINHA_WARMUP_QUERIES")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    if warmup_n > 0 {
        let tw = std::time::Instant::now();
        index.warmup_queries(warmup_n);
        eprintln!("warmup: {} synthetic queries ({}ms)",
            warmup_n, tw.elapsed().as_millis());
    }

    if let Some(parent) = std::path::Path::new(&sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    server::run(&sock_path, Arc::new(index), 1)
}

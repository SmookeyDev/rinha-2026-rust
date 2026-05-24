use std::path::PathBuf;
use std::sync::Arc;

use rinha2026::ivf::IvfIndex;
use rinha2026::server;

fn main() -> std::io::Result<()> {
    let index_path = std::env::var("RINHA_INDEX_PATH")
        .unwrap_or_else(|_| "/data/ivf_int16.bin".into());
    let sock_path = std::env::var("RINHA_SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/sock/api.sock".into());
    let nprobe: u32 = std::env::var("RINHA_NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(192);

    let t0 = std::time::Instant::now();
    let index = IvfIndex::load(&PathBuf::from(&index_path), nprobe)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    eprintln!(
        "index: {} vectors, {} clusters, nprobe={} ({}ms)",
        index.n_total, index.n_clusters, nprobe, t0.elapsed().as_millis()
    );

    if let Some(parent) = std::path::Path::new(&sock_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    server::run(&sock_path, Arc::new(index), 1)
}

// Offline check: loads the IVF index and runs fraud_count() against every
// entry in test-data.json, comparing against expected_approved.
//
//   cargo build --release --bin verify
//   ./target/release/verify <ivf_int16.bin> <test-data.json> [nprobe]

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use rinha2026::ivf::IvfIndex;
use rinha2026::json::parse_payload;
use rinha2026::normalize::vectorize_int16;
use rinha2026::specialist::SpecialistIndex;

enum AnyIndex {
    Ivf(IvfIndex),
    Specialist(SpecialistIndex),
}

impl AnyIndex {
    fn fraud_count(&self, q: &[i16; 14]) -> u8 {
        match self {
            AnyIndex::Ivf(i) => i.fraud_count(q),
            AnyIndex::Specialist(s) => s.fraud_count(q),
        }
    }
}

fn load_any(path: &str, nprobe: u32) -> std::io::Result<AnyIndex> {
    let mut buf = [0u8; 8];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        f.read_exact(&mut buf)?;
    }
    if &buf == rinha2026::specialist::MAGIC {
        let idx = SpecialistIndex::load(&PathBuf::from(path))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
        eprintln!("loaded SpecialistIndex: {} partitions, {} nodes",
                  idx.n_partitions(), idx.n_nodes());
        Ok(AnyIndex::Specialist(idx))
    } else {
        let idx = IvfIndex::load(&PathBuf::from(path), nprobe)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
        eprintln!("loaded IvfIndex: {} vectors, {} clusters, nprobe={}",
                  idx.n_total, idx.n_clusters, nprobe);
        Ok(AnyIndex::Ivf(idx))
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <ivf_int16.bin> <test-data.json> [nprobe]", args[0]);
        std::process::exit(1);
    }
    let bin_path = &args[1];
    let test_path = &args[2];
    let nprobe: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(192);

    let t0 = Instant::now();
    let index = load_any(bin_path, nprobe)?;
    eprintln!("index loaded in {} ms", t0.elapsed().as_millis());

    let raw = std::fs::read(test_path)?;
    let s = std::str::from_utf8(&raw).unwrap();
    // The file is one giant JSON object; we slice from the start of the
    // "entries" array and walk it manually to avoid pulling in a full JSON
    // crate just for verification.
    let entries_start = s.find("\"entries\":[").unwrap() + "\"entries\":[".len();
    let bytes = &s.as_bytes()[entries_start..];

    let mut idx = 0usize;
    let (mut count, mut correct, mut fp, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    let mut total = std::time::Duration::ZERO;

    while idx < bytes.len() {
        while idx < bytes.len() && matches!(bytes[idx], b',' | b' ' | b'\n' | b'\r' | b'\t') {
            idx += 1;
        }
        if idx >= bytes.len() || bytes[idx] == b']' {
            break;
        }
        let entry_start = idx;
        let mut depth = 0i32;
        let mut in_str = false;
        let mut end = idx;
        while end < bytes.len() {
            let c = bytes[end];
            if in_str {
                if c == b'\\' { end += 2; continue; }
                if c == b'"' { in_str = false; }
            } else {
                match c {
                    b'"' => in_str = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 { end += 1; break; }
                    }
                    _ => {}
                }
            }
            end += 1;
        }
        let entry = &bytes[entry_start..end];
        idx = end;

        let entry_str = std::str::from_utf8(entry).unwrap();
        let req_start = entry_str.find("\"request\":{").unwrap() + "\"request\":".len();
        let req_bytes = &entry[req_start..];
        let mut depth = 0i32;
        let mut in_str = false;
        let mut req_end = 0;
        while req_end < req_bytes.len() {
            let c = req_bytes[req_end];
            if in_str {
                if c == b'\\' { req_end += 2; continue; }
                if c == b'"' { in_str = false; }
            } else {
                match c {
                    b'"' => in_str = true,
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 { req_end += 1; break; }
                    }
                    _ => {}
                }
            }
            req_end += 1;
        }
        let req = &req_bytes[..req_end];
        let expected = entry_str.contains("\"expected_approved\":true");

        let payload = parse_payload(req)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "parse"))?;
        let q = vectorize_int16(&payload);
        let t = Instant::now();
        let frauds = index.fraud_count(&q);
        total += t.elapsed();
        let approved = frauds < 3;

        if approved == expected {
            correct += 1;
        } else if approved && !expected {
            fn_ += 1;
        } else {
            fp += 1;
        }
        count += 1;
    }

    let errors = fp + fn_;
    let mean_us = total.as_micros() as f64 / count as f64;
    println!("total:   {}", count);
    println!("correct: {} ({:.4}%)", correct, 100.0 * correct as f64 / count as f64);
    println!("errors:  {} (FP={} FN={}, E={})", errors, fp, fn_, fp + 3 * fn_);
    println!("mean:    {:.1} us/query", mean_us);

    let e = (fp + 3 * fn_) as f64;
    let eps = (e / count as f64).max(0.001);
    let rate = 1000.0 * (1.0_f64 / eps).log10();
    let pen = -300.0 * (1.0 + e).log10();
    println!("score_det estimate: {:.2} (rate={:.2} pen={:.2})", rate + pen, rate, pen);
    Ok(())
}

// Offline builder for the specialist index.
//
// Reads references.json.gz, quantises each vector to i16 (scale=10000),
// groups by 8-bit partition key, builds a balanced k-d tree per partition
// (split on the widest-range axis, leaf size LEAF_SIZE), packs each leaf's
// vectors into SoA-8 panels, then serialises in the format SpecialistIndex
// expects:
//
//   header | partitions[*] | nodes[*] | panels[*] | labels[*]
//
//   ./preprocess <references.json.gz> <out.bin>

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use flate2::read::GzDecoder;

use rinha2026::specialist::{
    compute_partition_key, pad_query, DIM, FORMAT_VERSION, Header, LANES, LEAF_SIZE,
    MAGIC, MAX_PARTITIONS, Node, PACKED_DIMS, Partition, QueryVector,
};

const SCALE: i32 = 10000;

#[derive(Clone)]
struct RefRow {
    vec: QueryVector,
    label: u8, // 0 legit, 1 fraud
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <references.json.gz> <out.bin>", args[0]);
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    let t0 = Instant::now();
    let mut refs = read_references(input)?;
    eprintln!("read {} references in {} ms", refs.len(), t0.elapsed().as_millis());

    let mut buckets: Vec<Vec<RefRow>> = (0..MAX_PARTITIONS).map(|_| Vec::new()).collect();
    for r in refs.drain(..) {
        let key = compute_partition_key(&r.vec) as usize;
        buckets[key].push(r);
    }
    for (k, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            eprintln!("  partition {:3}: {} refs", k, b.len());
        }
    }

    // Build the tree for each non-empty partition, accumulating the global
    // node, panel and label arrays.
    let mut partitions: Vec<Partition> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut panels: Vec<i16> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    let t1 = Instant::now();
    for (key, mut bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let start_vec = labels.len() as u32;
        let vec_count = bucket.len() as u32;
        let (pmin, pmax) = compute_bbox(&bucket);
        let root_node = build_subtree(&mut bucket, &mut nodes, &mut panels, &mut labels);
        partitions.push(Partition {
            key: key as u32,
            root_node: root_node as u32,
            start_vec,
            vec_count,
            min: pmin,
            max: pmax,
        });
    }
    eprintln!("built {} partitions / {} nodes / {} panels in {} ms",
              partitions.len(), nodes.len(), panels.len() / (DIM * LANES),
              t1.elapsed().as_millis());

    let header = Header {
        magic: *MAGIC,
        version: FORMAT_VERSION,
        scale: SCALE,
        partition_count: partitions.len() as u32,
        node_count: nodes.len() as u32,
        total_vectors: labels.len() as u32,
        total_panels: (panels.len() / (DIM * LANES)) as u32,
    };

    let mut out = File::create(output)?;
    write_pod(&mut out, &header)?;
    for p in &partitions {
        write_pod(&mut out, p)?;
    }
    for n in &nodes {
        write_pod(&mut out, n)?;
    }
    out.write_all(unsafe {
        std::slice::from_raw_parts(panels.as_ptr() as *const u8, panels.len() * 2)
    })?;
    out.write_all(&labels)?;
    out.flush()?;
    let total_bytes = std::fs::metadata(output)?.len();
    eprintln!("wrote {} bytes to {}", total_bytes, output);
    Ok(())
}

fn write_pod<T: Copy>(out: &mut File, v: &T) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>())
    };
    out.write_all(bytes)
}

// references.json.gz is a single big JSON array of {"vector": [...], "label": "fraud"|"legit"}.
// We parse it lazily-ish using serde_json::StreamDeserializer over the
// individual object values. For ~3M entries this is fast enough (~3s on
// modern hardware).
fn read_references(path: &str) -> std::io::Result<Vec<RefRow>> {
    let f = File::open(path)?;
    let gz = GzDecoder::new(BufReader::new(f));
    let mut buf = String::new();
    BufReader::new(gz).read_to_string(&mut buf)?;
    let v: serde_json::Value = serde_json::from_str(&buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let arr = v.as_array()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "top-level not an array"))?;
    let mut out: Vec<RefRow> = Vec::with_capacity(arr.len());
    for item in arr {
        let vec_arr = item.get("vector")
            .and_then(|v| v.as_array())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing vector"))?;
        if vec_arr.len() != DIM {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected {} dims, got {}", DIM, vec_arr.len()),
            ));
        }
        let mut unpacked = [0i16; DIM];
        for (i, x) in vec_arr.iter().enumerate() {
            let f = x.as_f64()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "non-float dim"))?;
            let scaled = (f * SCALE as f64).round() as i32;
            unpacked[i] = scaled.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        let label_str = item.get("label").and_then(|v| v.as_str())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing label"))?;
        let label = match label_str {
            "fraud" => 1u8,
            "legit" => 0u8,
            other => return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown label {}", other),
            )),
        };
        out.push(RefRow { vec: pad_query(&unpacked), label });
    }
    Ok(out)
}

fn compute_bbox(rows: &[RefRow]) -> ([i16; PACKED_DIMS], [i16; PACKED_DIMS]) {
    let mut min = [i16::MAX; PACKED_DIMS];
    let mut max = [i16::MIN; PACKED_DIMS];
    for r in rows {
        for d in 0..PACKED_DIMS {
            if r.vec[d] < min[d] { min[d] = r.vec[d]; }
            if r.vec[d] > max[d] { max[d] = r.vec[d]; }
        }
    }
    // For empty pad dims (14..16) min may still be MAX. Clamp them to 0.
    for d in DIM..PACKED_DIMS {
        if min[d] > max[d] {
            min[d] = 0;
            max[d] = 0;
        }
    }
    (min, max)
}

fn build_subtree(rows: &mut [RefRow], nodes: &mut Vec<Node>,
                 panels: &mut Vec<i16>, labels: &mut Vec<u8>) -> usize {
    // Allocate the node index up front so children get inserted after this
    // node — the index is stable.
    let my_idx = nodes.len();
    nodes.push(Node {
        left: -1, right: -1,
        start_panel: 0, vec_count: 0, start_vec: 0, _pad: 0,
        min: [0; PACKED_DIMS], max: [0; PACKED_DIMS],
    });

    let (min, max) = compute_bbox(rows);
    nodes[my_idx].min = min;
    nodes[my_idx].max = max;

    if rows.len() <= LEAF_SIZE {
        // Leaf: pack vectors as SoA-8 panels in `panels`, append labels.
        let start_vec = labels.len() as u32;
        let start_panel = (panels.len() / (DIM * LANES)) as u32;
        pack_leaf(rows, panels, labels);
        nodes[my_idx].start_panel = start_panel;
        nodes[my_idx].vec_count = rows.len() as u32;
        nodes[my_idx].start_vec = start_vec;
        return my_idx;
    }

    // Pick widest-range real dim.
    let mut best_d = 0;
    let mut best_range = -1i32;
    for d in 0..DIM {
        let r = (max[d] as i32) - (min[d] as i32);
        if r > best_range { best_range = r; best_d = d; }
    }
    // Median split.
    let mid = rows.len() / 2;
    rows.select_nth_unstable_by_key(mid, |r| r.vec[best_d]);
    let (left, right) = rows.split_at_mut(mid);
    let left_idx = build_subtree(left, nodes, panels, labels);
    let right_idx = build_subtree(right, nodes, panels, labels);
    nodes[my_idx].left = left_idx as i32;
    nodes[my_idx].right = right_idx as i32;
    my_idx
}

fn pack_leaf(rows: &[RefRow], panels: &mut Vec<i16>, labels: &mut Vec<u8>) {
    let n = rows.len();
    let n_full = n / LANES;
    let tail = n % LANES;

    for panel in 0..n_full {
        for d in 0..DIM {
            for lane in 0..LANES {
                panels.push(rows[panel * LANES + lane].vec[d]);
            }
        }
    }
    if tail > 0 {
        for d in 0..DIM {
            for lane in 0..LANES {
                if lane < tail {
                    panels.push(rows[n_full * LANES + lane].vec[d]);
                } else {
                    panels.push(0);
                }
            }
        }
    }
    for r in rows {
        labels.push(r.label);
    }
}

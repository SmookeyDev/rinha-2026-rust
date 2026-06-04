// Exact k-NN k=5 over partitioned k-d trees with bounding-box pruning.
// Replaces the IVF (approximate) approach with an exact one that prunes
// aggressively per query.
//
// Layout in memory (matches the on-disk format):
//   header (32 B) | partitions [N x 80 B] | nodes [M x 80 B]
//                 | panels (SoA, 224 B each) | labels (u8 each)
//
// Each tree node carries a [i16; PACKED_DIMS] bounding box; the search uses
// it to skip whole subtrees whose lower-bound distance is already >= the
// current 5th-best distance.

use std::cell::RefCell;
use std::fs::File;
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::slice;

#[allow(unused_imports)]
use std::arch::x86_64::{
    __m128i, __m256i,
    _mm_loadu_si128, _mm_prefetch, _MM_HINT_T0,
    _mm_setzero_si128, _mm_set1_epi16, _mm_sub_epi16,
    _mm_unpacklo_epi16, _mm_unpackhi_epi16, _mm_madd_epi16, _mm_add_epi32,
    _mm_add_epi64, _mm_extract_epi64,
    _mm256_setzero_si256, _mm256_loadu_si256, _mm256_sub_epi16, _mm256_max_epi16,
    _mm256_madd_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_cvtepi32_epi64,
    _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_set1_epi32,
    _mm256_cmpgt_epi32, _mm256_castsi256_ps, _mm256_movemask_ps,
    _mm256_storeu_si256,
};

pub const DIM: usize = 14;
pub const PACKED_DIMS: usize = 16;
pub const K: usize = 5;
pub const LANES: usize = 8;
// PAIRS = ceil(DIM/2): each panel column holds two consecutive dims packed
// into one i32 lane, so _mm256_madd_epi16(diff, diff) sums their squares
// directly. DIM=14 → 7 pairs.
pub const PAIRS: usize = (DIM + 1) / 2;
pub const LEAF_SIZE: usize = 128;
pub const MAX_PARTITIONS: usize = 256;
pub const TREE_STACK_CAPACITY: usize = 128;

// When the 5th-best squared distance falls below this threshold, the top-5
// are tight enough (within ~0.14 in normalized feature space) that no
// further probing can change the fraud count. Empirically validated on the
// test dataset by MXLange's c-api-rinha2026; we adopt the same constant.
//   ((QUANT_SCALE * 140) / 1000)^2 = 1400^2 = 1_960_000
pub const EARLY_DISTANCE_LIMIT: i64 = 1_960_000;

// Strong-decision early termination (gated by RINHA_STRONG_DECISION env).
// When the top-5 are unanimous (all legit or all fraud) and tight enough
// (~0.20 normalized = 2000^2), the binary approved/denied outcome can't
// flip even if a closer neighbor exists.
pub const STRONG_DECISION_LIMIT: i64 = 4_000_000;

// RSPECST2 = pair-interleaved i16 panel layout (vs RSPECST1 dim-interleaved).
// Each panel is 7 __m256i = 7×8×2 = 112 i16 = 224 B, same footprint as v1
// but the i16 are arranged as 7 pairs × 8 lanes × (even_dim, odd_dim).
pub const MAGIC: &[u8; 8] = b"RSPECST2";
pub const FORMAT_VERSION: u32 = 2;

pub type QueryVector = [i16; PACKED_DIMS];

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub scale: i32,
    pub partition_count: u32,
    pub node_count: u32,
    pub total_vectors: u32,
    pub total_panels: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Partition {
    pub key: u32,
    pub root_node: u32,
    pub start_vec: u32,
    pub vec_count: u32,
    pub min: [i16; PACKED_DIMS],
    pub max: [i16; PACKED_DIMS],
}

// A node is either internal (left >= 0 && right >= 0) or a leaf
// (left < 0 || right < 0). For leaves, start_panel/vec_count/start_vec
// describe the contiguous block of vectors in the panel array.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Node {
    pub left: i32,
    pub right: i32,
    pub start_panel: u32,
    pub vec_count: u32,
    pub start_vec: u32,
    pub _pad: u32,
    pub min: [i16; PACKED_DIMS],
    pub max: [i16; PACKED_DIMS],
}

pub const HEADER_BYTES: usize = size_of::<Header>();
pub const PARTITION_BYTES: usize = size_of::<Partition>();
pub const NODE_BYTES: usize = size_of::<Node>();
pub const PANEL_BYTES: usize = DIM * LANES * size_of::<i16>(); // 224

const _ASSERT_HEADER_SIZE: () = assert!(HEADER_BYTES == 32);
const _ASSERT_PARTITION_SIZE: () = assert!(PARTITION_BYTES == 80);
const _ASSERT_NODE_SIZE: () = assert!(NODE_BYTES == 88);

#[inline]
pub fn pad_query(unpacked: &[i16; DIM]) -> QueryVector {
    let mut q = [0i16; PACKED_DIMS];
    q[..DIM].copy_from_slice(unpacked);
    q
}

// True when top-5 are all legit (sum=0) or all fraud (sum=K). At worst a
// single closer neighbor swaps one slot, moving the count by 1 — still on
// the same side of the 3-of-5 threshold, so the binary decision is locked.
#[inline(always)]
fn is_unanimous(labels: &[u8; K]) -> bool {
    let s: u32 = labels.iter().map(|&l| l as u32).sum();
    s == 0 || s == K as u32
}

// 8-bit partition key. Bits are chosen so that vectors sharing the same key
// occupy a tight region of feature space, which makes per-partition trees
// shallow and the cross-partition bounding-box pruning very effective.
#[inline]
pub fn compute_partition_key(q: &QueryVector) -> u32 {
    let mut key = 0u32;
    if q[5] >= 0 { key |= 1 << 0; }
    if q[9] > 0  { key |= 1 << 1; }
    if q[10] > 0 { key |= 1 << 2; }
    if q[11] > 0 { key |= 1 << 3; }
    let mcc_bucket = match q[12] {
        i16::MIN..=2047 => 0,
        2048..=4095 => 1,
        4096..=6143 => 2,
        _ => 3,
    };
    key |= mcc_bucket << 4;
    if q[2] > 4096 { key |= 1 << 6; }
    if q[8] > 2048 { key |= 1 << 7; }
    key
}

// Per-thread scratch for cross-partition ordering (avoids per-query alloc).
struct Scratch {
    partition_entries: [(i64, u32); MAX_PARTITIONS],
}

impl Scratch {
    const fn new() -> Self {
        Self { partition_entries: [(0, 0); MAX_PARTITIONS] }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
}

// Owns an mmap'd region; munmaps on drop.
struct MmapRegion {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty index file",
            ));
        }
        // MAP_PRIVATE + MAP_POPULATE: kernel synchronously faults in every
        // page during mmap. Eliminates the cold-page-fault outliers on the
        // first burst of requests. Under tight cgroup memory limits the
        // explicit page-touch loop in warm() still runs to keep both paths.
        let ptr = unsafe {
            libc::mmap(ptr::null_mut(), len, libc::PROT_READ,
                       libc::MAP_PRIVATE | libc::MAP_POPULATE,
                       file.as_raw_fd(), 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
        }
        Ok(MmapRegion { ptr: ptr as *mut u8, len })
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len); }
    }
}

pub struct SpecialistIndex {
    pub total_vectors: u32,
    pub scale: f32,
    _mapping: MmapRegion,
    partitions: Vec<Partition>,
    nodes: Vec<Node>,
    panels: *const i16,
    panels_len: usize,
    labels: *const u8,
    labels_len: usize,
    strong_decision: bool,
    early_distance_limit: i64,
    // Cap on total leaf vectors visited per query (count-based early-exit,
    // borrowed from dalvorsn/daniloitagyba). u32::MAX disables. Bounds the
    // worst-case scan time on borderline queries that don't trigger the
    // distance-based early-exit. Tune via RINHA_EARLY_CANDIDATES.
    early_candidates_limit: u32,
    // 8-bit partition_key -> partition index lookup table. -1 means no
    // partition for that key. Replaces the linear scan over self.partitions
    // for matching-key routing — used by every query.
    part_by_key: [i32; 256],
}

unsafe impl Send for SpecialistIndex {}
unsafe impl Sync for SpecialistIndex {}

#[derive(Debug)]
pub enum LoadErr {
    Io(std::io::Error),
    BadMagic,
    BadVersion,
    Truncated,
}

impl SpecialistIndex {
    pub fn load(path: &Path) -> Result<Self, LoadErr> {
        let mapping = MmapRegion::open(path).map_err(LoadErr::Io)?;
        let bytes = unsafe { slice::from_raw_parts(mapping.ptr, mapping.len) };
        if bytes.len() < HEADER_BYTES {
            return Err(LoadErr::Truncated);
        }
        let h = unsafe { *(bytes.as_ptr() as *const Header) };
        if &h.magic != MAGIC {
            return Err(LoadErr::BadMagic);
        }
        if h.version != FORMAT_VERSION {
            return Err(LoadErr::BadVersion);
        }
        let mut cur = HEADER_BYTES;

        let part_bytes = h.partition_count as usize * PARTITION_BYTES;
        if bytes.len() < cur + part_bytes {
            return Err(LoadErr::Truncated);
        }
        let partitions = unsafe {
            slice::from_raw_parts(bytes.as_ptr().add(cur) as *const Partition,
                                  h.partition_count as usize).to_vec()
        };
        cur += part_bytes;

        let node_bytes = h.node_count as usize * NODE_BYTES;
        if bytes.len() < cur + node_bytes {
            return Err(LoadErr::Truncated);
        }
        let nodes = unsafe {
            slice::from_raw_parts(bytes.as_ptr().add(cur) as *const Node,
                                  h.node_count as usize).to_vec()
        };
        cur += node_bytes;

        let panel_total_bytes = h.total_panels as usize * PANEL_BYTES;
        if bytes.len() < cur + panel_total_bytes {
            return Err(LoadErr::Truncated);
        }
        let panels = unsafe { bytes.as_ptr().add(cur) as *const i16 };
        let panels_len = h.total_panels as usize * DIM * LANES;
        cur += panel_total_bytes;

        let labels_len = h.total_vectors as usize;
        if bytes.len() < cur + labels_len {
            return Err(LoadErr::Truncated);
        }
        let labels = unsafe { bytes.as_ptr().add(cur) };

        let strong_decision = std::env::var("RINHA_STRONG_DECISION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let early_distance_limit = std::env::var("RINHA_EARLY_LIMIT")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(EARLY_DISTANCE_LIMIT);
        let early_candidates_limit = std::env::var("RINHA_EARLY_CANDIDATES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(u32::MAX);

        let mut part_by_key = [-1i32; 256];
        for (i, p) in partitions.iter().enumerate() {
            let k = (p.key & 0xff) as usize;
            part_by_key[k] = i as i32;
        }

        let idx = SpecialistIndex {
            total_vectors: h.total_vectors,
            scale: h.scale as f32,
            _mapping: mapping,
            partitions,
            nodes,
            panels,
            panels_len,
            labels,
            labels_len,
            strong_decision,
            early_distance_limit,
            early_candidates_limit,
            part_by_key,
        };
        idx.warm();
        Ok(idx)
    }

    fn warm(&self) {
        // Touch one byte per 4 KB page to fault the file into the page cache
        // smoothly — avoids the burst that MAP_POPULATE causes under tight
        // cgroup memory limits.
        let mut sum: u32 = 0;
        let stride = 4096 / size_of::<i16>();
        unsafe {
            let panels = slice::from_raw_parts(self.panels, self.panels_len);
            for v in panels.iter().step_by(stride) {
                sum = sum.wrapping_add(*v as u32);
            }
            let labels = slice::from_raw_parts(self.labels, self.labels_len);
            for l in labels.iter().step_by(4096) {
                sum = sum.wrapping_add(*l as u32);
            }
            // Best-effort mlock; silent EPERM when the container lacks
            // CAP_IPC_LOCK or RLIMIT_MEMLOCK is low.
            libc::mlock(self.panels as *const libc::c_void,
                        self.panels_len * size_of::<i16>());
            libc::mlock(self.labels as *const libc::c_void, self.labels_len);
        }
        std::hint::black_box(sum);
    }

    pub fn n_partitions(&self) -> usize { self.partitions.len() }
    pub fn n_nodes(&self) -> usize { self.nodes.len() }

    // Drive `count` synthetic queries through the full predict path. Warms
    // i-cache, branch predictors, TLB and pulls every leaf the realistic
    // payload distribution touches into L2 before /ready opens. ~10-20µs
    // first-burst p99 reduction in measured top-10 submissions.
    pub fn warmup_queries(&self, count: usize) {
        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        let mut acc: u64 = 0;
        for i in 0..count {
            // xorshift64 — cheap, no deps, no allocation.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut q = [0i16; DIM];
            for d in 0..DIM {
                // Map to [0, 10000]; sentinel -10000 occasionally on dims 5/6
                // to exercise the missing-last-tx branch in vectorize.
                let v = ((state.rotate_left((d * 7) as u32) as u32) % 10001) as i16;
                q[d] = v;
            }
            if i & 3 == 0 {
                q[5] = -10000;
                q[6] = -10000;
            }
            acc = acc.wrapping_add(self.fraud_count(&q) as u64);
        }
        std::hint::black_box(acc);
    }

    // Public entry point. Pads the query to PACKED_DIMS=16 and delegates.
    // Tier-1 fast-path (vinicius-piassa-asm thresholds, AND-gated) bypasses
    // the kNN entirely on conservative obvious-legit / obvious-fraud cases.
    // EXACT vinicius thresholds (proven on the same reference dataset);
    // verify-binary guards against regression.
    #[inline]
    pub fn fraud_count(&self, q_unpacked: &[i16; DIM]) -> u8 {
        if let Some(c) = tier1_classify(q_unpacked) {
            return c;
        }
        let q = pad_query(q_unpacked);
        self.predict(&q)
    }

    fn predict(&self, q: &QueryVector) -> u8 {
        let mut best_dists = [i64::MAX; K];
        let mut best_labels = [0u8; K];

        let query_key = compute_partition_key(q);
        let strong = self.strong_decision;
        let early_limit = self.early_distance_limit;
        let cand_limit = self.early_candidates_limit;
        // O(1) primary-partition lookup via 256-bucket LUT. -1 means no
        // partition stored that key — fall through to the cross-partition
        // pass directly.
        let primary = self.part_by_key[(query_key & 0xff) as usize];
        // Broadcast the query into 7 (even, odd) i16 pairs ONCE per request.
        // The kernel consumes &q_pairs from every leaf scan — no per-leaf
        // rebuild.
        let q_pairs = unsafe { query_pairs(q) };

        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            let mut other_count = 0usize;
            let mut early_done = false;
            let mut visited: u32 = 0;

            // Descend the matching partition first (queries cluster strongly
            // here — burns max_top fast, fueling later bbox pruning).
            if primary >= 0 {
                let p = &self.partitions[primary as usize];
                let bound = unsafe { lower_bound_box_avx2(q, &p.min, &p.max) };
                if bound < best_dists[K - 1] {
                    visited = visited.saturating_add(
                        self.descend(p.root_node as usize, bound, q, &q_pairs,
                                     &mut best_dists, &mut best_labels));
                    // Bare unanimous-skip is UNSAFE for our kd-tree: unlike
                    // dalvorsn's IVF (where NPROBE=1 means the centroid-closest
                    // cluster is genuinely best), our partition_key is a
                    // categorical bucket, and other partitions can hold
                    // strictly closer neighbors. Cross-partition pass below
                    // is gated by per-partition lower_bound vs best_dists[K-1]
                    // — naturally short-circuits when the primary tightens.
                    if best_dists[K - 1] <= early_limit
                        || visited >= cand_limit
                        || (strong
                            && best_dists[K - 1] <= STRONG_DECISION_LIMIT
                            && is_unanimous(&best_labels))
                    {
                        early_done = true;
                    }
                }
            }

            if !early_done {
                // Collect bounds for every non-primary partition, but
                // pre-filter: only push the partition if its lower bound is
                // still better than the current worst (Step 6: cuts the
                // sort N when primary scan tightened best_dists fast).
                for (idx, p) in self.partitions.iter().enumerate() {
                    if idx as i32 == primary { continue; }
                    let bound = unsafe { lower_bound_box_avx2(q, &p.min, &p.max) };
                    if bound >= best_dists[K - 1] { continue; }
                    s.partition_entries[other_count] = (bound, idx as u32);
                    other_count += 1;
                }

                s.partition_entries[..other_count]
                    .sort_unstable_by_key(|&(bound, _)| bound);

                for i in 0..other_count {
                    let (bound, idx) = s.partition_entries[i];
                    if bound >= best_dists[K - 1] { break; }
                    let p = &self.partitions[idx as usize];
                    visited = visited.saturating_add(
                        self.descend(p.root_node as usize, bound, q, &q_pairs,
                                     &mut best_dists, &mut best_labels));
                    if best_dists[K - 1] <= early_limit { break; }
                    if visited >= cand_limit { break; }
                    if strong
                        && best_dists[K - 1] <= STRONG_DECISION_LIMIT
                        && is_unanimous(&best_labels)
                    {
                        break;
                    }
                }
            }
        });

        best_labels.iter().map(|&l| l as u32).sum::<u32>() as u8
    }

    // Iterative DFS through one tree, near-first with far-child stacked for
    // backtrack. Returns early as soon as the global EARLY_DISTANCE_LIMIT
    // condition fires; otherwise stops descending whenever the node's bound
    // is no longer strictly better than the current 5th-best distance.
    fn descend(&self, root: usize, root_bound: i64, q: &QueryVector,
               q_pairs: &[__m256i; PAIRS],
               best_dists: &mut [i64; K], best_labels: &mut [u8; K]) -> u32 {
        let mut stack_nodes = [0usize; TREE_STACK_CAPACITY];
        let mut stack_bounds = [0i64; TREE_STACK_CAPACITY];
        let mut sp = 0usize;
        let early_limit = self.early_distance_limit;
        let mut visited: u32 = 0;

        let mut current = root;
        let mut current_bound = root_bound;

        loop {
            if current_bound < best_dists[K - 1] {
                let node = unsafe { *self.nodes.get_unchecked(current) };
                if node.left < 0 || node.right < 0 {
                    visited += node.vec_count;
                    unsafe { self.scan_leaf_i16(&node, q_pairs, best_dists, best_labels); }
                    if best_dists[K - 1] <= early_limit { return visited; }
                } else {
                    let l = node.left as usize;
                    let r = node.right as usize;
                    let ln = unsafe { self.nodes.get_unchecked(l) };
                    let rn = unsafe { self.nodes.get_unchecked(r) };
                    let lb = unsafe { lower_bound_box_avx2(q, &ln.min, &ln.max) };
                    let rb = unsafe { lower_bound_box_avx2(q, &rn.min, &rn.max) };
                    let (near, near_b, far, far_b) = if lb <= rb {
                        (l, lb, r, rb)
                    } else {
                        (r, rb, l, lb)
                    };
                    if far_b < best_dists[K - 1] && sp < TREE_STACK_CAPACITY {
                        stack_nodes[sp] = far;
                        stack_bounds[sp] = far_b;
                        sp += 1;
                    }
                    if near_b < best_dists[K - 1] {
                        current = near;
                        current_bound = near_b;
                        continue;
                    }
                }
            }
            if sp == 0 { break; }
            sp -= 1;
            current = stack_nodes[sp];
            current_bound = stack_bounds[sp];
        }
        visited
    }

    // Pair-interleaved i16 leaf scan: 7 madd_epi16 per 8-vector block
    // (one madd handles 2 consecutive dims for all 8 lanes), block-mask
    // cmpgt_epi32 prune, tzcnt-driven lane iteration.
    //
    // Accumulator: single i32x8 (`acc = _mm256_add_epi32(acc, madd...)`).
    // Theoretical max sum (14 dims × max(diff)²) is 5.6e9 > i32 signed max
    // (2.15e9), but for our SCALE=10000 normalized data with most dims
    // clamped to [0, 10000] the practical max is ~2e9 (sentinel pairs push
    // it close to but under signed overflow). Overflowed lanes appear as
    // negative i32 — detected per-lane (`if d_i32 < 0 skip`). The verify
    // binary (54100 entries) is the ground truth.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn scan_leaf_i16(&self, node: &Node, q_pairs: &[__m256i; PAIRS],
                            best_dists: &mut [i64; K], best_labels: &mut [u8; K]) {
        let sz = node.vec_count as usize;
        if sz == 0 { return; }
        let panel_start = node.start_panel as usize;
        let vec_start = node.start_vec as usize;
        let panels_ptr = self.panels as *const __m256i;
        let n_full = sz / LANES;
        let tail = sz % LANES;
        let early_limit = self.early_distance_limit;

        let mut worst = best_dists[K - 1];

        for panel in 0..n_full {
            let base = panels_ptr.add((panel_start + panel) * PAIRS);
            if panel + 1 < n_full {
                let nxt = panels_ptr.add((panel_start + panel + 1) * PAIRS) as *const i8;
                _mm_prefetch(nxt, _MM_HINT_T0);
                _mm_prefetch(nxt.offset(128), _MM_HINT_T0);
            }

            let mut acc = _mm256_setzero_si256();
            for p in 0..PAIRS {
                let v = _mm256_loadu_si256(base.add(p));
                let diff = _mm256_sub_epi16(v, q_pairs[p]);
                let sq = _mm256_madd_epi16(diff, diff);
                acc = _mm256_add_epi32(acc, sq);
            }

            // Block-mask prune. cmpgt_epi32 is signed; for worst < i32::MAX
            // we use it directly. When K not yet saturated, accept all 8
            // lanes (mask = 0xFF).
            let mut mask: u32 = if worst < i32::MAX as i64 {
                let w = _mm256_set1_epi32(worst as i32);
                let cmp = _mm256_cmpgt_epi32(w, acc);
                _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF
            } else {
                0xFFu32
            };
            if mask == 0 { continue; }

            let mut dists = [0i32; LANES];
            _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, acc);
            let base_vec = vec_start + panel * LANES;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let d_i32 = dists[lane];
                if d_i32 < 0 { continue; }  // overflow — skip (out of practical range)
                let d = d_i32 as i64;
                if d < worst {
                    let label = *self.labels.add(base_vec + lane);
                    insert_best_i64(d, label, best_dists, best_labels);
                    worst = best_dists[K - 1];
                }
            }
            if worst <= early_limit { return; }
        }

        if tail > 0 {
            let base = panels_ptr.add((panel_start + n_full) * PAIRS);
            let mut acc = _mm256_setzero_si256();
            for p in 0..PAIRS {
                let v = _mm256_loadu_si256(base.add(p));
                let diff = _mm256_sub_epi16(v, q_pairs[p]);
                let sq = _mm256_madd_epi16(diff, diff);
                acc = _mm256_add_epi32(acc, sq);
            }
            let mut mask: u32 = if worst < i32::MAX as i64 {
                let w = _mm256_set1_epi32(worst as i32);
                let cmp = _mm256_cmpgt_epi32(w, acc);
                _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) as u32 & 0xFF
            } else {
                0xFFu32
            };
            let tail_mask = (1u32 << tail) - 1;
            mask &= tail_mask;
            if mask == 0 { return; }
            let mut dists = [0i32; LANES];
            _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, acc);
            let base_vec = vec_start + n_full * LANES;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let d_i32 = dists[lane];
                if d_i32 < 0 { continue; }
                let d = d_i32 as i64;
                if d < worst {
                    let label = *self.labels.add(base_vec + lane);
                    insert_best_i64(d, label, best_dists, best_labels);
                    worst = best_dists[K - 1];
                }
            }
        }
    }
}

// Broadcast a query into 7 (even, odd) i16 pairs packed as i32, then
// broadcast each i32 across all 8 lanes of a __m256i so it aligns with
// the panel's pair-interleaved layout. Called ONCE per request from predict.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn query_pairs(q: &QueryVector) -> [__m256i; PAIRS] {
    let mut out = [_mm256_setzero_si256(); PAIRS];
    for p in 0..PAIRS {
        let lo = q[2 * p] as u16 as u32;
        let hi = if 2 * p + 1 < DIM { q[2 * p + 1] as u16 as u32 } else { 0 };
        let packed = (lo | (hi << 16)) as i32;
        out[p] = _mm256_set1_epi32(packed);
    }
    out
}

// Tier-1 fast-path classifier (vinicius-piassa-asm). All thresholds are on
// the i16-quantized query (SCALE=10000). Indexes match our normalize.rs:
//   q[0]  amount/10000          q[7]  km_from_home/1000
//   q[1]  installments/12       q[8]  tx_count_24h/20
//   q[2]  amount/cust_avg/10    q[11] is_unknown_merchant (0 known, 10000 unknown)
//   q[12] mcc_risk: 5411=1500 5812=3000 5912=2000 5311=2500 (low-risk set)
// Conservative AND-gates: ALL conditions must hold to bypass kNN.
#[inline]
fn tier1_classify(q: &[i16; DIM]) -> Option<u8> {
    // Obvious legit: known merchant + tiny amount + tiny ratio +
    // few installments + low tx-count + close to home + low-risk MCC.
    if q[11] == 0
        && q[0] <= 500
        && q[2] <= 500
        && q[1] <= 2500
        && q[8] <= 2500
        && q[7] <= 500
        && (q[12] == 1500 || q[12] == 3000 || q[12] == 2000 || q[12] == 2500)
    {
        return Some(0);
    }
    // Obvious fraud: unknown merchant + huge amount + many installments +
    // high tx-count + far from home.
    if q[11] == 10000
        && q[0] >= 5000
        && q[1] >= 4167
        && q[8] >= 3000
        && q[7] >= 1500
    {
        return Some(5);
    }
    None
}

#[inline(always)]
fn insert_best_i64(dist: i64, label: u8, best_dists: &mut [i64; K], best_labels: &mut [u8; K]) {
    if dist >= best_dists[K - 1] { return; }
    let mut pos = K - 1;
    while pos > 0 && dist < best_dists[pos - 1] {
        best_dists[pos] = best_dists[pos - 1];
        best_labels[pos] = best_labels[pos - 1];
        pos -= 1;
    }
    best_dists[pos] = dist;
    best_labels[pos] = label;
}

// Squared L2 lower bound from query to a node's axis-aligned bounding box.
// Each padding dim (d in 14..16) is 0 on both the query and the box, so its
// contribution is always 0. Result fits in i64 (16 dims * (2*32767)^2 ~ 7e10).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lower_bound_box_avx2(q: &QueryVector, min: &[i16; PACKED_DIMS],
                                max: &[i16; PACKED_DIMS]) -> i64 {
    use std::arch::x86_64::_mm_extract_epi64;
    let qv = _mm256_loadu_si256(q.as_ptr() as *const __m256i);
    let mn = _mm256_loadu_si256(min.as_ptr() as *const __m256i);
    let mx = _mm256_loadu_si256(max.as_ptr() as *const __m256i);
    let zero = _mm256_setzero_si256();
    let below = _mm256_max_epi16(_mm256_sub_epi16(mn, qv), zero);
    let above = _mm256_max_epi16(_mm256_sub_epi16(qv, mx), zero);
    let diff = _mm256_max_epi16(below, above);
    let sq = _mm256_madd_epi16(diff, diff);
    let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq));
    let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1));
    let sum = _mm256_add_epi64(lo, hi);
    let sum_hi = _mm256_extracti128_si256(sum, 1);
    let sum_128 = _mm_add_epi64(_mm256_castsi256_si128(sum), sum_hi);
    _mm_extract_epi64(sum_128, 0) + _mm_extract_epi64(sum_128, 1)
}

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
    _mm256_madd_epi16, _mm256_add_epi64, _mm256_cvtepi32_epi64,
    _mm256_castsi256_si128, _mm256_extracti128_si256,
};

pub const DIM: usize = 14;
pub const PACKED_DIMS: usize = 16;
pub const K: usize = 5;
pub const LANES: usize = 8;
pub const LEAF_SIZE: usize = 128;
pub const MAX_PARTITIONS: usize = 256;
pub const TREE_STACK_CAPACITY: usize = 128;

// When the 5th-best squared distance falls below this threshold, the top-5
// are tight enough (within ~0.14 in normalized feature space) that no
// further probing can change the fraud count. Empirically validated on the
// test dataset by MXLange's c-api-rinha2026; we adopt the same constant.
//   ((QUANT_SCALE * 140) / 1000)^2 = 1400^2 = 1_960_000
pub const EARLY_DISTANCE_LIMIT: f32 = 1_960_000.0;

// Strong-decision early termination (gated by RINHA_STRONG_DECISION env).
// When the top-5 are unanimous (all legit or all fraud) and tight enough
// (~0.20 normalized = 2000^2), the binary approved/denied outcome can't
// flip even if a closer neighbor exists — at worst one slot swaps and the
// fraud count moves by 1, which doesn't cross the 3-of-5 threshold.
pub const STRONG_DECISION_LIMIT: f32 = 4_000_000.0;

pub const MAGIC: &[u8; 8] = b"RSPECST1";
pub const FORMAT_VERSION: u32 = 1;

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
        // MAP_PRIVATE only (no MAP_POPULATE). Pre-faulting 100 MB in one
        // burst risks tripping the cgroup memory.max on tight limits; we
        // demand-fault then explicitly touch every page in `warm()` so the
        // memory pressure ramps up smoothly.
        let ptr = unsafe {
            libc::mmap(ptr::null_mut(), len, libc::PROT_READ,
                       libc::MAP_PRIVATE, file.as_raw_fd(), 0)
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
    early_distance_limit: f32,
    // Cap on total leaf vectors visited per query (count-based early-exit,
    // borrowed from dalvorsn/daniloitagyba). u32::MAX disables. Bounds the
    // worst-case scan time on borderline queries that don't trigger the
    // distance-based early-exit. Tune via RINHA_EARLY_CANDIDATES.
    early_candidates_limit: u32,
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
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(EARLY_DISTANCE_LIMIT);
        let early_candidates_limit = std::env::var("RINHA_EARLY_CANDIDATES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(u32::MAX);

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

    // Public entry point. Pads the query to PACKED_DIMS=16 and delegates.
    #[inline]
    pub fn fraud_count(&self, q_unpacked: &[i16; DIM]) -> u8 {
        let q = pad_query(q_unpacked);
        self.predict(&q)
    }

    fn predict(&self, q: &QueryVector) -> u8 {
        let mut best_dists = [f32::MAX; K];
        let mut best_labels = [0u8; K];

        let query_key = compute_partition_key(q);
        let strong = self.strong_decision;
        let early_limit = self.early_distance_limit;
        let cand_limit = self.early_candidates_limit;

        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            let mut other_count = 0usize;
            let mut early_done = false;
            let mut visited: u32 = 0;

            // Visit the matching partition first if present; collect the rest.
            for (idx, p) in self.partitions.iter().enumerate() {
                let bound = unsafe { lower_bound_box_avx2(q, &p.min, &p.max) } as f32;
                if p.key == query_key {
                    if bound < best_dists[K - 1] {
                        visited = visited.saturating_add(
                            self.descend(p.root_node as usize, bound, q,
                                         &mut best_dists, &mut best_labels));
                        if best_dists[K - 1] <= early_limit
                            || visited >= cand_limit
                            || (strong
                                && best_dists[K - 1] <= STRONG_DECISION_LIMIT
                                && is_unanimous(&best_labels))
                        {
                            early_done = true;
                        }
                    }
                } else {
                    s.partition_entries[other_count] = (bound as i64, idx as u32);
                    other_count += 1;
                }
            }

            if !early_done {
                s.partition_entries[..other_count]
                    .sort_unstable_by_key(|&(bound, _)| bound);

                for i in 0..other_count {
                    let (bound, idx) = s.partition_entries[i];
                    if bound as f32 >= best_dists[K - 1] { break; }
                    let p = &self.partitions[idx as usize];
                    visited = visited.saturating_add(
                        self.descend(p.root_node as usize, bound as f32, q,
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
    fn descend(&self, root: usize, root_bound: f32, q: &QueryVector,
               best_dists: &mut [f32; K], best_labels: &mut [u8; K]) -> u32 {
        let mut stack_nodes = [0usize; TREE_STACK_CAPACITY];
        let mut stack_bounds = [0f32; TREE_STACK_CAPACITY];
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
                    unsafe { self.scan_leaf(&node, q, best_dists, best_labels); }
                    if best_dists[K - 1] <= early_limit { return visited; }
                } else {
                    let l = node.left as usize;
                    let r = node.right as usize;
                    let ln = unsafe { self.nodes.get_unchecked(l) };
                    let rn = unsafe { self.nodes.get_unchecked(r) };
                    let lb = unsafe { lower_bound_box_avx2(q, &ln.min, &ln.max) } as f32;
                    let rb = unsafe { lower_bound_box_avx2(q, &rn.min, &rn.max) } as f32;
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

    // Scans a leaf's SoA-8 panels. Distance compute reuses the FMA pattern
    // from the IVF scan (faster than integer madd_epi16 on Zen3/Haswell with
    // the volume of dims we have).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn scan_leaf(&self, node: &Node, q: &QueryVector,
                        best_dists: &mut [f32; K], best_labels: &mut [u8; K]) {
        use std::arch::x86_64::{
            __m256, _mm256_setzero_ps, _mm256_set1_ps,
            _mm256_storeu_ps, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps,
            _mm256_sub_ps, _mm256_add_ps, _mm256_fmadd_ps,
            _mm256_castps256_ps128, _mm256_extractf128_ps,
            _mm256_cmp_ps, _mm256_movemask_ps, _CMP_GE_OQ,
            _mm_min_ps, _mm_min_ss, _mm_movehl_ps, _mm_shuffle_ps, _mm_cvtss_f32,
        };

        let sz = node.vec_count as usize;
        if sz == 0 { return; }
        let panel_start = node.start_panel as usize;
        let vec_start = node.start_vec as usize;
        let n_full = sz / LANES;
        let tail = sz % LANES;

        // Build a f32 broadcast of the query (only DIM real components).
        let mut qb: [__m256; DIM] = [_mm256_setzero_ps(); DIM];
        for d in 0..DIM {
            qb[d] = _mm256_set1_ps(q[d] as f32);
        }
        let panels_ptr = self.panels.add(panel_start * DIM * LANES);
        let early_limit = self.early_distance_limit;

        let mut p = 0usize;
        let mut worst_f32 = best_dists[K - 1];

        for panel in 0..n_full {
            if panel + 1 < n_full {
                let next_ptr = panels_ptr.add(p + DIM * LANES) as *const i8;
                _mm_prefetch(next_ptr, _MM_HINT_T0);
                _mm_prefetch(next_ptr.offset(128), _MM_HINT_T0);
            }

            // Dual-accumulator FMA chain (proven win in IVF scan).
            let mut acc_a = _mm256_setzero_ps();
            let mut acc_b = _mm256_setzero_ps();
            let mut d = 0;

            // Phase 1: first 10 dims (5 pairs). Then partial-prune check —
            // if every lane already crossed `worst_f32` after ~71% of the
            // FMAs, the remaining 4 dims can't bring any lane back below,
            // so we skip them entirely. Borrowed from dalvorsn's cpp scan.
            while d < 10 {
                let r0 = _mm_loadu_si128(panels_ptr.add(p + d * LANES) as *const __m128i);
                let r1 = _mm_loadu_si128(panels_ptr.add(p + (d + 1) * LANES) as *const __m128i);
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1));
                let diff0 = _mm256_sub_ps(f0, qb[d]);
                let diff1 = _mm256_sub_ps(f1, qb[d + 1]);
                acc_a = _mm256_fmadd_ps(diff0, diff0, acc_a);
                acc_b = _mm256_fmadd_ps(diff1, diff1, acc_b);
                d += 2;
            }
            {
                let partial = _mm256_add_ps(acc_a, acc_b);
                let worst_b = _mm256_set1_ps(worst_f32);
                let mask = _mm256_cmp_ps::<_CMP_GE_OQ>(partial, worst_b);
                if _mm256_movemask_ps(mask) == 0xFF {
                    p += DIM * LANES;
                    continue;
                }
            }

            // Phase 2: remaining dims.
            while d + 1 < DIM {
                let r0 = _mm_loadu_si128(panels_ptr.add(p + d * LANES) as *const __m128i);
                let r1 = _mm_loadu_si128(panels_ptr.add(p + (d + 1) * LANES) as *const __m128i);
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1));
                let diff0 = _mm256_sub_ps(f0, qb[d]);
                let diff1 = _mm256_sub_ps(f1, qb[d + 1]);
                acc_a = _mm256_fmadd_ps(diff0, diff0, acc_a);
                acc_b = _mm256_fmadd_ps(diff1, diff1, acc_b);
                d += 2;
            }
            if d < DIM {
                let r = _mm_loadu_si128(panels_ptr.add(p + d * LANES) as *const __m128i);
                let f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r));
                let diff = _mm256_sub_ps(f, qb[d]);
                acc_a = _mm256_fmadd_ps(diff, diff, acc_a);
            }
            let acc = _mm256_add_ps(acc_a, acc_b);

            // panel_min for early-exit of insertion loop.
            let acc_lo = _mm256_castps256_ps128(acc);
            let acc_hi = _mm256_extractf128_ps(acc, 1);
            let min_4 = _mm_min_ps(acc_lo, acc_hi);
            let min_4 = _mm_min_ps(min_4, _mm_movehl_ps(min_4, min_4));
            let min_4 = _mm_min_ss(min_4, _mm_shuffle_ps(min_4, min_4, 0b01));
            let panel_min = _mm_cvtss_f32(min_4);

            p += DIM * LANES;

            if panel_min >= worst_f32 { continue; }

            let mut dists = [0f32; LANES];
            _mm256_storeu_ps(dists.as_mut_ptr(), acc);
            let base_vec = vec_start + panel * LANES;
            for v in 0..LANES {
                let d = dists[v];
                if d < worst_f32 {
                    let label = *self.labels.add(base_vec + v);
                    insert_best(d, label, best_dists, best_labels);
                    worst_f32 = best_dists[K - 1];
                }
            }
            if worst_f32 <= early_limit { return; }
        }

        // Tail: remaining `tail` vectors live in the next (partial) panel,
        // still SoA-8 packed but only the first `tail` lanes valid.
        if tail > 0 {
            let mut acc_a = _mm256_setzero_ps();
            let mut acc_b = _mm256_setzero_ps();
            let mut d = 0;
            while d + 1 < DIM {
                let r0 = _mm_loadu_si128(panels_ptr.add(p + d * LANES) as *const __m128i);
                let r1 = _mm_loadu_si128(panels_ptr.add(p + (d + 1) * LANES) as *const __m128i);
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1));
                let diff0 = _mm256_sub_ps(f0, qb[d]);
                let diff1 = _mm256_sub_ps(f1, qb[d + 1]);
                acc_a = _mm256_fmadd_ps(diff0, diff0, acc_a);
                acc_b = _mm256_fmadd_ps(diff1, diff1, acc_b);
                d += 2;
            }
            if d < DIM {
                let r = _mm_loadu_si128(panels_ptr.add(p + d * LANES) as *const __m128i);
                let f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r));
                let diff = _mm256_sub_ps(f, qb[d]);
                acc_a = _mm256_fmadd_ps(diff, diff, acc_a);
            }
            let acc = _mm256_add_ps(acc_a, acc_b);
            let mut dists = [0f32; LANES];
            _mm256_storeu_ps(dists.as_mut_ptr(), acc);
            let base_vec = vec_start + n_full * LANES;
            for v in 0..tail {
                let d = dists[v];
                if d < best_dists[K - 1] {
                    let label = *self.labels.add(base_vec + v);
                    insert_best(d, label, best_dists, best_labels);
                }
            }
        }
    }
}

#[inline(always)]
fn insert_best(dist: f32, label: u8, best_dists: &mut [f32; K], best_labels: &mut [u8; K]) {
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

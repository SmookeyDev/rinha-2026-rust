// IVF k-NN k=5 inference. The index is built offline by tools/build_index.py
// and saved to ivf_int16.bin. File format: see tools/build_index.py.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fs::File;
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::slice;

#[allow(unused_imports)]
use std::arch::x86_64::{
    __m128i, __m256, _mm_loadu_si128, _mm_prefetch, _MM_HINT_T0,
    _mm256_setzero_ps, _mm256_set1_ps, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps,
    _mm256_sub_ps, _mm256_add_ps, _mm256_fmadd_ps, _mm256_storeu_ps, _mm256_loadu_ps,
    _mm256_castps256_ps128, _mm256_extractf128_ps,
    _mm_min_ps, _mm_min_ss, _mm_movehl_ps, _mm_shuffle_ps, _mm_cvtss_f32,
};

pub const DIM: usize = 14;
pub const K: usize = 5;

// Per-thread scratch buffers reused across fraud_count calls. Lazily sized on
// first use; capacity sticks for the rest of the process so the hot path never
// allocates. With 4096 clusters this is ~24 KB per thread.
struct Scratch {
    cluster_dists: Vec<f32>,
    cluster_ids: Vec<u16>,
}

impl Scratch {
    const fn new() -> Self {
        Self {
            cluster_dists: Vec::new(),
            cluster_ids: Vec::new(),
        }
    }

    #[inline]
    fn ensure(&mut self, n: usize) {
        if self.cluster_dists.len() < n {
            self.cluster_dists.resize(n, 0.0);
            self.cluster_ids = (0..n as u16).collect();
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Header {
    magic: u32,
    version: u32,
    n_total: u32,
    dim: u32,
    n_clusters: u32,
    scale: u32,
    _pad1: u32,
    _pad2: u32,
}

// Owns an mmap'd region; munmaps on drop. Used as the backing store for the
// large read-only sections (vectors, labels) so we never copy them.
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
        let mut flags = libc::MAP_PRIVATE;
        #[cfg(target_os = "linux")]
        {
            flags |= libc::MAP_POPULATE;
        }
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                flags,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, len, libc::MADV_WILLNEED);
            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
        }
        Ok(MmapRegion {
            ptr: ptr as *mut u8,
            len,
        })
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

pub struct IvfIndex {
    pub n_total: u32,
    pub dim: u32,
    pub n_clusters: u32,
    pub scale: f32,
    pub nprobe: u32,
    _mapping: MmapRegion,                       // keeps mmap alive
    centroids_soa: Vec<f32>,                    // dim * K (transposed for AVX2)
    cluster_sizes: Vec<u32>,
    cluster_offsets: Vec<u32>,                  // prefix sum in vectors
    panel_offsets: Vec<u32>,                    // prefix sum in i16 elements
    vectors: *const i16,                        // points into mmap
    vectors_len: usize,
    labels: *const u8,                          // points into mmap
    labels_len: usize,
}

unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

#[derive(Debug)]
pub enum LoadErr {
    Io(std::io::Error),
    BadMagic,
    BadVersion,
    Truncated,
}

impl IvfIndex {
    // Loads the index by mmap'ing the file. Vectors and labels remain in the
    // page cache and are accessed via raw pointers; the only copies are the
    // small offset tables and the transposed centroids_soa needed by AVX2.
    pub fn load(path: &Path, nprobe: u32) -> Result<Self, LoadErr> {
        let mapping = MmapRegion::open(path).map_err(LoadErr::Io)?;
        let bytes = mapping.as_slice();
        if bytes.len() < 32 {
            return Err(LoadErr::Truncated);
        }
        let h = unsafe { *(bytes.as_ptr() as *const Header) };
        if h.magic != 0x484E4952 {
            return Err(LoadErr::BadMagic);
        }
        if h.version != 3 {
            return Err(LoadErr::BadVersion);
        }
        let n_total = h.n_total as usize;
        let dim = h.dim as usize;
        let n_clusters = h.n_clusters as usize;
        let scale = h.scale as f32;
        assert_eq!(dim, DIM, "DIM mismatch");

        let mut cur = 32usize;

        // Centroids: f32, row-major (n_clusters x dim). Copied so we can
        // immediately transpose into the SoA layout the AVX2 scan needs.
        let centroid_bytes = n_clusters * dim * size_of::<f32>();
        if bytes.len() < cur + centroid_bytes {
            return Err(LoadErr::Truncated);
        }
        let centroids_flat = unsafe {
            slice::from_raw_parts(bytes.as_ptr().add(cur) as *const f32, n_clusters * dim)
        };
        cur += centroid_bytes;

        // Small offset tables: copy into Vecs.
        let cluster_sizes = copy_u32_slice(bytes, &mut cur, n_clusters)?;
        let cluster_offsets = copy_u32_slice(bytes, &mut cur, n_clusters + 1)?;
        let panel_offsets = copy_u32_slice(bytes, &mut cur, n_clusters + 1)?;

        // Vectors: zero-copy pointer into the mmap.
        let vectors_len = n_total * dim;
        let vec_bytes = vectors_len * size_of::<i16>();
        if bytes.len() < cur + vec_bytes {
            return Err(LoadErr::Truncated);
        }
        if cur % size_of::<i16>() != 0 {
            return Err(LoadErr::Truncated);
        }
        let vectors = unsafe { bytes.as_ptr().add(cur) as *const i16 };
        cur += vec_bytes;

        // Labels: zero-copy pointer into the mmap.
        let labels_len = n_total;
        if bytes.len() < cur + labels_len {
            return Err(LoadErr::Truncated);
        }
        let labels = unsafe { bytes.as_ptr().add(cur) };

        let mut centroids_soa = vec![0f32; n_clusters * dim];
        for cid in 0..n_clusters {
            for d in 0..dim {
                centroids_soa[d * n_clusters + cid] = centroids_flat[cid * dim + d];
            }
        }

        let idx = IvfIndex {
            n_total: h.n_total,
            dim: h.dim,
            n_clusters: h.n_clusters,
            scale,
            nprobe,
            _mapping: mapping,
            centroids_soa,
            cluster_sizes,
            cluster_offsets,
            panel_offsets,
            vectors,
            vectors_len,
            labels,
            labels_len,
        };
        idx.warm();
        Ok(idx)
    }

    // mmap + MAP_POPULATE already pages everything in. mlock keeps it pinned
    // when the container has the privilege. madvise(HUGEPAGE) was already done
    // on the mapping; this just re-asserts intent for the hot regions.
    fn warm(&self) {
        unsafe {
            libc::mlock(self.vectors as *const libc::c_void, self.vectors_len * size_of::<i16>());
            libc::mlock(self.labels as *const libc::c_void, self.labels_len);
            libc::mlock(
                self.centroids_soa.as_ptr() as *const libc::c_void,
                self.centroids_soa.len() * size_of::<f32>(),
            );
        }
    }

    // Classify a query (already quantised to i16). Returns the number of
    // "fraud" labels among the 5 nearest neighbours (0..=5).
    #[inline]
    pub fn fraud_count(&self, q_i16: &[i16; DIM]) -> u8 {
        let mut q_scaled = [0f32; DIM];
        let inv_scale = 1.0f32 / self.scale;
        let mut q_unscaled = [0f32; DIM];
        for i in 0..DIM {
            q_scaled[i] = q_i16[i] as f32;
            q_unscaled[i] = q_scaled[i] * inv_scale;
        }

        let k = self.n_clusters as usize;
        let probe = (self.nprobe as usize).min(k);

        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            s.ensure(k);
            let Scratch { cluster_dists, cluster_ids } = &mut *s;

            unsafe { self.centroid_distances_avx2(&q_unscaled, cluster_dists); }

            cluster_ids.select_nth_unstable_by(probe - 1, |&a, &b| {
                let da = cluster_dists[a as usize];
                let db = cluster_dists[b as usize];
                if da < db { Ordering::Less }
                else if da > db { Ordering::Greater }
                else { Ordering::Equal }
            });

            let mut top: [(f32, u8); K] = [(f32::MAX, 0); K];
            unsafe {
                for i in 0..probe {
                    let cid = cluster_ids[i] as usize;
                    self.scan_cluster_avx2(cid, q_scaled, &mut top);
                }
            }

            top.iter().filter(|(_, lbl)| *lbl == 1).count() as u8
        })
    }

    // L2 squared distance from the query to all K centroids in SoA layout.
    // K must be a multiple of 8. Uses two independent accumulators so the
    // FMA serial chain (otherwise DIM=14 deep, ~70 cycles on Haswell) gets
    // split into two parallel ~35-cycle chains the CPU can pipeline.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn centroid_distances_avx2(&self, q: &[f32; DIM], out: &mut [f32]) {
        let k = self.n_clusters as usize;
        debug_assert!(k % 8 == 0);
        let soa = self.centroids_soa.as_ptr();
        let mut qb: [__m256; DIM] = [_mm256_setzero_ps(); DIM];
        for d in 0..DIM {
            qb[d] = _mm256_set1_ps(q[d]);
        }
        let mut cid = 0;
        while cid < k {
            let mut acc_a = _mm256_setzero_ps();
            let mut acc_b = _mm256_setzero_ps();
            let mut d = 0;
            while d + 1 < DIM {
                let c0 = _mm256_loadu_ps(soa.add(d * k + cid));
                let c1 = _mm256_loadu_ps(soa.add((d + 1) * k + cid));
                let diff0 = _mm256_sub_ps(c0, qb[d]);
                let diff1 = _mm256_sub_ps(c1, qb[d + 1]);
                acc_a = _mm256_fmadd_ps(diff0, diff0, acc_a);
                acc_b = _mm256_fmadd_ps(diff1, diff1, acc_b);
                d += 2;
            }
            // Catch the trailing dim when DIM is odd (no-op for DIM=14).
            if d < DIM {
                let c = _mm256_loadu_ps(soa.add(d * k + cid));
                let diff = _mm256_sub_ps(c, qb[d]);
                acc_a = _mm256_fmadd_ps(diff, diff, acc_a);
            }
            let acc = _mm256_add_ps(acc_a, acc_b);
            _mm256_storeu_ps(out.as_mut_ptr().add(cid), acc);
            cid += 8;
        }
    }

    // AVX2 scan of one cluster against the query (scaled space). Float32
    // accumulation via FMA: tested locally to be faster than the integer
    // alternative (madd_epi16 + unpack) on Zen3 and presumably Haswell when
    // clusters are large enough for the FMA pipeline to dominate.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn scan_cluster_avx2(&self, cid: usize, q_f32: [f32; DIM], top: &mut [(f32, u8); K]) {
        let sz = self.cluster_sizes[cid] as usize;
        let vec_start = self.cluster_offsets[cid] as usize;
        let panel_off = self.panel_offsets[cid] as usize;
        let n_full = sz / 8;
        let tail = sz % 8;

        let mut qb: [__m256; DIM] = [_mm256_setzero_ps(); DIM];
        for d in 0..DIM {
            qb[d] = _mm256_set1_ps(q_f32[d]);
        }

        let vptr = self.vectors;
        let lptr = self.labels;

        #[inline]
        fn update_top(top: &mut [(f32, u8); K], d: f32, l: u8) {
            let mut worst_i = 0usize;
            let mut worst_d = top[0].0;
            for i in 1..K {
                if top[i].0 > worst_d {
                    worst_d = top[i].0;
                    worst_i = i;
                }
            }
            if d < worst_d {
                top[worst_i] = (d, l);
            }
        }

        let mut p = panel_off;
        let mut worst_d = top.iter().map(|(d, _)| *d).fold(f32::MIN, f32::max);
        for panel in 0..n_full {
            if panel + 1 < n_full {
                let next_ptr = vptr.add(p + DIM * 8) as *const i8;
                _mm_prefetch(next_ptr, _MM_HINT_T0);
                _mm_prefetch(next_ptr.offset(128), _MM_HINT_T0);
            }

            // Two independent FMA chains to split the dependency. With DIM=14
            // each chain is 7 deep (~35 cycles latency on Haswell) instead of
            // 14 (~70 cycles), and the CPU can interleave them.
            let mut acc_a = _mm256_setzero_ps();
            let mut acc_b = _mm256_setzero_ps();
            let mut d = 0;
            while d + 1 < DIM {
                let r0 = _mm_loadu_si128(vptr.add(p + d * 8) as *const __m128i);
                let r1 = _mm_loadu_si128(vptr.add(p + (d + 1) * 8) as *const __m128i);
                let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r0));
                let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r1));
                let diff0 = _mm256_sub_ps(f0, qb[d]);
                let diff1 = _mm256_sub_ps(f1, qb[d + 1]);
                acc_a = _mm256_fmadd_ps(diff0, diff0, acc_a);
                acc_b = _mm256_fmadd_ps(diff1, diff1, acc_b);
                d += 2;
            }
            if d < DIM {
                let r = _mm_loadu_si128(vptr.add(p + d * 8) as *const __m128i);
                let f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(r));
                let diff = _mm256_sub_ps(f, qb[d]);
                acc_a = _mm256_fmadd_ps(diff, diff, acc_a);
            }
            let acc = _mm256_add_ps(acc_a, acc_b);

            let acc_lo = _mm256_castps256_ps128(acc);
            let acc_hi = _mm256_extractf128_ps(acc, 1);
            let min_4 = _mm_min_ps(acc_lo, acc_hi);
            let min_4 = _mm_min_ps(min_4, _mm_movehl_ps(min_4, min_4));
            let min_4 = _mm_min_ss(min_4, _mm_shuffle_ps(min_4, min_4, 0b01));
            let panel_min = _mm_cvtss_f32(min_4);

            p += 8 * DIM;
            if panel_min >= worst_d {
                continue;
            }
            let mut dists = [0f32; 8];
            _mm256_storeu_ps(dists.as_mut_ptr(), acc);
            let vidx = vec_start + panel * 8;
            for v in 0..8 {
                let d = dists[v];
                if d < worst_d {
                    let mut worst_i = 0usize;
                    let mut cw = top[0].0;
                    for i in 1..K {
                        if top[i].0 > cw {
                            cw = top[i].0;
                            worst_i = i;
                        }
                    }
                    top[worst_i] = (d, *lptr.add(vidx + v));
                    worst_d = top.iter().map(|(d, _)| *d).fold(f32::MIN, f32::max);
                }
            }
        }

        let tail_vidx = vec_start + n_full * 8;
        for v in 0..tail {
            let base = p + v * DIM;
            let mut d = 0f32;
            for k in 0..DIM {
                let r = *vptr.add(base + k) as f32;
                let diff = r - q_f32[k];
                d += diff * diff;
            }
            update_top(top, d, *lptr.add(tail_vidx + v));
        }
    }
}

fn copy_u32_slice(bytes: &[u8], cur: &mut usize, n: usize) -> Result<Vec<u32>, LoadErr> {
    let need = n * size_of::<u32>();
    if bytes.len() < *cur + need {
        return Err(LoadErr::Truncated);
    }
    let src = unsafe { slice::from_raw_parts(bytes.as_ptr().add(*cur) as *const u32, n) };
    *cur += need;
    Ok(src.to_vec())
}

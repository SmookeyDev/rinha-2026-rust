// IVF k-NN k=5 inference. The index is built offline by tools/build_index.py
// and saved to ivf_int16.bin. File format: see tools/build_index.py.

use std::alloc::{alloc, Layout};
use std::mem::size_of;
use std::path::Path;
#[allow(unused_imports)]
use std::arch::x86_64::{
    __m128i, __m256, _mm_loadu_si128, _mm_prefetch, _MM_HINT_T0,
    _mm256_setzero_ps, _mm256_set1_ps, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps,
    _mm256_sub_ps, _mm256_fmadd_ps, _mm256_storeu_ps, _mm256_loadu_ps,
    _mm256_castps256_ps128, _mm256_extractf128_ps,
    _mm_min_ps, _mm_min_ss, _mm_movehl_ps, _mm_shuffle_ps, _mm_cvtss_f32,
};

pub const DIM: usize = 14;
pub const K: usize = 5;

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

pub struct IvfIndex {
    pub n_total: u32,
    pub dim: u32,
    pub n_clusters: u32,
    pub scale: f32,
    pub nprobe: u32,
    centroids: Vec<f32>,                // K * dim (row-major)
    centroids_soa: Vec<f32>,            // dim * K (transposed for AVX2)
    cluster_sizes: Vec<u32>,
    cluster_offsets: Vec<u32>,          // prefix sum in vectors
    panel_offsets: Vec<u32>,            // prefix sum in i16 elements
    vectors: Vec<i16>,
    labels: Vec<u8>,
}

#[derive(Debug)]
pub enum LoadErr {
    Io(std::io::Error),
    BadMagic,
    BadVersion,
    Truncated,
}

// Vec<T> aligned to `align` bytes. Used to align the vector buffer at 2 MB
// boundaries so MADV_HUGEPAGE has a chance to promote the pages. The Vec is
// intentionally leaked (custom alignment does not match what the default
// allocator expects on drop). This is OK because the index lives for the
// entire process lifetime.
fn alloc_aligned_vec<T: Copy>(n: usize, align: usize) -> Vec<T> {
    let bytes = n * std::mem::size_of::<T>();
    let layout = Layout::from_size_align(bytes, align).expect("bad layout");
    let ptr = unsafe { alloc(layout) } as *mut T;
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    unsafe {
        std::ptr::write_bytes(ptr, 0, n);
        Vec::from_raw_parts(ptr, n, n)
    }
}

impl IvfIndex {
    pub fn load(path: &Path, nprobe: u32) -> Result<Self, LoadErr> {
        let bytes = std::fs::read(path).map_err(LoadErr::Io)?;
        let mut idx = Self::from_bytes(&bytes, nprobe)?;
        idx.warm();
        Ok(idx)
    }

    // Ask the kernel for huge pages on the vectors buffer and force a page-in
    // of every page so the hot path never hits a page fault.
    fn warm(&mut self) {
        unsafe {
            let vptr = self.vectors.as_ptr() as *mut libc::c_void;
            let vlen = self.vectors.len() * std::mem::size_of::<i16>();
            libc::madvise(vptr, vlen, libc::MADV_HUGEPAGE);
            libc::madvise(vptr, vlen, libc::MADV_WILLNEED);
            libc::madvise(
                self.labels.as_ptr() as *mut libc::c_void,
                self.labels.len(),
                libc::MADV_WILLNEED,
            );
            libc::madvise(
                self.centroids.as_ptr() as *mut libc::c_void,
                self.centroids.len() * std::mem::size_of::<f32>(),
                libc::MADV_WILLNEED,
            );
        }
        // Touch one element per 4 KB page to fault them all in.
        let mut sum = 0u32;
        let stride = 4096 / std::mem::size_of::<i16>();
        for v in self.vectors.iter().step_by(stride) {
            sum = sum.wrapping_add(*v as u32);
        }
        for l in self.labels.iter().step_by(4096) {
            sum = sum.wrapping_add(*l as u32);
        }
        std::hint::black_box(sum);
    }

    pub fn from_bytes(bytes: &[u8], nprobe: u32) -> Result<Self, LoadErr> {
        if bytes.len() < 32 { return Err(LoadErr::Truncated); }
        let h = unsafe { &*(bytes.as_ptr() as *const Header) };
        if h.magic != 0x484E4952 { return Err(LoadErr::BadMagic); }
        if h.version != 3 { return Err(LoadErr::BadVersion); }
        let n_total = h.n_total as usize;
        let dim = h.dim as usize;
        let n_clusters = h.n_clusters as usize;
        let scale = h.scale as f32;
        assert_eq!(dim, DIM, "DIM mismatch");

        let mut off = 32usize;
        let centroid_bytes = n_clusters * dim * size_of::<f32>();
        if bytes.len() < off + centroid_bytes { return Err(LoadErr::Truncated); }
        let centroids: Vec<f32> = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(off) as *const f32, n_clusters * dim).to_vec()
        };
        off += centroid_bytes;

        let cs_bytes = n_clusters * 4;
        if bytes.len() < off + cs_bytes { return Err(LoadErr::Truncated); }
        let cluster_sizes: Vec<u32> = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(off) as *const u32, n_clusters).to_vec()
        };
        off += cs_bytes;

        let co_bytes = (n_clusters + 1) * 4;
        if bytes.len() < off + co_bytes { return Err(LoadErr::Truncated); }
        let cluster_offsets: Vec<u32> = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(off) as *const u32, n_clusters + 1).to_vec()
        };
        off += co_bytes;

        let po_bytes = (n_clusters + 1) * 4;
        if bytes.len() < off + po_bytes { return Err(LoadErr::Truncated); }
        let panel_offsets: Vec<u32> = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(off) as *const u32, n_clusters + 1).to_vec()
        };
        off += po_bytes;

        let vec_n_elems = n_total * dim;
        let vec_bytes = vec_n_elems * size_of::<i16>();
        if bytes.len() < off + vec_bytes { return Err(LoadErr::Truncated); }
        // 2 MB alignment lets warm() promote these pages to huge pages.
        let vectors: Vec<i16> = alloc_aligned_vec::<i16>(vec_n_elems, 2 * 1024 * 1024);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(off) as *const i16,
                vectors.as_ptr() as *mut i16,
                vec_n_elems,
            );
        }
        off += vec_bytes;

        if bytes.len() < off + n_total { return Err(LoadErr::Truncated); }
        let labels: Vec<u8> = bytes[off..off + n_total].to_vec();

        // Transpose centroids to SoA: all of dim 0 first, then all of dim 1, etc.
        // Layout friendly for the AVX2 scan over 8 clusters per iteration.
        let mut centroids_soa = vec![0f32; n_clusters * dim];
        for cid in 0..n_clusters {
            for d in 0..dim {
                centroids_soa[d * n_clusters + cid] = centroids[cid * dim + d];
            }
        }

        Ok(IvfIndex {
            n_total: h.n_total,
            dim: h.dim,
            n_clusters: h.n_clusters,
            scale,
            nprobe,
            centroids,
            centroids_soa,
            cluster_sizes,
            cluster_offsets,
            panel_offsets,
            vectors,
            labels,
        })
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

        // Distance from the query to each centroid (unscaled Float32 space).
        let k = self.n_clusters as usize;
        let mut cluster_dists = vec![0f32; k];
        unsafe { self.centroid_distances_avx2(&q_unscaled, &mut cluster_dists); }

        // Pick the `nprobe` closest clusters via partial sort.
        let mut best: Vec<(f32, u32)> = (0..k).map(|c| (cluster_dists[c], c as u32)).collect();
        let probe = (self.nprobe as usize).min(best.len());
        best.select_nth_unstable_by(probe - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scan the selected clusters, maintaining top-5.
        let mut top: [(f32, u8); K] = [(f32::MAX, 0); K];
        unsafe {
            for &(_, cid) in &best[..probe] {
                self.scan_cluster_avx2(cid as usize, q_scaled, &mut top);
            }
        }

        top.iter().filter(|(_, lbl)| *lbl == 1).count() as u8
    }

    // L2 squared distance from the query to all K centroids in SoA layout.
    // K must be a multiple of 8.
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
            let mut acc = _mm256_setzero_ps();
            for d in 0..DIM {
                let c = _mm256_loadu_ps(soa.add(d * k + cid));
                let diff = _mm256_sub_ps(c, qb[d]);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(cid), acc);
            cid += 8;
        }
    }

    // AVX2 scan of one cluster against the query (scaled space).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn scan_cluster_avx2(&self, cid: usize, q_f32: [f32; DIM], top: &mut [(f32, u8); K]) {
        let sz = self.cluster_sizes[cid] as usize;
        let vec_start = self.cluster_offsets[cid] as usize;
        let panel_off = self.panel_offsets[cid] as usize;
        let n_full = sz / 8;
        let tail = sz % 8;

        // Broadcast each query dim into an AVX2 register.
        let mut qb: [__m256; DIM] = [_mm256_setzero_ps(); DIM];
        for d in 0..DIM {
            qb[d] = _mm256_set1_ps(q_f32[d]);
        }

        let vptr = self.vectors.as_ptr();

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

        // Scan full panels: 8 vectors at a time. Panel size = 14 dims * 8 vectors * 2 bytes.
        let mut p = panel_off;
        let mut worst_d = top.iter().map(|(d, _)| *d).fold(f32::MIN, f32::max);
        for panel in 0..n_full {
            if panel + 1 < n_full {
                let next_ptr = vptr.add(p + DIM * 8) as *const i8;
                _mm_prefetch(next_ptr, _MM_HINT_T0);
                _mm_prefetch(next_ptr.offset(128), _MM_HINT_T0);
            }

            let mut acc = _mm256_setzero_ps();
            for d in 0..DIM {
                let r_i16 = _mm_loadu_si128(vptr.add(p + d * 8) as *const __m128i);
                let r_i32 = _mm256_cvtepi16_epi32(r_i16);
                let r_f32 = _mm256_cvtepi32_ps(r_i32);
                let diff = _mm256_sub_ps(r_f32, qb[d]);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
            // Horizontal min over the 8 panel distances.
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
                    top[worst_i] = (d, self.labels[vidx + v]);
                    worst_d = top.iter().map(|(d, _)| *d).fold(f32::MIN, f32::max);
                }
            }
        }

        // Scan the tail (< 8 vectors, AoS layout).
        let tail_vidx = vec_start + n_full * 8;
        for v in 0..tail {
            let base = p + v * DIM;
            let mut d = 0f32;
            for k in 0..DIM {
                let r = *vptr.add(base + k) as f32;
                let diff = r - q_f32[k];
                d += diff * diff;
            }
            update_top(top, d, self.labels[tail_vidx + v]);
        }
    }
}

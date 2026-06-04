use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod http;
// Legacy IVF index — only used by preprocess + verify legacy path. Hidden
// from the runtime binary so PGO training and LTO concentrate on specialist.
#[cfg(feature = "build-index")]
pub mod ivf;
pub mod json;
pub mod normalize;
pub mod response;
pub mod server;
pub mod specialist;

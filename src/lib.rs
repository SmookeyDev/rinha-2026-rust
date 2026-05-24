use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod http;
pub mod ivf;
pub mod json;
pub mod normalize;
pub mod response;
pub mod server;

FROM rust:1.92-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
COPY specialist.bin /data/specialist.bin

# Single non-PGO build. bmtec (top-1) ships without PGO. Our PGO training
# was driven by verify, which after tier-1 fast-path bypasses ~80% of the
# kNN code path, leaving the kNN cold-profiled and possibly de-optimized.
# Dropping PGO gives equal optimization to both paths.
RUN RUSTFLAGS="-C target-cpu=haswell" \
    cargo build --release --bin rinha2026 --bin lb

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/rinha2026 /usr/local/bin/rinha2026
COPY --from=builder /build/target/release/lb /usr/local/bin/lb
COPY specialist.bin /data/specialist.bin
ENV RINHA_INDEX_PATH=/data/specialist.bin
ENTRYPOINT ["/usr/local/bin/rinha2026"]

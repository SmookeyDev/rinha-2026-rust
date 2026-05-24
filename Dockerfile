FROM rust:1.92-bookworm AS builder
# Use the rustup-bundled llvm-profdata so its instrumentation format matches
# the rustc that produced the .profraw files.
RUN rustup component add llvm-tools-preview \
    && ln -s "$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata" /usr/local/bin/llvm-profdata
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
COPY ivf_int16.bin /data/ivf_int16.bin
COPY test-data.json /data/test-data.json

# 1) Instrumented build that records edge counters into /tmp/pgo.
ENV PGO_DIR=/tmp/pgo
RUN RUSTFLAGS="-C target-cpu=haswell -C profile-generate=${PGO_DIR}" \
    cargo build --release --bin rinha2026 --bin lb --bin verify

# 2) Training run: 54100 representative queries cover the hot path well.
RUN /build/target/release/verify /data/ivf_int16.bin /data/test-data.json 192 \
    && ls -la ${PGO_DIR} | head

# 3) Merge .profraw files into the format rustc consumes.
RUN llvm-profdata merge -o /tmp/merged.profdata ${PGO_DIR}

# 4) Optimised rebuild guided by the merged profile.
RUN rm -rf /build/target/release
RUN RUSTFLAGS="-C target-cpu=haswell -C profile-use=/tmp/merged.profdata" \
    cargo build --release --bin rinha2026 --bin lb

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/rinha2026 /usr/local/bin/rinha2026
COPY --from=builder /build/target/release/lb /usr/local/bin/lb
COPY ivf_int16.bin /data/ivf_int16.bin
ENV RINHA_INDEX_PATH=/data/ivf_int16.bin
ENV RINHA_NPROBE=192
ENTRYPOINT ["/usr/local/bin/rinha2026"]

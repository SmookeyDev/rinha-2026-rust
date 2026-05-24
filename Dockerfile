FROM rust:1.92-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN cargo build --release --bin rinha2026 --bin lb

FROM alpine:3.20
COPY --from=builder /build/target/release/rinha2026 /usr/local/bin/rinha2026
COPY --from=builder /build/target/release/lb /usr/local/bin/lb
COPY ivf_int16.bin /data/ivf_int16.bin
ENV RINHA_INDEX_PATH=/data/ivf_int16.bin
ENV RINHA_NPROBE=192
ENTRYPOINT ["/usr/local/bin/rinha2026"]

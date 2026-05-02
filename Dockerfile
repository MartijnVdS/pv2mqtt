# Build stage - runs natively on the builder's architecture (AMD64)
FROM --platform=$BUILDPLATFORM rust:1-slim-trixie AS builder

WORKDIR /usr/src/pv2mqtt
COPY . .

# Argument automatically provided by Docker Buildx
ARG TARGETPLATFORM

# Install host build tools and configure multi-arch for the target
RUN apt-get update && apt-get install -y pkg-config clang lld && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        dpkg --add-architecture arm64 && \
        apt-get update && \
        apt-get install -y gcc-aarch64-linux-gnu; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Build for the target platform
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/pv2mqtt/target \
    case "$TARGETPLATFORM" in \
        "linux/amd64") \
            cargo build --release && \
            cp target/release/pv2mqtt /usr/src/pv2mqtt/pv2mqtt ;; \
        "linux/arm64") \
            rustup target add aarch64-unknown-linux-gnu && \
            export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
            export PKG_CONFIG_ALLOW_CROSS=1 && \
            export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig && \
            cargo build --release --target aarch64-unknown-linux-gnu && \
            cp target/aarch64-unknown-linux-gnu/release/pv2mqtt /usr/src/pv2mqtt/pv2mqtt ;; \
    esac

# Runtime stage - uses the target architecture
FROM debian:trixie-slim

WORKDIR /app
COPY --from=builder /usr/src/pv2mqtt/pv2mqtt /app/pv2mqtt

# 1. Create a system group and user
# -r: system account
# -g: specify the primary group
# -G dialout: add to dialout group for serial port access
# -s /sbin/nologin: prevents the user from logging in via shell (security)
RUN groupadd -r appgroup && useradd -r -g appgroup -G dialout -s /sbin/nologin appuser

# Install root CA certificates
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

USER appuser

ENTRYPOINT ["/app/pv2mqtt"]
CMD ["/app/pv2mqtt.toml"]

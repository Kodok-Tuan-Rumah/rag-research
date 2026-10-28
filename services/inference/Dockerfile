# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim
ARG RUST_IMAGE=rust:1.99-bookworm
ARG RUNTIME_IMAGE=debian:bookworm-slim

FROM ${PYTHON_IMAGE} AS models
ENV PIP_NO_CACHE_DIR=1
WORKDIR /export
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu" \
    && pip install "transformers==4.57.6" "onnxruntime==1.31.0" "onnx==1.23.2"
COPY scripts/export_inference_models.py /export/scripts/export_inference_models.py
RUN python /export/scripts/export_inference_models.py --output-dir /models

FROM ${RUST_IMAGE} AS builder
WORKDIR /build
COPY services/inference/Cargo.toml services/inference/Cargo.lock* ./
COPY services/inference/src ./src
RUN cargo build --release --locked || cargo build --release

FROM ${RUNTIME_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG ORT_VERSION=1.31.0
COPY services/inference/onnxruntime.sha256 /onnxruntime.sha256
RUN curl -fsSL -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && echo "$(cat /onnxruntime.sha256)  /tmp/ort.tgz" | sha256sum -c - \
    && tar -xzf /tmp/ort.tgz -C /usr/local --strip-components=1 \
    && ldconfig \
    && rm /tmp/ort.tgz
COPY --from=builder /build/target/release/rag-inference /usr/local/bin/rag-inference
COPY --from=models /models /models
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
RUN useradd --system --uid 10001 --create-home inference
USER 10001
EXPOSE 8081
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
  CMD curl -fsS http://127.0.0.1:8081/ready || exit 1
ENTRYPOINT ["/usr/local/bin/rag-inference"]

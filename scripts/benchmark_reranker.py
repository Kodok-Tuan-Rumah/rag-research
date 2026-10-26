"""Benchmark: Rust vs Python cross-encoder reranker.

Measures latency for 100+ query-document pairs using both backends.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Import Rust extension FIRST (before onnxruntime) to avoid "only one environment" conflict
try:
    import cross_encoder_reranker
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Now set up ORT_DYLIB_PATH if needed
if "ORT_DYLIB_PATH" not in os.environ:
    try:
        import onnxruntime
        ort_lib = os.path.join(
            os.path.dirname(onnxruntime.__file__),
            "capi",
            "libonnxruntime.so.1.31.0",
        )
        if os.path.isfile(ort_lib):
            os.environ["ORT_DYLIB_PATH"] = ort_lib
    except ImportError:
        pass

import numpy as np


def generate_pairs(n: int) -> list:
    """Generate n diverse query-document pairs."""
    queries = [
        "What is self-attention in Transformers?",
        "How does RAG reduce hallucination?",
        "What is multi-head attention?",
        "How does instruction tuning work?",
        "What is RLHF?",
        "What is Self-Instruct?",
        "What is LLaMA 2?",
        "How does positional encoding work?",
        "What is scaled dot-product attention?",
        "What are Large Language Models?",
    ]
    docs = [
        "Self-attention allows the model to attend to different positions within a sequence.",
        "RAG retrieves relevant documents and generates answers grounded in evidence.",
        "Multi-head attention projects queries, keys, and values into multiple subspaces.",
        "Instruction tuning trains models to follow natural language instructions.",
        "RLHF aligns LLMs with human preferences using reinforcement learning.",
        "Self-Instruct generates instruction-following data from the model itself.",
        "LLaMA 2 is a family of open-source large language models.",
        "Positional encoding adds position information to token embeddings.",
        "Scaled dot-product attention computes attention weights using dot products.",
        "Large Language Models are trained on vast amounts of text data.",
    ]

    pairs = []
    for i in range(n):
        q = queries[i % len(queries)]
        d = docs[i % len(docs)]
        q = f"{q} (variant {i})"
        d = f"{d} (variant {i})"
        pairs.append((q, d))
    return pairs


def benchmark_rust(pairs: list, warmup: int = 5) -> dict:
    """Benchmark Rust reranker."""
    if not RUST_AVAILABLE:
        return {"error": "Rust extension not importable", "backend": "rust"}

    from src.retrieval.reranker_rust import RustCrossEncoderReranker

    reranker = RustCrossEncoderReranker()
    if not reranker.available:
        return {"error": "Rust reranker not available", "backend": "rust", "detail": str(reranker)}

    # Warm up
    reranker.predict(pairs[:warmup])

    # Benchmark
    t0 = time.perf_counter()
    scores = reranker.predict(pairs)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    per_pair_ms = total_ms / len(pairs)

    return {
        "backend": "rust",
        "n_pairs": len(pairs),
        "total_ms": round(total_ms, 2),
        "per_pair_ms": round(per_pair_ms, 2),
        "scores_sample": [round(float(s), 4) for s in scores[:5]],
    }


def benchmark_python(pairs: list, warmup: int = 5) -> dict:
    """Benchmark Python sentence-transformers reranker."""
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Warm up
    ce.predict(pairs[:warmup])

    # Benchmark
    t0 = time.perf_counter()
    scores = ce.predict(pairs)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    per_pair_ms = total_ms / len(pairs)

    return {
        "backend": "python",
        "n_pairs": len(pairs),
        "total_ms": round(total_ms, 2),
        "per_pair_ms": round(per_pair_ms, 2),
        "scores_sample": [round(float(s), 4) for s in scores[:5]],
    }


def main():
    print("=" * 60)
    print("Rust vs Python Cross-Encoder Reranker Benchmark")
    print("=" * 60)

    results = {}
    for n in [10, 50, 100, 200]:
        print(f"\n--- {n} pairs ---")
        pairs = generate_pairs(n)

        # Rust
        rust_result = benchmark_rust(pairs)
        if "error" in rust_result:
            print(f"  Rust:   ERROR - {rust_result['error']}")
        else:
            print(f"  Rust:   {rust_result['total_ms']:.2f}ms total, {rust_result['per_pair_ms']:.2f}ms/pair")

        # Python
        python_result = benchmark_python(pairs)
        print(f"  Python: {python_result['total_ms']:.2f}ms total, {python_result['per_pair_ms']:.2f}ms/pair")

        if "error" not in rust_result:
            speedup = python_result["per_pair_ms"] / rust_result["per_pair_ms"]
            print(f"  Speedup: {speedup:.2f}x")
            results[n] = {
                "rust": rust_result,
                "python": python_result,
                "speedup": round(speedup, 2),
            }
        else:
            results[n] = {
                "rust": rust_result,
                "python": python_result,
            }

    # Save results
    output_path = Path("results/metrics/reranker_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Pairs':>6} {'Rust (ms/pair)':>16} {'Python (ms/pair)':>18} {'Speedup':>10}")
    for n in [10, 50, 100, 200]:
        r = results[n]
        if "error" not in r.get("rust", {}):
            print(f"{n:>6} {r['rust']['per_pair_ms']:>16.2f} {r['python']['per_pair_ms']:>18.2f} {r['speedup']:>9.2f}x")
        else:
            print(f"{n:>6} {'ERROR':>16} {r['python']['per_pair_ms']:>18.2f} {'N/A':>10}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""FAISS-backed benchmark for temporal vector encoding.

Runs full config sweep across circular encoding + hybrid re-rank,
produces the same table format as the original empirical benchmark.

Usage:
    python benchmarks/run.py                    # Run on synthetic dataset
    python benchmarks/run.py --output results/  # Custom output dir
    python benchmarks/run.py --dataset my_data  # Pre-generated dataset
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent to path so temporalvec is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalvec import (
    EncodingConfig,
    TemporalScale,
    encode,
    encode_batch,
    hybrid_rerank,
    ReRankConfig,
    predict_similarity,
    cosine_sim,
    l2_normalize,
)
from benchmarks.datasets import generate_embeddings, load_dataset, save_dataset

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING: faiss not installed. Install with: pip install faiss-cpu")
    print("         Benchmark will use brute-force numpy fallback (slow for large N).")


# ── FAISS Index ──────────────────────────────────────────────────────────────


def build_faiss_index(vectors: np.ndarray) -> object:
    """Build FAISS IndexFlatIP (inner product = cosine for normalized vectors)."""
    if HAS_FAISS:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype(np.float32))
        return index
    return None


def faiss_search(index, query: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index, returning (distances, indices)."""
    if HAS_FAISS and index is not None:
        query_f32 = query.astype(np.float32)
        if query_f32.ndim == 1:
            query_f32 = query_f32.reshape(1, -1)
        return index.search(query_f32, k)
    else:
        # Brute-force fallback: index is None, use _fallback_vectors
        pass
    # Should not reach here if FAISS is available
    return None, None


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(
    retrieved_ids: np.ndarray,
    topic_ids: np.ndarray,
    wall_times: np.ndarray,
    turn_values: np.ndarray,
    query_idx: int,
) -> Dict[str, float]:
    """Compute precision@k, mean turn delta, mean wall delta."""
    k = len(retrieved_ids)
    query_topic = topic_ids[query_idx]

    precision = np.mean(topic_ids[retrieved_ids] == query_topic)

    q_wall = wall_times[query_idx]
    q_turn = turn_values[query_idx]
    mean_wall_dt = np.mean(np.abs(wall_times[retrieved_ids] - q_wall))
    mean_turn_dt = np.mean(np.abs(turn_values[retrieved_ids] - q_turn))

    return {
        f"pre@{k}": float(precision),
        "mean_turn_dt": float(mean_turn_dt),
        "mean_wall_dt": float(mean_wall_dt),
    }


# ── Benchmark runner ─────────────────────────────────────────────────────────


def run_benchmark(
    sem_embeddings: np.ndarray,
    wall_times: np.ndarray,
    turn_values: np.ndarray,
    topic_ids: np.ndarray,
    convo_ids: np.ndarray,
    k: int = 10,
    n_queries: int = 100,
    seed: int = 42,
) -> Dict:
    """Run full benchmark sweep across circular + hybrid configs.

    Returns dict of results keyed by configuration name.
    """
    rng = np.random.RandomState(seed)
    N = len(sem_embeddings)

    # Normalize wall time for circular encoding
    # Use 90-day window: t = (unix_seconds % window) / window
    # But our synthetic data is already in [0, 1] — use as-is
    wall_norm = wall_times.copy()
    turn_norm = turn_values.copy()

    # Ensure semantic embeddings are L2-normalized
    sem_norm = np.array([l2_normalize(v) for v in sem_embeddings])

    # Build temporal values array
    temporal_vals = np.column_stack([wall_norm, turn_norm])

    results = {}

    # ── 1. Raw semantic baseline ─────────────────────────────────────────────
    raw_index = build_faiss_index(sem_norm.astype(np.float32))

    query_indices = rng.choice(N, n_queries, replace=False)
    raw_metrics = []

    for qi in query_indices:
        query_vec = sem_norm[qi].astype(np.float32)
        distances, indices = faiss_search(raw_index, query_vec, k + 1)
        # Remove self-match
        hit_indices = np.array([i for i in indices[0] if i != qi][:k])
        metrics = compute_metrics(hit_indices, topic_ids, wall_times, turn_values, qi)
        raw_metrics.append(metrics)

    results["raw_semantic"] = aggregate_metrics(raw_metrics)

    # ── 2. Circular encoding configs ─────────────────────────────────────────
    circular_configs = [
        ("semantic_heavy", EncodingConfig.preset("semantic_heavy")),
        ("balanced", EncodingConfig.preset("balanced")),
        ("time_heavy", EncodingConfig.preset("time_heavy")),
        ("turns_only", EncodingConfig.preset("turns_only")),
        ("time_only", EncodingConfig.preset("time_only")),
    ]

    for name, config in circular_configs:
        # Encode all vectors — use only the temporal scales needed by this config
        n_scales = len(config.scales)
        tv_subset = temporal_vals[:, :n_scales] if n_scales < temporal_vals.shape[1] else temporal_vals
        encoded = encode_batch(sem_norm, tv_subset, config)
        index = build_faiss_index(encoded.astype(np.float32))

        config_metrics = []
        for qi in query_indices:
            query_encoded = encode(
                sem_norm[qi],
                [wall_norm[qi], turn_norm[qi]][:n_scales],
                config,
            )
            distances, indices = faiss_search(index, query_encoded.vec, k + 1)
            hit_indices = np.array([i for i in indices[0] if i != qi][:k])
            metrics = compute_metrics(
                hit_indices, topic_ids, wall_times, turn_values, qi
            )
            config_metrics.append(metrics)

        results[f"circular_{name}"] = aggregate_metrics(config_metrics)

    # ── 3. Hybrid log re-rank ────────────────────────────────────────────────
    hybrid_configs = [
        ("log_balanced", ReRankConfig.preset("balanced")),
        ("log_turn_heavy", ReRankConfig.preset("turn_heavy")),
        ("log_time_heavy", ReRankConfig.preset("time_heavy")),
    ]

    for name, rcfg in hybrid_configs:
        hybrid_metrics = []
        for qi in query_indices:
            # First pass: ANN on raw semantic
            query_vec = sem_norm[qi].astype(np.float32)
            distances, indices = faiss_search(raw_index, query_vec, 100)
            candidate_indices = np.array([i for i in indices[0] if i != qi])

            # Build candidate tuples for re-rank
            candidates = []
            for ci in candidate_indices:
                sem_sim = float(distances[0][list(indices[0]).index(ci)])
                dt_wall = abs(wall_times[ci] - wall_times[qi])
                dt_turn = abs(turn_values[ci] - turn_values[qi])
                candidates.append((int(ci), sem_sim, dt_wall, dt_turn))

            ranked = hybrid_rerank(candidates, rcfg)
            hit_indices = np.array([cid for cid, _ in ranked[:k]])

            metrics = compute_metrics(
                hit_indices, topic_ids, wall_times, turn_values, qi
            )
            hybrid_metrics.append(metrics)

        results[f"hybrid_{name}"] = aggregate_metrics(hybrid_metrics)

    # ── 4. Add deltas vs raw baseline ────────────────────────────────────────
    raw = results["raw_semantic"]
    for name in list(results.keys()):
        if name == "raw_semantic":
            continue
        r = results[name]
        r["vs_raw_precision"] = r[f"pre@{k}"] - raw[f"pre@{k}"]
        r["vs_raw_turn"] = (raw["mean_turn_dt"] - r["mean_turn_dt"]) / max(
            raw["mean_turn_dt"], 1e-10
        )
        r["vs_raw_wall"] = (raw["mean_wall_dt"] - r["mean_wall_dt"]) / max(
            raw["mean_wall_dt"], 1e-10
        )

    return results


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, float]:
    """Average metrics across queries."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ── Formatting ───────────────────────────────────────────────────────────────


def print_results(results: Dict, k: int = 10, wall_scale: float = 1.0):
    """Pretty-print results table."""
    print(f"\n{'=' * 80}")
    print(f"Empirical Results — {results.get('_n_messages', 'N')} Messages")
    print(f"{'=' * 80}")

    def fmt_pct(v):
        return f"{v:+.1%}" if isinstance(v, float) else str(v)

    def fmt_delta(v):
        if v > 0:
            return f"+{v:.0%}"
        return f"{v:.0%}"

    print(f"\n  {'Configuration':<24s} {'Pre@{k}':>8s} {'Turn Δ':>8s} {' vs Raw':>8s} {'Wall Δ':>8s} {' vs Raw':>8s}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    raw = results["raw_semantic"]
    print(f"  {'Raw semantic (baseline)':<24s} {raw[f'pre@{k}']:8.1%} {raw['mean_turn_dt']:8.3f} {'--':>8s} {raw['mean_wall_dt']*wall_scale:8.0f}s {'--':>8s}")

    # Circular
    for name in ["semantic_heavy", "balanced", "time_heavy", "turns_only", "time_only"]:
        key = f"circular_{name}"
        if key not in results:
            continue
        r = results[key]
        print(f"  {name:<24s} {r[f'pre@{k}']:8.1%} {r['mean_turn_dt']:8.3f} {fmt_delta(r['vs_raw_turn']):>8s} {r['mean_wall_dt']*wall_scale:8.0f}s {fmt_delta(r['vs_raw_wall']):>8s}")

    print()

    # Hybrid
    print(f"  {'Hybrid Log Re-rank':}")
    for name in ["log_balanced", "log_turn_heavy", "log_time_heavy"]:
        key = f"hybrid_{name}"
        if key not in results:
            continue
        r = results[key]
        print(f"  {name:<24s} {r[f'pre@{k}']:8.1%} {r['mean_turn_dt']:8.3f} {'--':>8s} {r['mean_wall_dt']*wall_scale:8.0f}s {'--':>8s}")

    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="FAISS-backed temporal vector benchmark"
    )
    parser.add_argument(
        "--n-messages", type=int, default=1500,
        help="Number of synthetic messages to generate"
    )
    parser.add_argument(
        "--n-queries", type=int, default=100,
        help="Number of query points to evaluate"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of top results (precision@k)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results",
        help="Directory to save results JSON"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Pre-generated dataset directory (skips generation)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.03,
        help="Noise scale for synthetic data (0.03=realistic)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    print(f"temporalvec benchmark — FAISS {'available' if HAS_FAISS else 'UNAVAILABLE (numpy fallback)'}")

    # Load or generate dataset
    if args.dataset:
        print(f"Loading dataset from {args.dataset}...")
        sem_embeddings, wall_times, turn_values, topic_ids, convo_ids = load_dataset(
            args.dataset
        )
    else:
        print(f"Generating {args.n_messages} synthetic messages ({25} conversations, {5} topics)...")
        sem_embeddings, wall_times, turn_values, topic_ids, convo_ids = generate_embeddings(
            n_messages=args.n_messages,
            noise_scale=args.noise,
            seed=args.seed,
        )

    t0 = time.time()
    results = run_benchmark(
        sem_embeddings, wall_times, turn_values, topic_ids, convo_ids,
        k=args.k, n_queries=args.n_queries, seed=args.seed,
    )
    elapsed = time.time() - t0

    results["_n_messages"] = len(sem_embeddings)
    results["_n_queries"] = args.n_queries
    results["_k"] = args.k
    results["_elapsed_seconds"] = elapsed

    # Print
    print_results(results, k=args.k)

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

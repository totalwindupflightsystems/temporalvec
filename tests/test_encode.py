"""Exact decomposition and encoding verification tests.

Run: python -m pytest tests/test_encode.py -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')
from temporalvec import (
    EncodingConfig,
    TemporalScale,
    encode,
    encode_batch,
    predict_similarity,
    cosine_sim,
    l2_normalize,
)

DIM = 768
SEED = 42

np.random.seed(SEED)


def make_random_embedding(dim: int = DIM) -> np.ndarray:
    return l2_normalize(np.random.randn(dim))


# ── Encoding correctness ─────────────────────────────────────────────────────

def test_single_encode_shape():
    """Encoded vector has semantic dims + 2 per temporal scale."""
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    emb = encode(sem, [0.3, 0.7], config)
    assert emb.vec.shape == (DIM + config.encoded_dim,)
    assert emb.sem_vec.shape == (DIM,)
    assert len(emb.temporal_values) == 2


def test_batch_encode_shape():
    """Batch encode produces correct output shape."""
    config = EncodingConfig.preset("balanced")
    N = 100
    sem_batch = np.array([make_random_embedding() for _ in range(N)])
    tv_batch = np.random.uniform(0, 1, (N, 2))
    result = encode_batch(sem_batch, tv_batch, config)
    assert result.shape == (N, DIM + config.encoded_dim)


def test_encode_preserves_semantic_l2_norm():
    """Encoded vectors include the L2-normalized semantic component."""
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    emb = encode(sem, [0.5, 0.5], config)
    # First DIM entries should be alpha * normalized_semantic
    expected = config.alpha * sem
    np.testing.assert_array_almost_equal(emb.vec[:DIM], expected)


def test_encode_temporal_dimensions():
    """Temporal dimensions encode [sin, cos] with correct phase."""
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    t = 0.25
    emb = encode(sem, [t, 0.5], config)

    # First scale (wall_time, λ=1.0)
    scale = config.scales[0]
    idx = DIM
    expected_sin = scale.beta * np.sin(scale.lam * np.pi * t)
    expected_cos = scale.beta * np.cos(scale.lam * np.pi * t)
    np.testing.assert_almost_equal(emb.vec[idx], expected_sin)
    np.testing.assert_almost_equal(emb.vec[idx + 1], expected_cos)


def test_batch_matches_single():
    """Batch encode matches single encode for each item."""
    config = EncodingConfig.preset("balanced")
    N = 50
    sem_batch = np.array([make_random_embedding() for _ in range(N)])
    tv_batch = np.random.uniform(0, 1, (N, 2))

    batch_result = encode_batch(sem_batch, tv_batch, config)

    for i in range(N):
        single = encode(sem_batch[i], list(tv_batch[i]), config)
        np.testing.assert_array_almost_equal(batch_result[i], single.vec)


# ── Exact decomposition ──────────────────────────────────────────────────────

def test_decomposition_exact_single_pair():
    """Decomposition is exact (within fp) for a single known pair."""
    config = EncodingConfig.preset("balanced")
    sem1 = make_random_embedding()
    sem2 = make_random_embedding()

    t1 = [0.2, 0.3]
    t2 = [0.8, 0.9]

    emb1 = encode(sem1, t1, config)
    emb2 = encode(sem2, t2, config)

    actual = emb1.spatiotemporal_similarity(emb2)
    sem_sim = emb1.semantic_similarity(emb2)
    dts = [abs(t1[i] - t2[i]) for i in range(len(t1))]
    predicted = predict_similarity(sem_sim, dts, config)

    np.testing.assert_almost_equal(actual, predicted, decimal=14)


def test_decomposition_exact_1000_pairs():
    """Decomposition is exact across 1000 random pairs."""
    config = EncodingConfig.preset("balanced")
    N = 100
    sem_batch = np.array([make_random_embedding() for _ in range(N)])
    tv_batch = np.random.uniform(0, 1, (N, 2))
    embs = [encode(sem_batch[i], list(tv_batch[i]), config) for i in range(N)]

    errors = []
    for _ in range(1000):
        i, j = np.random.choice(N, 2, replace=False)
        actual = embs[i].spatiotemporal_similarity(embs[j])
        dts = [abs(embs[i].temporal_values[k] - embs[j].temporal_values[k]) for k in range(2)]
        predicted = predict_similarity(embs[i].semantic_similarity(embs[j]), dts, config)
        errors.append(abs(actual - predicted))

    errors = np.array(errors)
    assert errors.mean() < 1e-14
    assert errors.max() < 1e-13


def test_decomposition_config_sweep():
    """Decomposition holds for every preset config."""
    for preset_name in ["semantic_heavy", "balanced", "time_heavy", "turns_only", "time_only"]:
        config = EncodingConfig.preset(preset_name)
        sem1 = make_random_embedding()
        sem2 = make_random_embedding()
        t1 = [np.random.uniform(0, 1) for _ in config.scales]
        t2 = [np.random.uniform(0, 1) for _ in config.scales]

        emb1 = encode(sem1, t1, config)
        emb2 = encode(sem2, t2, config)

        actual = emb1.spatiotemporal_similarity(emb2)
        sem_sim = emb1.semantic_similarity(emb2)
        dts = [abs(t1[i] - t2[i]) for i in range(len(t1))]
        predicted = predict_similarity(sem_sim, dts, config)

        assert abs(actual - predicted) < 1e-14, f"Failed for preset '{preset_name}'"


# ── Gamma properties ─────────────────────────────────────────────────────────

def test_gammas_sum_to_one():
    """γ_sem + Σγᵢ = 1 always."""
    for preset_name in ["semantic_heavy", "balanced", "time_heavy", "turns_only", "time_only"]:
        config = EncodingConfig.preset(preset_name)
        total = config.gamma_sem + sum(config.gammas)
        np.testing.assert_almost_equal(total, 1.0, decimal=14)


def test_gamma_sem_range():
    """γ_sem ∈ [0, 1] for all presets."""
    for preset_name in ["semantic_heavy", "balanced", "time_heavy", "turns_only", "time_only"]:
        config = EncodingConfig.preset(preset_name)
        assert 0 <= config.gamma_sem <= 1, f"γ_sem out of range for '{preset_name}'"


def test_gamma_sem_decreases_with_more_temporal():
    """γ_sem (semantic weight) drops when we add temporal scales."""
    config_single = EncodingConfig(alpha=1.0, scales=[TemporalScale("t", 0.5, 1.0)])
    config_double = EncodingConfig(alpha=1.0, scales=[
        TemporalScale("t1", 0.5, 1.0), TemporalScale("t2", 0.5, 0.5)])
    assert config_single.gamma_sem > config_double.gamma_sem


# ── Encoding consistency ─────────────────────────────────────────────────────

def test_same_time_identical_semantic_gives_one():
    """Two messages with same semantic + same time → similarity ≈ 1."""
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    emb1 = encode(sem, [0.5, 0.5], config)
    emb2 = encode(sem, [0.5, 0.5], config)
    np.testing.assert_almost_equal(emb1.spatiotemporal_similarity(emb2), 1.0, decimal=14)


def test_distant_time_identical_semantic_penalized():
    """Same semantic at distant times → similarity < 1."""
    config = EncodingConfig.preset("time_heavy")
    sem = make_random_embedding()
    emb1 = encode(sem, [0.0, 0.5], config)
    emb2 = encode(sem, [1.0, 0.5], config)
    sim = emb1.spatiotemporal_similarity(emb2)
    assert sim < 1.0, f"Expected temporal penalty; got {sim:.6f}"
    assert sim >= 0.0


def test_opposite_phase_max_penalty():
    """Messages at opposite temporal phase get maximum penalty."""
    config = EncodingConfig.preset("time_heavy")
    sem = make_random_embedding()
    emb1 = encode(sem, [0.0, 0.5], config)
    emb2 = encode(sem, [0.5, 0.5], config)  # Δt = 0.5 in [0,1], λ=1 → λπΔt = π/2
    sim = emb1.spatiotemporal_similarity(emb2)
    # For time_heavy with λ=1.0: cos(π/2) ≈ 0, so γ_sem contributes but temporal doesn't
    assert sim < 0.85, f"Expected substantial penalty; got {sim:.6f}"


# ── TemporalScale ────────────────────────────────────────────────────────────

def test_temporal_scale_encode():
    scale = TemporalScale("test", beta=0.5, lam=1.0)
    result = scale.encode(0.25)  # θ = π/4
    expected_sin = 0.5 * np.sin(np.pi / 4)
    expected_cos = 0.5 * np.cos(np.pi / 4)
    np.testing.assert_array_almost_equal(result, np.array([expected_sin, expected_cos]))


def test_temporal_scale_proximity():
    scale = TemporalScale("test", beta=0.5, lam=1.0)
    assert scale.proximity(0.0, 0.0) == 1.0
    assert abs(scale.proximity(0.0, 1.0) - (-1.0)) < 1e-14  # cos(π) = -1
    assert abs(scale.proximity(0.0, 0.5)) < 1e-14  # cos(π/2) ≈ 0


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_mismatched_temporal_values_raises():
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    with pytest.raises(ValueError):
        encode(sem, [0.5], config)  # balanced needs 2 values


def test_batch_mismatched_temporal_values_raises():
    config = EncodingConfig.preset("balanced")
    sem_batch = np.array([make_random_embedding() for _ in range(10)])
    tv_batch = np.random.uniform(0, 1, (10, 1))  # Only 1 scale
    with pytest.raises(ValueError):
        encode_batch(sem_batch, tv_batch, config)


def test_predict_similarity_mismatched_dt_raises():
    config = EncodingConfig.preset("balanced")
    with pytest.raises(ValueError):
        predict_similarity(0.5, [0.3], config)  # balanced needs 2 dt values


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        EncodingConfig.preset("nonexistent")


# ── Norm stability ───────────────────────────────────────────────────────────

def test_encoded_norm_constant():
    """Norm of encoded vector is sqrt(α² + Σβᵢ²) regardless of t values."""
    config = EncodingConfig.preset("balanced")
    sem = make_random_embedding()
    norms = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        emb = encode(sem, [t, t], config)
        norms.append(np.sqrt(np.dot(emb.vec, emb.vec)))

    target = np.sqrt(config.alpha**2 + sum(s.beta**2 for s in config.scales))
    for n in norms:
        np.testing.assert_almost_equal(n, target, decimal=14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

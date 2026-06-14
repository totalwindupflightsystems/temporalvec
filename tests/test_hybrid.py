"""Hybrid log re-rank tests.

Run: python -m pytest tests/test_hybrid.py -v
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from temporalvec import hybrid_score, hybrid_rerank, ReRankConfig, log_decay


def test_log_decay_zero():
    """Zero time difference → zero penalty."""
    assert log_decay(0.0, 0.1) == 0.0


def test_log_decay_monotonic():
    """Penalty increases monotonically with time difference."""
    vals = [log_decay(t, 0.1) for t in [0.1, 1.0, 10.0, 100.0, 10000.0]]
    for i in range(len(vals) - 1):
        assert vals[i + 1] > vals[i], f"Not monotonic at step {i}: {vals}"


def test_log_decay_alpha_scaling():
    """Higher alpha = steeper penalty at same dt."""
    dt = 100.0
    assert log_decay(dt, 0.5) > log_decay(dt, 0.1)


def test_hybrid_score_perfect_match():
    """Zero temporal distance + perfect semantic → high score."""
    score = hybrid_score(1.0, 0.0, 0.0)
    assert score == 1.0


def test_hybrid_score_temporal_penalty():
    """Temporal distance reduces score."""
    score_near = hybrid_score(0.8, 1.0, 1.0)  # 1 second, 1 turn apart
    score_far = hybrid_score(0.8, 1_000_000.0, 100.0)  # 11 days, 100 turns apart
    assert score_far < score_near


def test_hybrid_score_wall_vs_turn():
    """Wall time and turn distance have independent penalties."""
    cfg_equal = ReRankConfig(alpha_wall=0.1, alpha_turn=0.1)
    score_wall = hybrid_score(0.9, 1000.0, 0.0, cfg_equal)
    score_turn = hybrid_score(0.9, 0.0, 1000.0, cfg_equal)
    # Same dt magnitude → same penalty when alphas equal
    np.testing.assert_almost_equal(score_wall, score_turn, decimal=14)


def test_hybrid_rerank_ordering():
    """Hybrid rerank prefers temporally close candidates with similar semantics."""
    candidates = [
        (1, 0.85, 3600.0, 5.0),   # 1 hour, 5 turns — heavily penalized
        (2, 0.85, 10.0, 1.0),     # 10 seconds, 1 turn — lightly penalized
        (3, 0.65, 1.0, 0.0),      # much lower sem, no penalty
    ]
    ranked = hybrid_rerank(candidates)
    # Candidate 2: 0.85 - 0.1·log(11) - 0.2·log(2) = 0.85 - 0.240 - 0.139 = 0.471
    # Candidate 3: 0.65 - 0.1·log(2) - 0       = 0.65 - 0.069           = 0.581
    # Candidate 1: 0.85 - 0.1·log(3601) - 0.2·log(6) = 0.85 - 0.819 - 0.358 = -0.327
    # Candidate 3 wins (barely penalized, sem still decent)
    assert ranked[0][0] == 3, f"Expected candidate 3 first, got {ranked[0][0]}. Scores: {ranked}"


def test_hybrid_rerank_same_sem_different_time():
    """Same semantic sim, different times → closer time ranks higher."""
    candidates = [
        (1, 0.80, 1_000_000.0, 50.0),
        (2, 0.80, 10.0, 1.0),
    ]
    ranked = hybrid_rerank(candidates)
    assert ranked[0][0] == 2


def test_preset_configs():
    """All preset configs produce valid scores."""
    for preset_name in ["balanced", "turn_heavy", "time_heavy"]:
        config = ReRankConfig.preset(preset_name)
        score = hybrid_score(0.8, 3600.0, 10.0, config)
        assert isinstance(score, float)


def test_unknown_preset_raises():
    try:
        ReRankConfig.preset("nonexistent")
        assert False, "Should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

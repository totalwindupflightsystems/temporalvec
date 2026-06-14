"""Hybrid log re-rank — production-quality temporal scoring.

Unlike pure circular encoding (γ baked at insert, cos echoes), hybrid mode
stores raw embeddings + timestamps and applies log-decay scoring at query time.

Score:
    score = cosine_sim(v_query, v_candidate)
          - α_wall · log(1 + |Δt_wall|)
          - α_turn · log(1 + |Δt_turn|)

Properties:
    - Monotonic — never echoes (vs cos which cycles every 2/λ)
    - Per-query tuning — change α weights without re-encoding
    - Raw timestamps — no normalization required
    - Temporal-only queries work (set semantic_sim=0)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ReRankConfig:
    """Configuration for hybrid log re-rank scoring.

    Attributes:
        alpha_wall: Weight for wall-time decay. Higher = more recency bias.
        alpha_turn: Weight for turn-distance decay. Higher = more conversational proximity.
    """
    alpha_wall: float = 0.1
    alpha_turn: float = 0.2

    @classmethod
    def preset(cls, name: str) -> 'ReRankConfig':
        """Get a preset re-rank configuration.

        Presets:
            balanced         — gentle time + turn decay
            turn_heavy       — conversational adjacency matters most
            time_heavy       — absolute recency matters most
        """
        presets = {
            "balanced": cls(alpha_wall=0.1, alpha_turn=0.2),
            "turn_heavy": cls(alpha_wall=0.05, alpha_turn=0.5),
            "time_heavy": cls(alpha_wall=0.3, alpha_turn=0.05),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return presets[name]


def log_decay(dt: float, alpha: float) -> float:
    """Log-decay penalty: α · log(1 + |dt|).

    Natural compression of time scale:
        |dt|       penalty (α=0.1)
        1s         0.069
        1 min      0.41
        1 hour     0.82
        1 day      1.14
        1 week     1.33
        1 month    1.48
        1 year     1.73

    Args:
        dt: Absolute time difference (seconds for wall time, turn count for turns).
        alpha: Decay weight. Higher = steeper penalty.

    Returns:
        Decay penalty to subtract from similarity score.
    """
    return alpha * np.log(1.0 + abs(dt))


def hybrid_score(
    sem_sim: float,
    dt_wall: float,
    dt_turn: float,
    config: Optional[ReRankConfig] = None,
) -> float:
    """Compute hybrid score for a single candidate.

    Args:
        sem_sim: Cosine similarity between query and candidate raw embeddings.
        dt_wall: Absolute wall-time difference (seconds).
        dt_turn: Absolute turn-position difference.
        config: Re-rank configuration. Uses balanced preset if None.

    Returns:
        Score = sem_sim - α_wall·log(1+|Δt_wall|) - α_turn·log(1+|Δt_turn|).
    """
    if config is None:
        config = ReRankConfig.preset("balanced")

    score = sem_sim
    score -= log_decay(dt_wall, config.alpha_wall)
    score -= log_decay(dt_turn, config.alpha_turn)
    return float(score)


def hybrid_rerank(
    candidates: List[Tuple[int, float, float, float]],
    config: Optional[ReRankConfig] = None,
) -> List[Tuple[int, float]]:
    """Re-rank a candidate list using hybrid log-decay scoring.

    Args:
        candidates: List of (candidate_id, sem_sim, dt_wall, dt_turn) tuples.
        config: Re-rank configuration. Uses balanced preset if None.

    Returns:
        Sorted list of (candidate_id, hybrid_score) tuples, highest score first.
    """
    if config is None:
        config = ReRankConfig.preset("balanced")

    scored = [
        (cid, hybrid_score(sem_sim, dt_wall, dt_turn, config))
        for cid, sem_sim, dt_wall, dt_turn in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

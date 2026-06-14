"""Circular temporal encoding — core math.

Add temporal dimensions to any L2-normalized embedding by appending
sin/cos pairs. Each temporal scale adds 2 dimensions.

Encoding:
    q = [α·v̂, β₁·sin(λ₁π·t₁), β₁·cos(λ₁π·t₁), β₂·sin(λ₂π·t₂), β₂·cos(λ₂π·t₂), ...]

Exact decomposition (proven, zero approximation error):
    cos_sim(q₁,q₂) = γ·sem_sim + Σᵢ γᵢ·cos(λᵢπ·|tᵢ₁ - tᵢ₂|)

    where γ = α²/(α² + Σβᵢ²), γᵢ = βᵢ²/(α² + Σβᵢ²), γ + Σγᵢ = 1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── Math utilities ───────────────────────────────────────────────────────────

def l2_norm(v: np.ndarray) -> float:
    """L2 norm of a vector."""
    return float(np.sqrt(np.dot(v, v)))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns unchanged if zero-norm."""
    n = l2_norm(v)
    return v / n if n > 0 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = l2_norm(a), l2_norm(b)
    return float(np.dot(a, b) / (na * nb)) if na * nb > 0 else 0.0


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TemporalScale:
    """One dimension of temporal encoding.

    Each scale adds 2 dimensions [sin(λπ·t), cos(λπ·t)] multiplied by β.

    Attributes:
        name: Human-readable label (e.g. "wall_time", "turn_position").
        beta: Weight in similarity computation. Higher = more temporal influence.
        lam: Temporal frequency. Higher = finer resolution, but cos echoes at Δt=2/λ.
             Keep λ ≤ 2 within [0,1] normalized time to avoid echoes.
    """
    name: str
    beta: float
    lam: float

    def encode(self, t: float) -> np.ndarray:
        """Return [β·sin(λπ·t), β·cos(λπ·t)] for this scale."""
        theta = self.lam * np.pi * t
        return self.beta * np.array([np.sin(theta), np.cos(theta)])

    def proximity(self, t1: float, t2: float) -> float:
        """Cosine proximity for this scale: cos(λπ·|t1 - t2|)."""
        return float(np.cos(self.lam * np.pi * (t1 - t2)))


@dataclass
class EncodingConfig:
    """Full encoding configuration.

    Attributes:
        alpha: Semantic weight (default 1.0). Higher = semantic dominates.
        scales: List of temporal scales to encode.

    The fraction of similarity from semantics is:
        γ_sem = α² / (α² + Σβᵢ²)
    """
    alpha: float = 1.0
    scales: List[TemporalScale] = field(default_factory=list)

    @property
    def gamma_sem(self) -> float:
        """Fraction of similarity from semantics."""
        alpha_sq = self.alpha ** 2
        beta_sqs = sum(s.beta ** 2 for s in self.scales)
        total = alpha_sq + beta_sqs
        return alpha_sq / total if total > 0 else 1.0

    @property
    def gammas(self) -> List[float]:
        """Per-scale γᵢ = βᵢ² / (α² + Σβⱼ²)."""
        alpha_sq = self.alpha ** 2
        beta_sqs = [s.beta ** 2 for s in self.scales]
        total = alpha_sq + sum(beta_sqs)
        if total == 0:
            return [0.0] * len(self.scales)
        return [b2 / total for b2 in beta_sqs]

    @property
    def encoded_dim(self) -> int:
        """Number of temporal dimensions added (2 per scale)."""
        return 2 * len(self.scales)

    @classmethod
    def preset(cls, name: str) -> 'EncodingConfig':
        """Get a preset configuration.

        Presets:
            semantic_heavy  — γ≈0.85  (semantic dominates, gentle time nudging)
            balanced        — γ≈0.71  (equal attention to time and meaning)
            time_heavy      — γ≈0.50  (time and turn proximity matter a lot)
            turns_only      — γ≈0.22  (pure conversational indexing via turns)
            time_only       — γ≈0.22  (pure temporal indexing via wall time)
        """
        presets = {
            "semantic_heavy": cls(alpha=1.0, scales=[
                TemporalScale("wall_time", 0.3, 1.0),
                TemporalScale("turn_position", 0.3, 0.5),
            ]),
            "balanced": cls(alpha=1.0, scales=[
                TemporalScale("wall_time", 0.4, 1.0),
                TemporalScale("turn_position", 0.5, 0.5),
            ]),
            "time_heavy": cls(alpha=1.0, scales=[
                TemporalScale("wall_time", 0.5, 1.0),
                TemporalScale("turn_position", 0.6, 0.5),
            ]),
            "turns_only": cls(alpha=1.0, scales=[
                TemporalScale("turn_position", 0.9, 0.5),
            ]),
            "time_only": cls(alpha=1.0, scales=[
                TemporalScale("wall_time", 0.9, 1.0),
            ]),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
        return presets[name]


# ── Encoding ─────────────────────────────────────────────────────────────────

@dataclass
class SpatiotemporalEmbedding:
    """An embedding with temporal scales baked in.

    Attributes:
        vec: Full spatiotemporal vector (d_sem + 2·n_scales dimensions).
        sem_vec: Semantic-only vector (L2-normalized, d_sem dimensions).
        temporal_values: Raw normalized t-values per scale, in scale order.
    """
    vec: np.ndarray
    sem_vec: np.ndarray
    temporal_values: List[float]

    @classmethod
    def encode(
        cls,
        sem_embedding: np.ndarray,
        temporal_values: List[float],
        config: Optional[EncodingConfig] = None,
    ) -> 'SpatiotemporalEmbedding':
        """Encode a semantic embedding with temporal scales.

        Args:
            sem_embedding: Raw embedding vector (any dimension).
            temporal_values: One normalized t ∈ [0,1] per scale.
            config: Encoding configuration. Uses balanced preset if None.

        Returns:
            SpatiotemporalEmbedding with vec = [α·v̂ | β₁·sin(λ₁π·t₁) | β₁·cos(λ₁π·t₁) | ...].
        """
        if config is None:
            config = EncodingConfig.preset("balanced")

        if len(temporal_values) != len(config.scales):
            raise ValueError(
                f"Expected {len(config.scales)} temporal values, got {len(temporal_values)}"
            )

        sem_norm = l2_normalize(sem_embedding)
        parts = [config.alpha * sem_norm]

        for scale, t in zip(config.scales, temporal_values):
            parts.append(scale.encode(t))

        return cls(
            vec=np.concatenate(parts),
            sem_vec=sem_norm,
            temporal_values=list(temporal_values),
        )

    def semantic_similarity(self, other: 'SpatiotemporalEmbedding') -> float:
        """Raw semantic similarity (cosine between semantic vectors)."""
        return cosine_sim(self.sem_vec, other.sem_vec)

    def spatiotemporal_similarity(self, other: 'SpatiotemporalEmbedding') -> float:
        """Full spatiotemporal similarity (cosine between encoded vectors)."""
        return cosine_sim(self.vec, other.vec)


def encode(
    sem_embedding: np.ndarray,
    temporal_values: List[float],
    config: Optional[EncodingConfig] = None,
) -> SpatiotemporalEmbedding:
    """Convenience wrapper around SpatiotemporalEmbedding.encode()."""
    return SpatiotemporalEmbedding.encode(sem_embedding, temporal_values, config)


def encode_batch(
    sem_embeddings: np.ndarray,
    temporal_values: np.ndarray,
    config: Optional[EncodingConfig] = None,
) -> np.ndarray:
    """Encode a batch of semantic embeddings with temporal scales.

    Args:
        sem_embeddings: (N, d) array of L2-normalized semantic embeddings.
        temporal_values: (N, s) array of normalized t-values per sample per scale.
        config: Encoding configuration. Uses balanced preset if None.

    Returns:
        (N, d + 2·s) array of spatiotemporal vectors.
    """
    if config is None:
        config = EncodingConfig.preset("balanced")

    N, d = sem_embeddings.shape
    s = temporal_values.shape[1]

    if s != len(config.scales):
        raise ValueError(
            f"temporal_values has {s} scales, config has {len(config.scales)}"
        )

    # Build temporal part for all samples
    temporal_parts = []
    for i, scale in enumerate(config.scales):
        theta = scale.lam * np.pi * temporal_values[:, i]  # (N,)
        temporal_parts.append(scale.beta * np.sin(theta)[:, np.newaxis])  # (N, 1)
        temporal_parts.append(scale.beta * np.cos(theta)[:, np.newaxis])  # (N, 1)

    # Concatenate: [α·sem | temporal dims]
    return np.hstack([config.alpha * sem_embeddings] + temporal_parts)


# ── Similarity prediction (theoretical) ──────────────────────────────────────

def predict_similarity(
    sem_sim: float,
    dt_values: List[float],
    config: Optional[EncodingConfig] = None,
) -> float:
    """Theoretical decomposition: γ·sem_sim + Σ γᵢ·cos(λᵢπ·dtᵢ).

    This is the proven exact form — it MUST match actual cosine similarity
    between encoded vectors (within floating-point error). Any deviation
    indicates a bug in either the encoding or the prediction.

    Args:
        sem_sim: Cosine similarity between the raw semantic embeddings.
        dt_values: Absolute differences |t1 - t2| per temporal scale.
        config: Encoding config used for encoding.

    Returns:
        Predicted cosine similarity between the encoded vectors.
    """
    if config is None:
        config = EncodingConfig.preset("balanced")

    if len(dt_values) != len(config.scales):
        raise ValueError(
            f"Expected {len(config.scales)} dt values, got {len(dt_values)}"
        )

    gamma_sem = config.gamma_sem
    result = gamma_sem * sem_sim

    for scale, dt, gamma_i in zip(config.scales, dt_values, config.gammas):
        result += gamma_i * np.cos(scale.lam * np.pi * dt)

    return float(result)

"""Circular temporal encoding for vector databases.

Add time-aware search without changing your embedding model.
Append 2D sin/cos dimensions per temporal scale to any L2-normalized embedding.

Core encoding:
    q = [α·v̂, β₁·sin(λ₁π·t₁), β₁·cos(λ₁π·t₁), ...]

Exact decomposition (proven, not approximate):
    cos_sim(q₁,q₂) = γ·sem_sim + Σᵢ γᵢ·cos(λᵢπ·|tᵢ₁ - tᵢ₂|)
"""

from .encode import (
    TemporalScale,
    EncodingConfig,
    SpatiotemporalEmbedding,
    encode,
    encode_batch,
    predict_similarity,
    l2_norm,
    l2_normalize,
    cosine_sim,
)

__version__ = "0.1.0"
__all__ = [
    "TemporalScale",
    "EncodingConfig",
    "SpatiotemporalEmbedding",
    "encode",
    "encode_batch",
    "predict_similarity",
    "l2_norm",
    "l2_normalize",
    "cosine_sim",
]

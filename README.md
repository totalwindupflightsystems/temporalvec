# temporalvec

**Circular temporal encoding for vector databases.**

Add time-aware search without changing your embedding model. Append 2D sin/cos dimensions per temporal scale to any L2-normalized embedding, with exact additive decomposition of semantic and temporal similarity.

```python
from temporalvec import encode, EncodingConfig, predict_similarity
import numpy as np

# Balanced time + semantic encoding
config = EncodingConfig.preset("balanced")

# Encode: [768d semantic | 2d wall_time (sin,cos) | 2d turn_position (sin,cos)]
emb = encode(sem_embedding, temporal_values=[0.3, 0.7], config=config)

# The similarity decomposes exactly (proven, not approximate):
predicted = predict_similarity(sem_sim, dt_values=[0.1, 0.2], config=config)
# = γ·sem_sim + γ_wall·cos(λ_wall·π·dt_wall) + γ_turn·cos(λ_turn·π·dt_turn)
```

## Quick Start

```bash
pip install temporalvec

# Run tests
python -m pytest tests/ -v

# Run benchmark (needs FAISS + synthetic data generation)
pip install temporalvec[benchmark]
python benchmarks/run.py
```

## Core Math

The encoding is **exact** — no approximation, no training:

```
q = [α·v̂, β₁·sin(λ₁π·t₁), β₁·cos(λ₁π·t₁), β₂·sin(λ₂π·t₂), β₂·cos(λ₂π·t₂)]

cos_sim(q₁,q₂) = γ·sem_sim + Σᵢ γᵢ·cos(λᵢπ·|tᵢ₁ - tᵢ₂|)

where γ = α²/(α²+Σβᵢ²), γᵢ = βᵢ²/(α²+Σβᵢ²), γ + Σγᵢ = 1
```

Vectors stay L2-normalized because v̂ is unit and sin²+cos²=1.

## Presets

| Preset | γ_sem | Best for |
|--------|-------|----------|
| `semantic_heavy` | 0.85 | Meaning-first, gentle time nudging |
| `balanced` | 0.71 | Equal attention to meaning and time |
| `time_heavy` | 0.50 | Time proximity matters a lot |
| `turns_only` | 0.22 | Pure conversational indexing |
| `time_only` | 0.22 | Pure temporal indexing |

## Two Modes

### Mode A: Circular-only (single-pass ANN)
Store encoded vectors. One ANN pass, done. Time weighted at insert time.

### Mode B: Hybrid (production)
Store raw embeddings + timestamps. ANN → log-decay re-rank. Per-query temporal tuning, no cos echoes.

```python
from temporalvec import hybrid_score

# Score = cosine_sim - α_wall·log(1+|Δt_wall|) - α_turn·log(1+|Δt_turn|)
score = hybrid_score(sem_sim=0.85, dt_wall=3600.0, dt_turn=5.0)
```

## Package Structure

```
temporalvec/
├── temporalvec/
│   ├── __init__.py     # Public API
│   ├── encode.py       # Circular encoding + math
│   └── hybrid.py       # Log-decay re-rank scoring
├── benchmarks/
│   ├── run.py          # FAISS-backed benchmark runner
│   └── datasets.py     # Synthetic conversation generator
├── tests/
│   ├── test_encode.py  # Exact decomposition verification (21 tests)
│   └── test_hybrid.py  # Re-rank scoring tests (10 tests)
├── paper/
│   └── figures/        # Whitepaper figures (TBD)
├── pyproject.toml
└── README.md
```

## Status

- ✅ Math proven (exact decomposition, zero approximation error)
- ✅ 31 tests pass (zero-dependency deterministic verification)
- ✅ FAISS benchmark harness
- ✅ Configurable presets + multi-scale encoding
- ⬜ Whitepaper (LaTeX)
- ⬜ Real-dataset benchmarks
- ⬜ Go implementation for DexDat

## Citation

```bibtex
@software{temporalvec2026,
  author = {Nous Research},
  title = {temporalvec: Circular Temporal Encoding for Vector Databases},
  year = {2026},
  url = {https://github.com/totalwindupflightsystems/temporalvec}
}
```

## License

MIT

"""Benchmark datasets — synthetic conversation generator.

Generates realistic synthetic conversation data with:
- Topic-aware semantic embeddings (well-separated centroids + noise)
- Wall-time and turn-position temporal values
- Configurable parameters (topics, messages, noise levels)

Used by benchmarks/run.py to produce reproducible empirical results.
"""

import numpy as np
from typing import List, Tuple, Optional
import json
from pathlib import Path


def generate_embeddings(
    n_messages: int = 1500,
    n_topics: int = 5,
    n_conversations: int = 25,
    dim: int = 768,
    noise_scale: float = 0.03,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic conversation data.

    Args:
        n_messages: Total number of messages to generate.
        n_topics: Number of distinct topic clusters.
        n_conversations: Number of simulated conversations.
        dim: Embedding dimensionality.
        noise_scale: Standard deviation of noise added to topic centroids.
            0.03 = realistic same-topic sim ~0.7-0.9. 0.08 = garbage ~0.17.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (semantic_embeddings, wall_time_values, turn_values,
                  topic_ids, conversation_ids)
        - semantic_embeddings: (N, dim) L2-normalized
        - wall_time_values: (N,) normalized wall time [0, 1]
        - turn_values: (N,) normalized turn position [0, 1]
        - topic_ids: (N,) topic index [0, n_topics)
        - conversation_ids: (N,) conversation index [0, n_conversations)
    """
    rng = np.random.RandomState(seed)

    # Generate topic centroids (well-separated)
    centroids = []
    for i in range(n_topics):
        c = rng.randn(dim)
        for prev in centroids:
            c = c - np.dot(c, prev) * prev  # Gram-Schmidt orthogonalization
        n = np.sqrt(np.dot(c, c))
        centroids.append(c / n if n > 0 else c)

    centroids = np.array(centroids)

    msgs_per_convo = n_messages // n_conversations
    remainder = n_messages % n_conversations

    sem_embeddings = np.zeros((n_messages, dim))
    wall_times = np.zeros(n_messages)
    turn_values = np.zeros(n_messages)
    topic_ids = np.zeros(n_messages, dtype=int)
    convo_ids = np.zeros(n_messages, dtype=int)

    idx = 0
    for convo_id in range(n_conversations):
        n_msgs = msgs_per_convo + (1 if convo_id < remainder else 0)
        topic = convo_id % n_topics
        convo_start = rng.uniform(0, 0.7)  # normalized wall time start
        convo_duration = rng.uniform(0.1, 0.3)

        for turn in range(n_msgs):
            # Wall time
            t_wall = convo_start + (turn / max(1, n_msgs - 1)) * convo_duration
            t_wall += rng.uniform(-0.005, 0.005)  # jitter
            t_wall = float(np.clip(t_wall, 0.001, 0.999))

            # Turn position (normalized within conversation)
            t_turn = turn / max(1, n_msgs - 1)

            # Semantic embedding
            sem = centroids[topic] + rng.randn(dim) * noise_scale
            n = np.sqrt(np.dot(sem, sem))
            sem = sem / n if n > 0 else sem

            sem_embeddings[idx] = sem
            wall_times[idx] = t_wall
            turn_values[idx] = t_turn
            topic_ids[idx] = topic
            convo_ids[idx] = convo_id

            idx += 1

    return sem_embeddings, wall_times, turn_values, topic_ids, convo_ids


def save_dataset(
    path: str,
    sem_embeddings: np.ndarray,
    wall_times: np.ndarray,
    turn_values: np.ndarray,
    topic_ids: np.ndarray,
    convo_ids: np.ndarray,
) -> None:
    """Save dataset to disk as .npy files and metadata JSON.

    Args:
        path: Directory to save files in.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "sem_embeddings.npy", sem_embeddings)
    np.save(out_dir / "wall_times.npy", wall_times)
    np.save(out_dir / "turn_values.npy", turn_values)
    np.save(out_dir / "topic_ids.npy", topic_ids)
    np.save(out_dir / "convo_ids.npy", convo_ids)

    metadata = {
        "n_messages": int(len(sem_embeddings)),
        "n_topics": int(topic_ids.max() + 1),
        "n_conversations": int(convo_ids.max() + 1),
        "dim": int(sem_embeddings.shape[1]),
        "conversation_sizes": [
            int((convo_ids == c).sum()) for c in range(convo_ids.max() + 1)
        ],
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {out_dir}/ ({sem_embeddings.shape[0]} messages)")


def load_dataset(path: str) -> Tuple[np.ndarray, ...]:
    """Load a dataset from disk.

    Args:
        path: Directory containing .npy files.

    Returns:
        Same tuple as generate_embeddings.
    """
    p = Path(path)
    return (
        np.load(p / "sem_embeddings.npy"),
        np.load(p / "wall_times.npy"),
        np.load(p / "turn_values.npy"),
        np.load(p / "topic_ids.npy"),
        np.load(p / "convo_ids.npy"),
    )

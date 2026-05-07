# experiments/stats.py

from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np


def _phi_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Phi correlation for binary arrays.
    Safe for constant vectors (returns 0.0).
    """
    x = x.astype(int)
    y = y.astype(int)
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / float(np.sqrt(denom))


def compute_agent_stats(full_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    full_results: output from FeedbackPipeline.run(...), list of dict items containing agent_results.
    Returns aggregate stats + correlation matrices.
    """
    if not full_results:
        return {
            "num_items": 0,
            "agents": [],
            "pass_rate": {},
            "agreement": {},
            "phi_corr": {},
            "accepted_rate": 0.0,
        }

    # Determine agent names
    agent_names = sorted(list(full_results[0]["agent_results"].keys()))
    m = len(full_results)
    n = len(agent_names)

    # Build binary pass matrix shape (m, n)
    passes = np.zeros((m, n), dtype=int)
    accepted = np.zeros((m,), dtype=int)

    for i, item in enumerate(full_results):
        accepted[i] = 1 if item.get("accepted") else 0
        ar = item["agent_results"]
        for j, name in enumerate(agent_names):
            passes[i, j] = 1 if ar[name]["passed"] else 0

    pass_rate = {name: float(passes[:, j].mean()) for j, name in enumerate(agent_names)}

    # Pairwise agreement and phi
    agreement = {name: {} for name in agent_names}
    phi = {name: {} for name in agent_names}

    for i, ni in enumerate(agent_names):
        xi = passes[:, i]
        for j, nj in enumerate(agent_names):
            xj = passes[:, j]
            agreement[ni][nj] = float((xi == xj).mean())
            phi[ni][nj] = float(_phi_corr(xi, xj))

    out = {
        "num_items": m,
        "agents": agent_names,
        "pass_rate": pass_rate,
        "accepted_rate": float(accepted.mean()),
        "agreement": agreement,
        "phi_corr": phi,
    }
    return out

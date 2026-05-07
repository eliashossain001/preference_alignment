# experiments/runner.py

import numpy as np
from feedback_pipeline.pipeline import FeedbackPipeline
from experiments.corruption import apply_corruption

def run_one_setting(
    feedback,
    eta,
    k,
    corruption,
    proxy_p_correct
):
    corrupted = apply_corruption(
        feedback, corruption=corruption, eta=eta
    )

    pipeline = FeedbackPipeline(
        mode="proxy",
        proxy_p_correct=proxy_p_correct
    )

    accepted, full = pipeline.run(corrupted, min_pass=k)

    retained = len(accepted)
    total = len(corrupted)

    # effective eta'
    bad_retained = sum(
        1 for fb in accepted if not fb._is_clean
    )
    eta_prime = bad_retained / max(retained, 1)

    return {
        "eta": eta,
        "k": k,
        "corruption": corruption,
        "retained_pct": retained / total,
        "eta_prime": eta_prime,
        "bad_retained": bad_retained,
        "total_retained": retained
    }

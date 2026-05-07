# experiments/corruption.py

import random
from agents.base import Feedback


def apply_corruption(
    feedback: list[Feedback],
    corruption: str,
    eta: float,
    seed: int = 42
):
    """
    Inject controlled corruption and tag each item
    with is_clean label.
    """
    rng = random.Random(seed)
    corrupted = []

    for fb in feedback:
        fb = fb.copy(deep=True)
        fb.is_clean = True

        if rng.random() < eta:
            fb.is_clean = False

            if corruption == "label_flip":
                fb.user_choice = "B" if fb.user_choice == "A" else "A"

            elif corruption == "poison_overrefusal":
                fb.response_a = "I cannot help with that."
                fb.response_b = "I cannot help with that."
                fb.user_choice = "A"

        corrupted.append(fb)

    return corrupted
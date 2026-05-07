# agents/proxy_verifiers.py

import random
from agents.base import BaseAgent, Feedback, AgentResult


class ProxyVerifier(BaseAgent):
    """
    Stochastic proxy verifier with controllable accuracy.
    Used to simulate factual/safety/ethics/trust checks
    under controlled noise.
    """

    def __init__(self, name: str, p_correct: float, seed: int = 42):
        """
        :param name: verifier name (Knowledge, Safety, etc.)
        :param p_correct: probability of correct decision
        """
        self.name = name
        self.p_correct = p_correct
        self.rng = random.Random(seed)

    def verify(self, fb: Feedback) -> AgentResult:
        """
        Uses fb.is_clean label injected by corruption engine.
        If not provided, assumes clean data.
        """
        is_clean = True if fb.is_clean is None else fb.is_clean

        if self.rng.random() < self.p_correct:
            passed = is_clean
        else:
            passed = not is_clean

        score = 1.0 if passed else 0.0
        return AgentResult(
            score=score,
            passed=passed,
            rationale=f"proxy_verifier(p_correct={self.p_correct})"
        )
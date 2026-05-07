# feedback_pipeline/pipeline.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base import Feedback
from agents.proxy_verifiers import ProxyVerifier


class FeedbackPipeline:
    """
    Multi-agent verification pipeline supporting either:
      - proxy verifiers (for controlled experiments), or
      - real LLM-based verifiers.

    Returns:
      accepted_feedback: List[Feedback]
      full_results: List[Dict[str, Any]]   # fully serializable records
    """

    def __init__(
        self,
        mode: str = "proxy",   # "proxy" or "llm"
        proxy_p_correct: float = 0.8,
        seed: int = 42,
        kv_kwargs: Optional[dict] = None,
        ba_kwargs: Optional[dict] = None,
        ee_kwargs: Optional[dict] = None,
        ta_kwargs: Optional[dict] = None,
    ):
        self.mode = mode
        self.proxy_p_correct = proxy_p_correct
        self.seed = seed

        kv_kwargs = kv_kwargs or {}
        ba_kwargs = ba_kwargs or {}
        ee_kwargs = ee_kwargs or {}
        ta_kwargs = ta_kwargs or {}

        if mode == "proxy":
            self.agents = [
                ("Knowledge", ProxyVerifier("Knowledge", proxy_p_correct, seed)),
                ("Behavior", ProxyVerifier("Behavior", proxy_p_correct, seed + 1)),
                ("Ethics", ProxyVerifier("Ethics", proxy_p_correct, seed + 2)),
                ("Trust", ProxyVerifier("Trust", proxy_p_correct, seed + 3)),
            ]
        elif mode == "llm":
            # Lazy imports so proxy mode does not require OPENAI_API_KEY
            from agents.knowledge_verifier import KnowledgeVerifier
            from agents.behavior_auditor import BehaviorAuditor
            from agents.ethics_evaluator import EthicsEvaluator
            from agents.trust_assessor import TrustAssessor

            self.agents = [
                ("Knowledge", KnowledgeVerifier(**kv_kwargs)),
                ("Behavior", BehaviorAuditor(**ba_kwargs)),
                ("Ethics", EthicsEvaluator(**ee_kwargs)),
                ("Trust", TrustAssessor(**ta_kwargs)),
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'proxy' or 'llm'.")

    @staticmethod
    def _feedback_to_serializable_dict(fb: Feedback) -> Dict[str, Any]:
        d = fb.model_dump() if hasattr(fb, "model_dump") else fb.dict()

        for attr in ["id", "dataset", "split", "source_row", "is_clean"]:
            if hasattr(fb, attr):
                d[attr] = getattr(fb, attr)

        return d

    def run(self, batch: List[Feedback], min_pass: int):
        accepted_feedback: List[Feedback] = []
        full_results: List[Dict[str, Any]] = []

        for fb in batch:
            agent_results: Dict[str, Any] = {}

            for agent_name, agent in self.agents:
                res = agent.verify(fb)

                threshold = getattr(agent, "threshold", None)
                model_name = getattr(agent, "model_name", None)
                verifier_name = getattr(agent, "name", agent_name)

                agent_results[agent_name] = {
                    "verifier_name": verifier_name,
                    "score": float(res.score),
                    "passed": bool(res.passed),
                    "rationale": str(res.rationale),
                    "threshold": threshold,
                    "model_name": model_name,
                    "mode": self.mode,
                }

            passes = sum(1 for r in agent_results.values() if r["passed"])
            accepted = passes >= min_pass

            item = self._feedback_to_serializable_dict(fb)
            item["agent_results"] = agent_results
            item["num_passed"] = passes
            item["min_pass"] = min_pass
            item["accepted"] = accepted
            item["pipeline_mode"] = self.mode

            if accepted:
                accepted_feedback.append(fb)

            full_results.append(item)

        return accepted_feedback, full_results
# agents/knowledge_verifier.py

from agents.base import BaseAgent, Feedback, AgentResult
from agents.local_llm_backend import LocalLLMBackend


class KnowledgeVerifier(BaseAgent):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        threshold: float = 0.7,
    ):
        self.backend = LocalLLMBackend(model_name=model_name)
        self.model_name = model_name
        self.threshold = threshold

        self.system_prompt = (
            "You evaluate factual correctness.\n"
            "You must output one JSON object only.\n"
            "Use a real numeric score between 0.0 and 1.0.\n"
            "Do not use placeholders like ...\n"
            "Do not use markdown.\n"
            "JSON keys must be exactly: score, rationale."
        )

    def verify(self, fb: Feedback) -> AgentResult:
        chosen = fb.response_a if fb.user_choice == "A" else fb.response_b

        user_msg = (
            "Task: rate the factual correctness of the chosen response.\n\n"
            f"User prompt: {fb.prompt}\n\n"
            f"Chosen response: {chosen}\n\n"
            "Scoring guide:\n"
            "- 1.0 = fully factually correct\n"
            "- 0.8 = mostly correct with minor issues\n"
            "- 0.5 = mixed or uncertain correctness\n"
            "- 0.2 = mostly incorrect\n"
            "- 0.0 = clearly false or fabricated\n\n"
            "Return exactly one JSON object.\n"
            'Example valid format: {"score": 0.9, "rationale": "Factually correct."}'
        )

        content = self.backend.generate([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ])

        data = self.backend.extract_json(content)
        score = max(0.0, min(1.0, float(data["score"])))
        rationale = str(data.get("rationale", ""))

        return AgentResult(score=score, passed=score >= self.threshold, rationale=rationale)
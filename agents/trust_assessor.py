# agents/trust_assessor.py

from agents.base import BaseAgent, Feedback, AgentResult
from agents.local_llm_backend import LocalLLMBackend


class TrustAssessor(BaseAgent):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        threshold: float = 0.7,
    ):
        self.backend = LocalLLMBackend(model_name=model_name)
        self.model_name = model_name
        self.threshold = threshold

        self.system_prompt = (
            "You are a trustworthiness assessor.\n"
            "Your only job is to judge whether the chosen response is reliable, coherent, and non-misleading.\n"
            "A response must receive a LOW score if it contains factual errors, conspiracy claims, fabricated claims, misleading medical claims, or deceptive framing.\n"
            "A correct, neutral, coherent factual answer should usually receive a HIGH score.\n"
            "You are not grading style. You are grading reliability and whether a user should trust this response.\n"
            "Return exactly one JSON object with keys: score, rationale.\n"
            "Use a real numeric score between 0.0 and 1.0.\n"
            "Do not use placeholders such as '...'.\n"
            "Do not return markdown or extra text."
        )

    def verify(self, fb: Feedback) -> AgentResult:
        chosen = fb.response_a if fb.user_choice == "A" else fb.response_b

        user_msg = (
            "Evaluate ONLY the trustworthiness of the chosen response.\n\n"
            f"User prompt: {fb.prompt}\n\n"
            f"Chosen response: {chosen}\n\n"
            "Scoring guide:\n"
            "- 1.0 = highly trustworthy, fully reliable, non-misleading\n"
            "- 0.8 = mostly trustworthy with minor uncertainty\n"
            "- 0.5 = mixed reliability or partially misleading\n"
            "- 0.2 = largely untrustworthy or misleading\n"
            "- 0.0 = clearly false, fabricated, conspiratorial, deceptive, or dangerous misinformation\n\n"
            "Important rules:\n"
            "- If the response contains factual misinformation, the score must be below 0.5.\n"
            "- If the response contains conspiracy-style or fabricated medical claims, the score must be near 0.0.\n"
            "- Do NOT give a high score just because the wording sounds confident or plausible.\n\n"
            'Return exactly one JSON object like {"score": 0.15, "rationale": "Contains misleading and unreliable misinformation."}'
        )

        content = self.backend.generate([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ])

        data = self.backend.extract_json(content)
        score = max(0.0, min(1.0, float(data["score"])))
        rationale = str(data.get("rationale", "")).strip()

        return AgentResult(
            score=score,
            passed=score >= self.threshold,
            rationale=rationale,
        )
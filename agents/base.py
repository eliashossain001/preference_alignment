# agents/base.py

from abc import ABC, abstractmethod
from typing import Literal, Optional
from pydantic import BaseModel


class Feedback(BaseModel):
    """
    A single preference item from a user:
     - prompt: the user's original query
     - response_a/b: the two candidate model outputs
     - user_choice: which response the user preferred ("A" or "B")

    Optional metadata fields are included for provenance and reproducibility.
    """
    prompt: str
    response_a: str
    response_b: str
    user_choice: Literal["A", "B"]

    # provenance / metadata
    id: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    source_row: Optional[int] = None

    # corruption / simulation metadata
    is_clean: Optional[bool] = None


class AgentResult(BaseModel):
    """
    What each agent returns:
     - score: float in [0.0,1.0]
     - passed: bool (did score ≥ threshold?)
     - rationale: brief explanation
    """
    score: float
    passed: bool
    rationale: str


class BaseAgent(ABC):
    @abstractmethod
    def verify(self, fb: Feedback) -> AgentResult:
        """
        Inspect one Feedback item and return an AgentResult.
        """
        ...
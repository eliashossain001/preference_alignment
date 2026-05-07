# agents/local_llm_backend.py

from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLMBackend:
    _MODEL = None
    _TOKENIZER = None
    _MODEL_NAME = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ensure_loaded()

    def _ensure_loaded(self):
        if (
            LocalLLMBackend._MODEL is not None
            and LocalLLMBackend._TOKENIZER is not None
            and LocalLLMBackend._MODEL_NAME == self.model_name
        ):
            self.model = LocalLLMBackend._MODEL
            self.tokenizer = LocalLLMBackend._TOKENIZER
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if self.device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        model.eval()

        LocalLLMBackend._MODEL = model
        LocalLLMBackend._TOKENIZER = tokenizer
        LocalLLMBackend._MODEL_NAME = self.model_name

        self.model = model
        self.tokenizer = tokenizer

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    def _infer_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return self.device

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt_text = self._format_messages(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_device = self._infer_device()
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated.startswith(prompt_text):
            generated = generated[len(prompt_text):].strip()

        return generated.strip()

    @staticmethod
    def extract_json(text: str) -> Dict:
        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        if "```json" in text:
            blocks = []
            start = 0
            while True:
                idx = text.find("```json", start)
                if idx == -1:
                    break
                obj_start = text.find("{", idx)
                obj_end = text.find("```", obj_start)
                if obj_start != -1 and obj_end != -1:
                    candidate = text[obj_start:obj_end].strip()
                    try:
                        blocks.append(json.loads(candidate))
                    except Exception:
                        pass
                start = idx + 7
            if blocks:
                return blocks[-1]

        # collect all balanced JSON objects and return the last one
        objs = []
        start_idx = None
        depth = 0

        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate = text[start_idx:i + 1]
                        try:
                            objs.append(json.loads(candidate))
                        except Exception:
                            pass
                        start_idx = None

        if objs:
            return objs[-1]

        raise ValueError(f"Could not extract JSON from model output:\n{text}")
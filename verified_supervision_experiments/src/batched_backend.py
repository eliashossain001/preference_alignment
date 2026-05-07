"""Batched HF-causal-LM backend for the verifier pool.

Loads one model and provides `generate_batch(messages_list)` -> list[str], so
the four role prompts for a single (prompt, response) pair can be processed in
a single forward pass instead of four sequential ones.

Tokenizer left-padding + attention mask are used to handle variable-length
prompts in the same batch.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BatchedLLMBackend:
    _MODEL = None
    _TOK = None
    _NAME = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 96,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if (
            BatchedLLMBackend._MODEL is not None
            and BatchedLLMBackend._NAME == self.model_name
        ):
            self.model = BatchedLLMBackend._MODEL
            self.tokenizer = BatchedLLMBackend._TOK
            return
        tok = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs = {"trust_remote_code": True}
        if self.device.startswith("cuda"):
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        model.eval()
        BatchedLLMBackend._MODEL = model
        BatchedLLMBackend._TOK = tok
        BatchedLLMBackend._NAME = self.model_name
        self.model = model
        self.tokenizer = tok

    def _format(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    @torch.no_grad()
    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [self._format(m) for m in batch_messages]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        device = next(self.model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gens = []
        for i in range(out.size(0)):
            tokens = out[i, input_ids.size(1):]
            gens.append(self.tokenizer.decode(tokens, skip_special_tokens=True).strip())
        return gens

    @staticmethod
    def extract_json(text: str) -> Dict:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        if "```json" in text:
            blocks = []
            i = 0
            while True:
                idx = text.find("```json", i)
                if idx == -1:
                    break
                obj_start = text.find("{", idx)
                obj_end = text.find("```", obj_start)
                if obj_start != -1 and obj_end != -1:
                    cand = text[obj_start:obj_end].strip()
                    try:
                        blocks.append(json.loads(cand))
                    except Exception:
                        pass
                i = idx + 7
            if blocks:
                return blocks[-1]
        objs, depth, start = [], 0, None
        for j, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = j
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        cand = text[start : j + 1]
                        try:
                            objs.append(json.loads(cand))
                        except Exception:
                            pass
                        start = None
        if objs:
            return objs[-1]
        raise ValueError(f"could not extract JSON from: {text!r}")

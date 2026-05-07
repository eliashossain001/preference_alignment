# fine_tuning/dpo_trainer.py

import inspect
from typing import List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from agents.base import Feedback


class DPOTrainerHelper:
    def __init__(
        self,
        base_model: str,
        ref_model: Optional[str] = None,
        output_dir: str = "./dpo_output",
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        epochs: int = 3,
        max_length: int = 512,
        seed: int = 42,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.max_length = max_length
        self.beta = beta
        self.loss_type = loss_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if use_lora:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        model_kwargs = {"trust_remote_code": True}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        ref = ref_model or base_model
        self.ref_model = AutoModelForCausalLM.from_pretrained(ref, **model_kwargs)

        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_cfg)

    def _build_dataset(self, feedback: List[Feedback]) -> Dataset:
        records = []
        for fb in feedback:
            chosen = fb.response_a if fb.user_choice == "A" else fb.response_b
            rejected = fb.response_b if fb.user_choice == "A" else fb.response_a
            records.append({"prompt": fb.prompt, "chosen": chosen, "rejected": rejected})
        return Dataset.from_list(records)

    def train(self, feedback: List[Feedback]) -> DPOTrainer:
        ds = self._build_dataset(feedback)

        dpo_args = DPOConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            seed=self.seed,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            max_length=self.max_length,
            beta=self.beta,
            loss_type=self.loss_type,
        )

        trainer_kwargs = dict(
            model=self.model,
            ref_model=self.ref_model,
            train_dataset=ds,
            args=dpo_args,
        )

        sig = inspect.signature(DPOTrainer.__init__)
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in sig.parameters:
            trainer_kwargs["tokenizer"] = self.tokenizer

        trainer = DPOTrainer(**trainer_kwargs)
        trainer.train()
        trainer.save_model(self.output_dir)
        return trainer
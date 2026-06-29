"""Simple trainer for fine-tuning on SST-2 or custom datasets.

Uses HuggingFace Trainer under the hood for simplicity.
This module is not exercised by unit tests (requires data download).
"""

from __future__ import annotations

from typing import Optional

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

from uncertainty_classifier.model.classifier import UncertaintyClassifier


def train_on_sst2(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./checkpoints",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_train_samples: Optional[int] = None,
) -> UncertaintyClassifier:
    """Fine-tune DistilBERT on SST-2 (sentiment classification).

    Args:
        model_name: Pretrained model identifier.
        output_dir: Where to save checkpoints.
        num_train_epochs: Training epochs.
        per_device_train_batch_size: Batch size per device.
        per_device_eval_batch_size: Eval batch size.
        learning_rate: Learning rate.
        max_train_samples: Truncate training set (for quick debugging).

    Returns:
        Fine-tuned UncertaintyClassifier.
    """
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model = UncertaintyClassifier(
        model_name=model_name,
        num_labels=2,
        id2label=id2label,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("glue", "sst2")
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        report_to="none",
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model.backbone,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return model

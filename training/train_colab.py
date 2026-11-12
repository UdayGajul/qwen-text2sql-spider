"""Train the Spider Text-to-SQL LoRA adapter on a Colab T4.

The script intentionally mirrors the original Kaggle notebook while removing
Kaggle-only APIs. Training data comes from the Hugging Face Spider dataset;
schema metadata comes from the official Spider checkout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM_PROMPT = (
    "You are an expert SQL generator. Given a database schema and a natural language "
    "question, write the correct SQL query. Output only the SQL query with no explanation "
    "or markdown."
)


def schema_prompt(db_id: str, schemas: dict) -> str:
    db = schemas[db_id]
    tables = db["table_names_original"]
    columns = db["column_names_original"]
    types = db["column_types"]
    primary_keys = set(db["primary_keys"])
    type_map = {"text": "TEXT", "number": "REAL", "time": "TEXT", "boolean": "INTEGER"}
    blocks = []

    for table_id, table_name in enumerate(tables):
        table_columns = []
        for column_id, (owner_id, name) in enumerate(columns):
            if owner_id != table_id:
                continue
            primary_key = " PRIMARY KEY" if column_id in primary_keys else ""
            sql_type = type_map.get(types[column_id].lower(), "TEXT")
            table_columns.append(f"  {name} {sql_type}{primary_key}")
        if table_columns:
            blocks.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(table_columns) + "\n)")

    foreign_keys = [
        f"-- {tables[columns[source][0]]}.{columns[source][1]} references "
        f"{tables[columns[target][0]]}.{columns[target][1]}"
        for source, target in db["foreign_keys"]
    ]
    result = "\n\n".join(blocks)
    return result + ("\n\n" + "\n".join(foreign_keys) if foreign_keys else "")


def build_training_dataset(train_rows: list[dict], schemas: dict, tokenizer, max_length: int):
    kept = []
    dropped = 0
    skipped = 0
    for row in train_rows:
        if row["db_id"] not in schemas:
            skipped += 1
            continue
        user = f"### Database Schema:\n{schema_prompt(row['db_id'], schemas)}\n\n"
        user += f"### Question:\n{row['question']}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["query"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if len(tokenizer(text)["input_ids"]) <= max_length:
            kept.append(
                {
                    "prompt": messages[:2],
                    "completion": [messages[2]],
                }
            )
        else:
            dropped += 1
    print(f"Training examples kept={len(kept)} dropped={dropped} skipped={skipped}")
    return Dataset.from_list(kept)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("qwen-text2sql-adapter"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("qwen-text2sql-checkpoints"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab select Runtime > Change runtime type > T4 GPU.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    with args.tables.open(encoding="utf-8") as handle:
        schemas = {row["db_id"]: row for row in json.load(handle)}
    spider = load_dataset("xlangai/spider")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"
    train_dataset = build_training_dataset(
        list(spider["train"]), schemas, tokenizer, args.max_length
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    training_config = SFTConfig(
        output_dir=str(args.checkpoint_dir),
        report_to="none",
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=0.3,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="paged_adamw_8bit",
        dataloader_pin_memory=False,
        group_by_length=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir.resolve()}")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

from src.data import CustomDataset, DataCollatorForSupervisedDataset


# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Training about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="model file path")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer path")
g.add_argument("--save_dir", type=str, default="resource/results", help="model save path")
g.add_argument("--batch_size", type=int, default=1, help="batch size (both train and eval)")
g.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
g.add_argument("--warmup_steps", type=int, help="scheduler warmup steps")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate")
g.add_argument("--epoch", type=int, default=5, help="training epoch")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 혼합 정밀도 사용
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CustomDataset("resource/data/대화맥락추론_train.json", tokenizer)
    valid_dataset = CustomDataset("resource/data/대화맥락추론_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
    })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=args.save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.1,
        num_train_epochs=args.epoch,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        log_level="info",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=5,
        bf16=True,  # bfloat16 사용
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=1024,
        packing=True,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    success = False
    attempts = 0
    max_attempts = 5

    while not success and attempts < max_attempts:
        try:
            trainer.train()
            success = True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            attempts += 1
            print(f"CUDA memory error, reducing batch size and retrying... (Attempt {attempts}/{max_attempts})")
            args.batch_size = max(1, args.batch_size // 2)  # 배치 크기 줄이기
            training_args.per_device_train_batch_size = args.batch_size
            training_args.per_device_eval_batch_size = args.batch_size
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                args=training_args,
            )

    if not success:
        print("Training failed after maximum attempts due to CUDA memory errors.")


if __name__ == "__main__":
    exit(main(parser.parse_args()))

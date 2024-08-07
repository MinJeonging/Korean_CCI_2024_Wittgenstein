import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, TrainingArguments
from trl import SFTTrainer, SFTConfig
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from peft import LoraConfig, PeftModel


# Argument Parser Configuration
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


lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

def main(args):
    # Model and Tokenizer Loading
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation='eager'
    )
    
    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, add_special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Custom Dataset Loading
    train_dataset = CustomDataset("/root/gemmaquan/resource/data/대화맥락추론_train+dev_120.json", tokenizer)
    valid_dataset = CustomDataset("/root/gemmaquan/resource/data/대화맥락추론_dev_121.json", tokenizer)

    # Conversion to Hugging Face Dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        'labels': train_dataset.label,
    })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        'labels': valid_dataset.label,
    })
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Training Configuration
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
        logging_dir=f"{args.save_dir}/logs",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=args.epoch,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=2048,
        packing=True,
        seed=42,
    )

    # Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        peft_config=lora_config
    )

    # Start Training
    trainer.train()

    ADAPTER_MODEL = "lora_adapter"

    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained(f"{args.save_dir}/ft_model")

if __name__ == "__main__":
    exit(main(parser.parse_args()))
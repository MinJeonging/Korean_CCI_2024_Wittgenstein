import argparse
import torch
import numpy as np
from sklearn.model_selection import KFold
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
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
g.add_argument("--k_folds", type=int, default=5, help="number of k-folds")

lora_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
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
    dataset = CustomDataset("/root/gemma2/resource/data/대화맥락추론_train+dev.json", tokenizer)
    dataset = Dataset.from_dict({
        'input_ids': dataset.inp,
        'labels': dataset.label,
    })

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

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

    for epoch in range(args.epoch):
        print(f"Epoch {epoch+1}/{args.epoch}")
        fold = 0
        train_losses = []
        val_losses = []
        
        for train_index, val_index in kf.split(np.arange(len(dataset))):
            fold += 1
            print(f"  Fold {fold}/{args.k_folds}")
            train_subdataset = dataset.select(train_index)
            val_subdataset = dataset.select(val_index)
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_subdataset,
                eval_dataset=val_subdataset,
                data_collator=data_collator,
                args=training_args,
                peft_config=lora_config
            )
            
            train_result = trainer.train()
            val_result = trainer.evaluate()
            
            print(f"Train result: {train_result}")
            print(f"Validation result: {val_result}")

            # Extracting loss values based on the observed structure
            train_loss = train_result[0] if isinstance(train_result, tuple) else train_result['training_loss']
            val_loss = val_result[0] if isinstance(val_result, tuple) else val_result['eval_loss']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"  Average Training Loss: {avg_train_loss}")
        print(f"  Average Validation Loss: {avg_val_loss}")

    ADAPTER_MODEL = "lora_adapter"

    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained(f"{args.save_dir}/ft_model")

if __name__ == "__main__":
    exit(main(parser.parse_args()))

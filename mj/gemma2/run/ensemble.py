import argparse
import json
import tqdm
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset

# 명령행 인수 설정
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--output_dir", type=str, required=True, help="directory to save output files")
g.add_argument("--model_ids", type=str, nargs='+', required=True, help="list of huggingface model ids")
g.add_argument("--ft_save_dirs", type=str, nargs='+', required=True, help="list of finetuned model dirs")
g.add_argument("--tokenizers", type=str, nargs='+', help="list of huggingface tokenizers")
g.add_argument("--device", type=str, required=True, help="device to load the models")

def main(args):
    if args.tokenizers is None:
        args.tokenizers = args.model_ids

    datasets = []
    tokenizers = []
    for tokenizer_id in args.tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, add_special_tokens=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)
        datasets.append(CustomDataset("/root/gemma2/resource/data/correct_answer.json", tokenizer))

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("/root/gemma2/resource/data/correct_answer.json", "r") as f:
        original_result = json.load(f)

    all_model_results = []
    
    for model_idx, (model_id, ft_save_dir, tokenizer, dataset) in enumerate(zip(args.model_ids, args.ft_save_dirs, tokenizers, datasets)):
        FINETUNE_MODEL = f"{ft_save_dir}/ft_model"

        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            FINETUNE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

        model_results = []

        correct_results = []
        incorrect_results = []

        for idx in tqdm.tqdm(range(len(dataset))):
            inp, _ = dataset[idx]
            try:
                outputs = model(inp.to(args.device).unsqueeze(0))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs = model(inp.to(args.device).unsqueeze(0))

            logits = outputs.logits[:, -1].flatten()
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[tokenizer.vocab['A']],
                            logits[tokenizer.vocab['B']],
                            logits[tokenizer.vocab['C']],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )

            predicted_label = answer_dict[np.argmax(probs)]
            model_results.append(predicted_label)
            
            original_label = original_result[idx].get("output")
            result_entry = original_result[idx].copy()
            result_entry["output"] = predicted_label

            entry_with_labels = result_entry.copy()
            entry_with_labels["predicted_output"] = predicted_label
            if original_label is not None:
                entry_with_labels["original_output"] = original_label
                if original_label == predicted_label:
                    correct_results.append(entry_with_labels)
                else:
                    incorrect_results.append(entry_with_labels)

        all_model_results.append(model_results)

        os.makedirs(args.output_dir, exist_ok=True)
        
        # 모델별 결과 파일 저장
        model_specific_result = [entry.copy() for entry in original_result]
        for idx in range(len(model_specific_result)):
            model_specific_result[idx]["output"] = model_results[idx]

        output_path = os.path.join(args.output_dir, f"{args.output.replace('.json', '')}_model_{model_idx+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_specific_result, ensure_ascii=False, indent=4))
        
        if correct_results:
            correct_output_path = os.path.join(args.output_dir, f"{args.output.replace('.json', '')}_model_{model_idx+1}_correct.json")
            with open(correct_output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(correct_results, ensure_ascii=False, indent=4))

        if incorrect_results:
            incorrect_output_path = os.path.join(args.output_dir, f"{args.output.replace('.json', '')}_model_{model_idx+1}_incorrect.json")
            with open(incorrect_output_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(incorrect_results, ensure_ascii=False, indent=4))

        # 모델 언로드
        del model
        torch.cuda.empty_cache()

    ensemble_results = []

    for i in range(len(original_result)):
        model_votes = [model_result[i] for model_result in all_model_results]
        vote_counts = {label: model_votes.count(label) for label in answer_dict.values()}
        majority_vote = max(vote_counts, key=vote_counts.get)
        if list(vote_counts.values()).count(vote_counts[majority_vote]) == 1:
            final_label = majority_vote
        else:
            logits = np.array(
                [
                    [
                        outputs.logits[:, -1].flatten()[tokenizer.vocab['A']],
                        outputs.logits[:, -1].flatten()[tokenizer.vocab['B']],
                        outputs.logits[:, -1].flatten()[tokenizer.vocab['C']],
                    ]
                    for outputs, tokenizer in zip(all_model_results, tokenizers)
                ]
            )
            final_label = answer_dict[np.argmax(logits.sum(axis=0))]

        original_result[i]["output"] = final_label

    # 앙상블 결과 파일 저장
    ensemble_output_path = os.path.join(args.output_dir, args.output)
    with open(ensemble_output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(original_result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    exit(main(parser.parse_args()))

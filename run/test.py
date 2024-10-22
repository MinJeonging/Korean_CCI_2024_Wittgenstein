import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # 혼합 정밀도 사용
        device_map='auto' if torch.cuda.is_available() else None, # 자동으로 GPU 메모리를 관리
    )
    model.eval()

    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = CustomDataset("/root/Korean_CCI_2024_Wittgenstein/resource/data/대화맥락추론_test.json", tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("/root/Korean_CCI_2024_Wittgenstein/resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    correct_results = []
    incorrect_results = []

    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _ = dataset[idx]
        try:
            outputs = model(
                inp.to(args.device).unsqueeze(0)
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache() # 캐시 비우기
            outputs = model(
                inp.to(args.device).unsqueeze(0)
            )
        
        logits = outputs.logits[:,-1].flatten()
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

        predicted_label = answer_dict[numpy.argmax(probs)]
        original_label = result[idx].get("output")
        result[idx]["output"] = predicted_label

        entry_with_labels = result[idx].copy()
        entry_with_labels["predicted_output"] = predicted_label
        if original_label is not None:
            entry_with_labels["original_output"] = original_label
            if original_label == predicted_label:
                correct_results.append(entry_with_labels)
            else:
                incorrect_results.append(entry_with_labels)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    
    if correct_results:
        with open(args.output.replace(".json", "_correct.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(correct_results, ensure_ascii=False, indent=4))

    if incorrect_results:
        with open(args.output.replace(".json", "_incorrect.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(incorrect_results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))

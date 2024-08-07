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
g.add_argument("--batch_size", type=int, default=16, help="batch size for inference")
# fmt: on

def main(args):
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='cuda' if torch.cuda.is_available() else None,
    )
    model.eval()
    print("Model loaded successfully.")

    if args.tokenizer is None:
        args.tokenizer = args.model_id
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")
    
    print("Loading dataset...")
    dataset = CustomDataset("/root/Korean_CCI_2024_Wittgenstein/resource/data/대화맥락추론_test.json", tokenizer)
    print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("/root/Korean_CCI_2024_Wittgenstein/resource/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)
    print("Original result file loaded.")

    batch_size = args.batch_size
    with torch.no_grad():
        for idx in tqdm.tqdm(range(0, len(dataset), batch_size)):
            print(f"Processing batch starting at index {idx}...")
            batch_inputs = [dataset[i][0].to(args.device) for i in range(idx, min(idx + batch_size, len(dataset)))]
            batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
            outputs = model(batch_inputs)
            logits = outputs.logits[:, -1, :]

            for i, logit in enumerate(logits):
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logit[tokenizer.vocab['A']],
                                logit[tokenizer.vocab['B']],
                                logit[tokenizer.vocab['C']],
                            ]
                        ),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                result[idx + i]["output"] = answer_dict[numpy.argmax(probs)]
            print(f"Batch processed successfully up to index {idx + batch_size - 1}.")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    print(f"Results written to {args.output}.")

if __name__ == "__main__":
    print("Parsing arguments...")
    args = parser.parse_args()
    print("Arguments parsed successfully.")
    exit(main(args))

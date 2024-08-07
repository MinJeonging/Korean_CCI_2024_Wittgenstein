import json
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100
        self.inp = []
        self.trg = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        answer_dict = {
            "": None,
            "inference_1": 0,
            "inference_2": 1,
            "inference_3": 2
        }

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            reference_id = f"[Reference id]\n{inp['reference_id']}\n"

            # 카테고리에 따른 조건문
            question = "[Question]\n대화의 맥락을 바탕으로 "
            if inp['category'] == "전제":
                question += "화자1과 화자2의 [Reference id]에 해당하는 발화문 이면에 깔린 전제 사실을 가능한 한 많이 추론하세요."
            elif inp['category'] == "원인":
                question += "화자1과 화자2의 [Reference id]에 해당하는 발화문 이면에 깔린 원인 사실을 추론하세요."
            elif inp['category'] == "동기":
                question += "화자1과 화자2의 [Reference id]에 해당하는 발화를 일으키는 화자1의 감정이나 기본 욕구을 추론하세요."
            elif inp['category'] == "반응":
                question += "화자1이 화자2의 [Reference id]에 해당하는 발화문을 듣고 느낀 반응과 화자2이 화자1의 [Reference id]에 해당하는 발화문을 듣고 느낀 반응을 각각 추론하세요."
            else:
                question += "[Reference id]을 힌트로 삼아 이 대화 이후 화자 1과 화자2에게 어떤 일이 벌어질지(후행사건)을 각각 추론하세요. "
            question += " 이때, 중요한 정보를 놓치지 않도록 유의하면서 대화의 주요 맥락과 흐름을 고려하십시오. 추론한 문장과 주어진 3개의 Option 중 가장 유사한 맥락의 지문은?"

            chat = chat + "\n\n" + reference_id + "\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat

        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = ""
            if example["output"] == "inference_1":
                target = f"A. {example['input']['inference_1']}{tokenizer.eos_token}"
            elif example["output"] == "inference_2":
                target = f"B. {example['input']['inference_2']}{tokenizer.eos_token}"
            elif example["output"] == "inference_3":
                target = f"C. {example['input']['inference_3']}{tokenizer.eos_token}"
                
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.trg.append(answer_dict[example["output"]])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx], self.trg[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
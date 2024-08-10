import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX = -100
        self.inp = []
        self.trg = []
        self.label = []
        self.categories = []

        explanation = {
            "반응": ['response is a subject to infer the emotional response the listener can show to the target utterance.',
                     '반응은 대상 발화 사건에 대해 \'청자\'가 보일 수 있는 감정 반응을 추론하는 주제입니다.'],
            "후행사건": ['following is a subject to infer the events that can happen after the target utterance.',
                        '후행사건은 대상 발화 이후에 일어날 수 있는 사건을 추론하는 주제입니다.'],
            "동기": ['motivation is a subject to infer the speaker\'s emotion or basic desire that triggers the target utterance.',
                    '동기는 대상 발화를 일으키는 화자의 감정이나 기본 욕구를 추론하는 주제입니다.'],
            "원인": ['cause is a subject to infer the events that trigger the target utterance.',
                    '원인은 대상 발화의 사건을 유발하는 사건을 추론하는 주제입니다.'],
            "전제": ['premise is a subject to infer the events that are necessary for the target utterance to happen.',
                    '전제는 대상 발화의 사건을 가능하게 하는 상태 혹은 사건을 추론하는 주제입니다.']
        }

        category_perspective = {
            "반응": "Read each utterance in the conversation by paying attention on the listener's emotion.",
            "원인": "Read the conversation by paying attention on the listener's emotion.",
            "동기": "Read each utterance by imagining the possible situations and emotions that cause the speaker say so.",
            "후행사건": "Read each utterance by imagining what will happen after the speaker's utterance.",
            "전제": "Read each utterance by inferring the necessary conditions that makes the speaker's utterance be true."
        }

        category_specified_guidance = {
            "반응" : '''Return your answers by the following process:
            First, guess the speaker's emotion.
            Second, find if the listener have explicitly reacted to the speaker.
            Third, if the listener reacted to the speaker, guess how the listener's emotion is.
            Otherwise, consider the neighboring utterances of the reference sentence and imagine how the listener would feel.
            Fourth, find the keyword in the conversation that makes the listener have such emotion.
            Fifth, according to the key word and the listener's emotion you thought, find the most similar option from your answer.
            ''',
            "원인":'''Make inference by the following process:
            First, write a sentence that can offer new information to a third person.
            Second, write a pair of the topic of reference sentence(s) and the key word(s).
            Third, find the best statement that explains why the statement is spoken.
            Fourth, find the most simliar option from the inference options.''',
            "동기":"""Make inference by the following process:
            First, find the speaker's overall emotion in the conversation.
            Second, extract the speakers behavior or thought from the reference sentence(s).
            Third, infer the speaker's emotion or desire of the given utterance.
            Fourth, think if there is any benefit if the utterance succeeds.
            If there is no benefit, think the intention of the speaker's utterance.""",
            "후행사건":"""Make inference by the following process:
            First, find if the reference sentence can trigger any incidence.
            Second, if there is at least one, then list the possible incidences.
            Third, find the key words from the utterance. Mind that the keyword can be in other utterance.
            Fourth, find the closest option that includes the keywords among the listed options.""",
            "전제":"""Make inference by following the process:
            First, write the given reference sentence as a statement.
            Second, list up the possible reasons how the statement could be true.
            Third, compare the inference options and select the closest option among the given list."""
        }

        category_specified_content_of_json:dict = {
            "반응":'''
            Here are the given keys in the json file and required type of value.
            {
                "speaker's_emotion":str,
                "existence_of_listener's_response":bool,
                "listener's_emotion":bool,
                "keyword":str,
                "output":str,
                }
                Fill up the values according to the keys.''',
            "원인":'''
            Here are the given keys in the json file and required type of value.
            {
                "new_information":str,
                "topic_and_keyword":tuple(str, str),
                "explanation":str,
                "output":str,
            }
            Fill up the values according to the keys.
            ''',
            "동기":"""
            Here are the given keys in the json file and required type of value.
            {
                "overall_emotion":str,
                "behavior_or_thought":str,
                "emotion_or_desire":str,
                "benefit":str,
                "output":str
            }
            Fill up the values according to the keys.
            """,
            "후행사건":"""
            Here are the given keys in the json file and required type of value.
            {
                "incidence_trigger":bool,
                "possible_incidences":list[str, ..., str],
                "keywords":list[str, ..., str],
                "output":str
            }
            Fill up the values according to the keys.
            """,
            "전제":"""
            Here are the given keys in the json file and required type of value.
            {
                "statement":str,
                "possible_reasons":list,
                "output":str
            }
            Fill up the values according to the keys.
            """
        }
        
        front_prompt:str = '''<Instruction>
        You have a deep understanding with common sense and daily conversation.
        Given the conversation, take a deep attention with the reference sentence(s) and the common sense category.
        Perspective when reading : {category_perspective}
        </Instruction>
        
        <Conversation>'''
        back_prompt:str = '''</conversation>
        
        <Reference>
        Category : {category}
        Explanation : {explanation}
        Reference sentence(s) : {reference_sentence}
        </Reference>
        
        <Guidance>
        {category_specified_guidance}
        There is only one answer.
        Return your final answer in json format.
        {category_specified_content_of_json}
        </Guidance>'''
        answer_prompt:str = """<Answer>
        Let's think step by step.
        """

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
                if "name" in str(utterance):
                    utterance = utterance.replace("name", "화자")
                chat.append(f"화자{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"

            chat = chat + "\n\n" + question + "\n\n[Option]\n"
            chat += f"A. {inp['inference_1']}\n"
            chat += f"B. {inp['inference_2']}\n"
            chat += f"C. {inp['inference_3']}"

            return chat

        def make_dialogue(inp):
            chat:list = []
            for cvt in inp['conversation']:
                speaker = "화자" + str(cvt['speaker'])
                utterance = cvt['utterance']
                if "name" in str(utterance):
                    utterance = utterance.replace("name", "화자")
                speak = {"role": speaker, "content":utterance}
                chat.append(speak)

            question = f"[Question]\n위 대화의 {inp['category']}"
            if (ord(inp['category'][-1]) - ord("가")) % 28 > 0:
                question += "으로"
            else:
                question = "로"
            question += " 올바른 지문은?"

            question = "\n\n[Option]\n"
            question += f"A. {inp['inference_1']}\n"
            question += f"B. {inp['inference_2']}\n"
            question += f"C. {inp['inference_3']}"
            print(chat)

            return chat, question
        
        for example in data:
            chat, question = make_dialogue(example["input"])
            # breakpoint()
            category = example['input']['category']
            conversation = example['input']['conversation']
            reference_id = example['input']['reference_id']
            reference_sentence = " ".join([utterance['utterance'] for utterance in conversation if utterance['utterance_id'] in reference_id])
            
            message = [
                {"role": "system", "content": f"{front_prompt.format(category_perspective=category_perspective)}"},
            ]
            message.extend(chat)
            message.append({"role":"system", "content": f"{back_prompt.format(category=category, explanation=explanation[category][0], reference_sentence=reference_sentence, category_specified_guidance=category_specified_guidance[category], category_specified_content_of_json=category_specified_content_of_json[category])}"})
            message.append({"role":"system", "content": f"{question}"})
            message.append({"role":"answer", "content":f"{answer_prompt}"})

            print(message)
            # breakpoint()
            

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
            self.categories.append(category)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {'input_ids': self.inp[idx], 'labels': self.label[idx], 'category': self.categories[idx]}
    
    def get_categories(self):
        return self.categories


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

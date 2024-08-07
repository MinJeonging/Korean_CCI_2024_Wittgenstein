import json
import pandas as pd

# JSON 파일 읽기
with open('/root/gemmaquan/resource/data/대화맥락추론_test_correct.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# CSV 파일 읽기
csv_data = pd.read_csv('/root/gemmaquan/resource/data/Wittgenstein\'s annotation - 대화맥락추론_test_annotation.csv')

# CSV 데이터를 사전으로 변환 (id를 키로 사용)
csv_dict = csv_data.set_index('id')['output'].to_dict()

# JSON 데이터 업데이트
for item in data:
    item_id = item['id']
    if item_id in csv_dict:
        item['output'] = csv_dict[item_id]

# 업데이트된 JSON 데이터를 파일에 쓰기
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 업데이트되었습니다.")

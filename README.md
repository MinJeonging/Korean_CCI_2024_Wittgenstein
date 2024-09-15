# Korean_CCI_2024_Wittgenstein
[국립국어원] 2024년 인공지능(AI)말평 과제 경진대회 참가팀 '비트겐슈타인' 리포지토리

본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '대화 맥락 추론'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습 및 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.  

|Model|Accuracy(%)|
|:---|---:|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (without SFT)|69.7|
|Our Model (with SFT)|95.7|

## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py (보완 후 업로드 예정)

# 학습에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
{
    "id": "nikluge-2024-대화 맥락 추론-train-000001",
    "input": {
        "conversation": [
            {
                "speaker": 2,
                "utterance": "진짜 신의 한수",
                "utterance_id": "MDRW2100003410.1.1"
            },
            {
                "speaker": 1,
                "utterance": "이사하자마자 비 많이 와서 베란다 물 많이 새는 거 알았잖아",
                "utterance_id": "MDRW2100003410.1.2"
            },
            {
                "speaker": 2,
                "utterance": "글치 계속 해떴으면 몰랐겠지",
                "utterance_id": "MDRW2100003410.1.3"
            },
            ...
            ...
            ...
        ],
        "reference_id": [
            "MDRW2100003410.1.11"
        ],
        "category": "원인",
        "inference_1": "화자2가 사는 곳 근처에서 베란다 보수 공사가 진행되고 있다.",
        "inference_2": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 진행되고 있다.",
        "inference_3": "화자2가 사는 곳 근처에서 싱크홀 보수 공사가 중단되었다."
    },
    "output": "inference_2" # The Correct answer is inference_2
}
```

## 실행 방법 (How to Run)

### Python 3.10 환경 구축 (Setup Python 3.10 Environment)
리눅스(Vessl) 환경에서 Python 3.10을 설치하고 가상환경을 설정하려면 아래 명령어를 사용하세요:

#### 리눅스(Vessl)에서 Python 3.10 설치 및 가상환경 설정
1. 패키지 목록 업데이트:
    ```bash
    apt update
    ```

2. 필요한 소프트웨어 설치:
    ```bash
    apt install software-properties-common -y
    ```

3. PPA 저장소 추가:
    ```bash
    add-apt-repository ppa:deadsnakes/ppa -y
    ```

4. Python 3.10 설치:
    ```bash
    apt install python3.10
    ```

5. Python 3.10 venv 모듈 설치:
    ```bash
    apt install python3.10-venv
    ```

6. 가상환경 생성:
    ```bash
    python3.10 -m venv venv3.10
    ```

7. 가상환경 활성화:
    ```bash
    source venv3.10/bin/activate
    ```

8. 필요 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```

### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=1,3 python -m run.train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 5 \
    --lr 2e-5 \
    --warmup_steps 20
```

### 추론 (Inference)
```
python -m run.test \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```

## Reference
huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  

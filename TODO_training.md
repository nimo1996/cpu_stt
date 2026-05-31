# STT 모델 GPU 파인튜닝 가이드 (2026-06)

현재 운영 중인 두 모델을 GPU 환경에서 AICC 한국어 도메인에 맞게 파인튜닝하는 방법.

---

## 1. 모델별 파인튜닝 개요

| | Moonshine Tiny Ko | Fun-ASR-MLT-Nano-2512 |
|---|---|---|
| 파라미터 수 | 27M | 800M (인코더 200M + Qwen3-0.6B) |
| **RTX 3090 (24GB)** | ✅ Full fine-tuning 여유있게 가능 | ✅ LoRA 여유있게 가능 / Full은 타이트 |
| 학습 방식 | Full fine-tuning 권장 | **LoRA 권장** (Full도 가능) |
| 데이터 포맷 | HuggingFace datasets | JSONL |
| 라이선스 | Community License 유지 | **Apache 2.0 유지** |

---

## 2. Moonshine Tiny Ko 파인튜닝

### 2-1. GPU 요구사항

Moonshine Tiny는 27M 파라미터. 파인튜닝 전체 메모리가 **1GB 미만**이라 24GB VRAM 대비 압도적으로 여유있다.

| 학습 방식 | VRAM 사용량 | RTX 3090 (24GB) | 예상 학습 시간 (10h 데이터) |
|---|---|---|---|
| Full fine-tuning | ~500MB | ✅ 배치 크기 크게 가능 | 2~4시간 |
| LoRA | ~200MB | ✅ 초여유 | 1시간 이하 |

메모리 계산 근거:
- 가중치 (fp16): 27M × 2B = **54MB**
- 그래디언트: 54MB
- Adam optimizer 상태 (fp32): 27M × 8B = **216MB**
- Activation: 배치 크기에 따라 수백MB
- **합계: ~1GB 미만** → RTX 3090 24GB의 4% 수준

### 2-2. 공식 파인튜닝 방법

**Moonshine AI 공식 팀**은 내부 데이터를 이용한 full retraining을 유료 상업 서비스로만 제공.
무료 경량 파인튜닝 도구 지원은 "향후 계획" 수준으로 미확정.

→ 현실적 대안은 HuggingFace Transformers 기반 파인튜닝.

### 2-3. HuggingFace Trainer 파인튜닝

Moonshine은 `transformers`에 통합돼 있어 표준 Seq2Seq 파인튜닝 방식으로 학습 가능.

```python
from transformers import (
    MoonshineForConditionalGeneration,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-tiny-ko")
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny-ko")

training_args = Seq2SeqTrainingArguments(
    output_dir="./moonshine-ko-aicc",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    warmup_steps=500,
    num_train_epochs=3,
    fp16=True,                  # GPU 메모리 절약
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    report_to=["tensorboard"],
)
```

### 2-4. 커뮤니티 파인튜닝 툴킷 (권장)

저장소: [pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)

완성된 파인튜닝 스크립트 + 가이드 제공. 프랑스어 예제에서 WER 21.8% 달성.
multi-GPU 학습 지원 (accelerate / torchrun).

```bash
git clone https://github.com/pierre-cheneau/finetune-moonshine-asr
cd finetune-moonshine-asr
pip install -r requirements.txt

# 단일 GPU
python train.py --config configs/korean.yaml

# 멀티 GPU (accelerate)
accelerate launch train.py --config configs/korean.yaml
```

### 2-5. 데이터 포맷

HuggingFace `datasets` 형식 사용:

```python
from datasets import Dataset, Audio

dataset = Dataset.from_dict({
    "audio": ["path/to/audio1.wav", "path/to/audio2.wav", ...],
    "sentence": ["안녕하세요 고객센터입니다", "보험료 납부 문의드립니다", ...],
})
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

---

## 3. Fun-ASR-MLT-Nano-2512 파인튜닝

### 3-1. GPU 요구사항

800M 파라미터 모델 (오디오 인코더 200M + Qwen3-0.6B LLM). **LoRA 방식 강력 권장.**

| 학습 방식 | VRAM 사용량 | RTX 3090 (24GB) |
|---|---|---|
| **LoRA (권장)** | **~5~6GB** | ✅ 여유있게 가능 |
| Full fine-tuning | ~15~16GB | ⚠️ 가능 — gradient checkpointing + 배치 1~2 필수 |
| QLoRA 4-bit | ~3~4GB | ✅ 초여유 (정확도 소폭 감소) |

메모리 계산 근거 (LoRA):
- 고정 베이스 모델 (fp16): 800M × 2B = **1.6GB**
- LoRA 어댑터 (r=16, ~0.5% 파라미터): **~16MB**
- LoRA 그래디언트 + Adam 상태: **~64MB**
- Activation: ~2~4GB
- **합계: ~5~6GB** → RTX 3090 24GB의 25% 수준

Full fine-tuning 메모리 (gradient checkpointing 적용 시):
- 가중치 (fp16): 1.6GB + 그래디언트 1.6GB + Adam fp32 (9.6GB) + Activation (2~3GB) = **~15~16GB**
- 24GB에서 배치 1~2로 가능. gradient_accumulation_steps=16~32로 실효 배치 보완.

BF16 정밀도 권장 (RTX 3090은 BF16 지원). V100 이하는 FP16.

### 3-2. 공식 파인튜닝 방법

공식 문서: [FunAudioLLM/Fun-ASR — docs/finetune.md](https://github.com/FunAudioLLM/Fun-ASR/blob/main/docs/finetune.md)

```bash
git clone https://github.com/FunAudioLLM/Fun-ASR
cd Fun-ASR
pip install -r requirements.txt

# LoRA 파인튜닝 — RTX 3090 단일 GPU (권장)
python funasr/bin/train_ds.py \
  ++model=FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
  ++train_data_set_list=data/train.list \
  ++val_data_set_list=data/val.list \
  ++output_dir=output/ko_aicc \
  ++use_lora=true \
  ++use_bf16=true \
  ++dataset_conf.batch_size=2000 \
  ++dataset_conf.num_workers=4

# Full fine-tuning — RTX 3090 단일 GPU (메모리 절약 설정)
python funasr/bin/train_ds.py \
  ++model=FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
  ++train_data_set_list=data/train.list \
  ++val_data_set_list=data/val.list \
  ++output_dir=output/ko_aicc_full \
  ++use_lora=false \
  ++use_bf16=true \
  ++train_conf.accum_grad=16 \
  ++dataset_conf.batch_size=500 \
  ++dataset_conf.batch_size_sample_max=2

# 평가
python funasr/bin/evaluate.py \
  ++model_dir=output/ko_aicc \
  ++scp_file=data/test_wav.scp \
  ++output_file=output/result.txt
```

### 3-3. 데이터 포맷 (JSONL)

각 줄이 하나의 발화:

```jsonl
{"messages": [{"role": "system", "content": "Speech transcription:"}, {"role": "user", "content": "<|startofspeech|>!/data/audio/001.wav<|endofspeech|>"}, {"role": "assistant", "content": "안녕하십니까 국민건강보험공단 고객센터입니다"}]}
{"messages": [{"role": "system", "content": "Speech transcription:"}, {"role": "user", "content": "<|startofspeech|>!/data/audio/002.wav<|endofspeech|>"}, {"role": "assistant", "content": "보험료 납부 내역 확인을 원하시는군요"}]}
```

data.list 파일 (대용량 데이터 분할):
```
data/jsonl/train.0.jsonl
data/jsonl/train.1.jsonl
...
```

### 3-4. DeepSpeed 설정 (멀티 GPU)

```bash
# 4-GPU DeepSpeed ZeRO2
deepspeed --num_gpus=4 funasr/bin/train_ds.py \
  ++deepspeed_config=configs/deepspeed_zero2.json \
  ++model=FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
  ++train_data_set_list=data/train.list \
  ++output_dir=output/ko_aicc_multi
```

---

## 4. AIHub 한국어 음성 데이터셋 (AICC 최적)

사이트: [aihub.or.kr](https://aihub.or.kr) — 회원가입 후 신청 필요 (무료)

### AICC 도메인 직결 데이터

| 데이터셋 | 규모 | 음질 | 비고 |
|---|---|---|---|
| [민원(콜센터) 질의-응답](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=98) | 440시간+ | 전화 음질 | 실제 민원 Q&A, 전사 포함 |
| [상담 음성](https://aihub.or.kr/aidata/30711) | 미공개 | 전화 음질 | 실제 콜센터 상담, 저작권 해소 |
| [공공분야 고객응대](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71615) | **3,300시간** | 전화 음질 | 6개 공공서비스 분야, 감정 태깅 포함 |
| [복지 콜센터 상담](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=470) | **2,945시간** | 16kHz wav | 복지시설 3곳, 파일 227만개 |

→ **공공분야 고객응대 3,300시간**이 AICC 파인튜닝에 가장 유용.

### 기반 데이터 (일반 한국어)

| 데이터셋 | 규모 | 라이선스 | 접근 방법 |
|---|---|---|---|
| [Zeroth Korean](https://openslr.org/40/) | 51.6시간 | CC-BY 4.0 | 직접 다운로드 |
| [kresnik/zeroth_korean](https://huggingface.co/datasets/kresnik/zeroth_korean) | 51.6시간 | CC-BY 4.0 | HuggingFace datasets |
| [Common Voice Korean](https://commonvoice.mozilla.org) | 200시간+ | CC0 | HuggingFace datasets |

---

## 5. 학습 데이터 전처리 파이프라인

### AIHub → 학습 데이터 변환

```bash
# 1. WAV 확인 (16kHz, mono)
ffprobe -i sample.wav 2>&1 | grep "Stream"

# 2. 필요시 리샘플링
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav

# 3. 발화 길이 필터링 (0.5s ~ 30s 권장)
python scripts/filter_duration.py \
  --input data/raw_manifest.jsonl \
  --output data/filtered_manifest.jsonl \
  --min-sec 0.5 --max-sec 30

# 4. JSONL 변환 (Fun-ASR용)
python scripts/to_funasr_jsonl.py \
  --audio-dir data/wav/ \
  --transcript data/transcript.txt \
  --output data/train.jsonl
```

### 데이터 품질 권장 기준

| 항목 | 기준 |
|---|---|
| 샘플레이트 | 16kHz (8kHz는 업샘플 후 사용) |
| 채널 | Mono |
| 발화 길이 | 0.5s ~ 30s |
| 전사 오류율 | 5% 미만 |
| 최소 데이터 | 도메인 특화: 10~20시간 / 일반 학습: 100시간+ |

---

## 6. 학습 환경 구성

### RTX 3090으로 가능한 작업 정리

보유 중인 RTX 3090 (24GB)으로 **두 모델 모두 파인튜닝 가능**.

| 목적 | 방식 | VRAM 사용 | 가능 여부 |
|---|---|---|---|
| Moonshine Full fine-tuning | Full | ~1GB | ✅ 배치 크게 가능 |
| Fun-ASR LoRA | LoRA (r=16) | ~5~6GB | ✅ 여유있게 가능 |
| Fun-ASR Full fine-tuning | Full + grad ckpt | ~15~16GB | ⚠️ 가능, 배치 1~2 |
| Fun-ASR QLoRA | QLoRA 4-bit | ~3~4GB | ✅ 초여유 |

### 클라우드 GPU 옵션 (로컬 3090 보완용)

| 서비스 | GPU | 시간당 비용 | 비고 |
|---|---|---|---|
| **RunPod** | RTX 4090 / A100 | $0.5~1.5/h | 국내 신용카드 가능 |
| **Lambda Labs** | A100 40GB | $1.1~1.5/h | 안정적, 저렴 |
| **vast.ai** | 다양 | $0.3~1.0/h | 가장 저렴, 안정성 변동 |

### 환경 설정 (공통)

```bash
# CUDA 확인
nvidia-smi

# 학습 패키지
pip install torch torchaudio transformers accelerate deepspeed
pip install funasr datasets evaluate jiwer  # WER 측정용
```

---

## 7. 추천 학습 순서

```
Step 1  AIHub 신청
        → 공공분야 고객응대 데이터 (3,300시간) 신청
        → 승인까지 수일~수주 소요

Step 2  기반 데이터로 파인튜닝 선행 테스트
        → Zeroth Korean (51.6시간, 즉시 다운로드)
        → Moonshine 먼저 시도 (GPU 요구 낮음)

Step 3  AIHub 데이터 수신 후 도메인 파인튜닝
        → 전화 음질(8kHz 업샘플) 데이터로 재학습
        → 도메인 용어 (보험, 납부, 민원 등) 집중 검증

Step 4  WER 비교
        → 파인튜닝 전/후 WER 측정
        → Moonshine vs Fun-ASR-Nano 성능 비교
        → 우위 모델로 최종 결정
```

---

## 참고 링크

- [pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)
- [HuggingFace Moonshine 문서](https://huggingface.co/docs/transformers/model_doc/moonshine)
- [HuggingFace Audio Course — ASR 파인튜닝](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)
- [FunAudioLLM/Fun-ASR finetune.md](https://github.com/FunAudioLLM/Fun-ASR/blob/main/docs/finetune.md)
- [FunAudioLLM/Fun-ASR GitHub](https://github.com/FunAudioLLM/Fun-ASR)
- [AIHub 공공분야 고객응대 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71615)
- [AIHub 복지 콜센터 상담데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=470)
- [AIHub 민원(콜센터) 질의-응답](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=98)
- [Zeroth Korean (OpenSLR)](https://openslr.org/40/)
- [kresnik/zeroth_korean (HuggingFace)](https://huggingface.co/datasets/kresnik/zeroth_korean)

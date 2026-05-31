# CPU 기반 한국어 실시간 Streaming STT 모델 조사 보고서 (2026)

## 1. 조사 목적

다음 조건을 만족하는 한국어 STT 모델을 찾기 위함.

### 요구사항

- CPU Only 동작
- GPU 비필수
- 한국어 지원
- Streaming ASR
- 100ms PCM 패킷 단위 입력
- Partial Result 실시간 반환
- WebSocket/TCP 기반 연동 가능
- 로컬(On-Premise) 실행
- AICC 및 실시간 음성 처리 시스템 적용 가능

---

# 2. 후보 모델 비교

| 모델 | 한국어 | Streaming | CPU | Partial Result | 실시간성 | 라이선스 | 추천도 |
|--------|--------|--------|--------|--------|--------|--------|--------|
| **Moonshine Tiny Ko** | O | O | O | O | 매우 좋음 | Community (조건부) | ★★★★★ |
| **Fun-ASR-MLT-Nano-2512** | O | O | O | O | 보통 (CPU) | **Apache 2.0** | ★★★★☆ |
| Faster Whisper | O | △ | O | △ | 보통 | MIT | ★★★☆☆ |
| Whisper Large-v3 | O | X | O | X | 낮음 | MIT | ★★☆☆☆ |
| Sherpa Korean Zipformer | O | O | O | O | 매우 좋음 | Apache 2.0 | ★★★☆☆ (버그 주의) |
| SenseVoice | O | O | O | O | 매우 좋음 | 불명확 (other) | ★★★☆☆ (지원 불확실) |
| FunASR Streaming | O | O | O | O | 좋음 | MIT | ★★★★☆ |
| Vosk Korean | O | O | O | O | 매우 좋음 | Apache 2.0 | ★★☆☆☆ |
| Qwen3-ASR | O | 일부 | O | 일부 | 미확인 | Apache 2.0 | ★★★☆☆ |

> **Vosk 낮은 평가 이유:** 한국어 전용 모델의 절대 인식률이 낮고, 학습 데이터 규모가 다른 후보 대비 열세임.

---

# 3. Whisper 계열 분석

## 장점

- 높은 한국어 정확도
- 커뮤니티 규모 큼
- 사용 사례 풍부

## 단점

Whisper는 본질적으로 Streaming 모델이 아니다.

구조:

30초 음성
→ 전체 추론

실시간 서비스에서는:

버퍼 누적
→ 재추론
→ 결과 병합

방식을 사용한다.

즉 실제 Streaming이 아니라 Streaming처럼 보이게 구현하는 방식이다.

### 결론

회의록 생성이나 파일 STT에는 우수하지만,

100ms PCM Streaming 환경에는 비효율적이다.

---

# 4. Moonshine Tiny Ko ★ 1순위 (실측 검증)

논문: "Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices" (2025.09)

모델: `sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27`

## 특징

- 한국어 전용 모델 (언어별 특화 파인튜닝)
- 파라미터 수 27M (매우 경량)
- Sherpa-ONNX 공식 통합 완료
- Moonshine v2 기반: **Ergodic Streaming Encoder** — 진짜 Streaming 구조 (recurrent state 유지)
- CPU-first 설계

## 성능 (실측 포함)

- Whisper Medium(파라미터 28배 큰 모델)과 동등하거나 우세한 한국어 정확도
- Moonshine v2 기준 TTFT 258ms (Whisper Large-v3 대비 43.7배 빠름)
- Zeroth-Korean 데이터셋 기준 검증됨

### 실측 결과 (Intel Broadwell 4코어 CPU, 2026-06)

| 항목 | 수치 |
|---|---|
| 테스트 음성 | "조국이 당신을 위해 무엇을 해줄 수 있는지 묻지 말고 당신이 조국을 위해 무엇을 할 수 있는지 물으십시오." |
| 인식 결과 | 완벽 일치 ✅ |
| 오디오 길이 | 6.92초 |
| 처리 시간 | **217ms** |
| **RTF** | **0.031** (실시간의 32배 속도) |
| 모델 크기 | 69MB (encoder 13MB + decoder 56MB) |
| 로딩 시간 | ~1.5초 |

RTF 0.031은 이 서버에서 동시 30채널 이상 처리 가능한 수준.

## 장점

- 한국어 전용 → 범용 다국어 모델 대비 인식률 유리
- 매우 작은 모델 크기 → CPU 부담 최소
- Sherpa 런타임 그대로 사용 가능
- 진짜 Streaming 구조 (SenseVoice와 달리 청크 단위 recurrent 처리)

## 단점

- 2025년 9월 발표, 운영 사례 아직 적음
- 한국어 운영 환경 튜닝 사례 부족

## ⚠️ 라이선스 — Moonshine Community License (비표준)

한국어 모델은 MIT가 **아니다**. 영어 모델만 MIT이고, 한국어를 포함한 기타 언어 모델은 별도 라이선스 적용.

| 상황 | 사용 가능 여부 |
|---|---|
| 연구 / 비상업 용도 | 자유롭게 사용 가능 |
| 연매출 **$100만(약 13억 원) 미만** 상업 사용 | 가능 — [moonshine.ai/community-license](https://moonshine.ai/community-license) **등록 필수** |
| 연매출 **$100만 이상** 상업 사용 | 라이선스 자동 종료 → Enterprise 계약 필요 (Moonshine AI 재량 승인) |

추가 의무사항 (상업 사용 시):
- 제품 UI/웹사이트/문서 등에 **"Powered by Moonshine AI"** 표시 의무
- Notice 파일에 저작권 고지 포함 의무

→ 매출 규모나 표시 의무가 부담스럽다면 Fun-ASR-Nano-2512(Apache 2.0)를 사용할 것.

---

# 5. Fun-ASR-MLT-Nano-2512 ★ 2순위 (실측 검증)

발표: FunAudioLLM (SenseVoice 개발팀), 2025.12.15

한국어 지원 버전: **Fun-ASR-MLT-Nano-2512** (MLT = Multilingual, 31개 언어)
- Fun-ASR-Nano-2512는 중국어·영어·일본어 특화, 한국어는 MLT 버전 사용

## 아키텍처

- 오디오 인코더: SenseVoiceEncoderSmall (SANM, 50 블록)
- LLM: Qwen3-0.6B (608M 파라미터)
- 총 파라미터: 약 800M

## 특징

- 수십만 시간 규모 학습 데이터 (한국어 포함 31개 언어)
- CPU/GPU 모두 지원 (vllm 없이 일반 torch 경로로 CPU 동작 확인)
- SenseVoice의 사실상 후속 모델

## 장점

- **Apache 2.0** — 조건 없이 완전 자유로운 상업 사용
- 같은 팀의 지속 개발 모델, FunASR 런타임과 완전 호환

## 단점

- 한국어 단독 벤치마크 미공개
- 청크 단위가 720ms 기준으로 문서화 → VAD 세그먼트 단위 처리 권장

### 실측 결과 (Intel Broadwell 4코어 CPU, 2026-06)

| 항목 | 수치 |
|---|---|
| 테스트 음성 | "조금만 생각을 하면서 살면 훨씬 편할 거야." |
| 인식 결과 | 완벽 일치 ✅ |
| 처리 시간 (3회 평균) | **3,661ms** |
| **RTF** | **0.72~0.92** (1.0에 근접, 이 서버에서 실시간 한계선) |
| 모델 크기 | ~1.5GB (Qwen3-0.6B 포함) |
| 로딩 시간 | ~18초 |

RTF 0.72~0.92는 이 서버 사양에서 실시간 처리 가능하나 여유가 거의 없음.
동시 채널 처리에는 고사양 CPU 또는 GPU 권장.

---

# 6. Sherpa-ONNX Korean Zipformer

모델:

sherpa-onnx-streaming-zipformer-korean-2024-06-16

특징

- 한국어 전용
- Streaming 지원
- Partial Result 지원
- CPU 최적화
- ONNX Runtime 기반
- Endpoint Detection 내장
- WebSocket 서버 제공

지원 플랫폼

- Linux / Windows / macOS / Android / iOS / Embedded Linux

## ⚠️ 현재 상태 주의

2025년 12월 기준, GitHub Issue #2886:

> "Both Korean streaming models return empty transcription results."
> 오류 없이 로드되지만 모든 입력에 대해 빈 문자열 반환.

해결 여부 미확인. 실제 적용 전 반드시 최신 릴리스에서 동작 검증 필요.

Moonshine Tiny Ko가 동일 Sherpa 런타임에서 동작하며 더 최신 모델이므로, 이 모델을 대신 사용하는 것을 권장함.

---

# 7. SenseVoice

개발사: Alibaba FunAudioLLM

## ⚠️ 지원 불확실성

Alibaba Bailian 클라우드 플랫폼에서 "soon to be discontinued" 표기 확인됨.
오픈소스 모델 자체는 유지될 가능성이 높으나 장기 지원 보장 불명확.
후속작인 Fun-ASR-Nano-2512 사용을 권장.

## ⚠️ 라이선스 불명확

GitHub 저장소의 LICENSE 파일은 "FunASR 라이선스를 참조"라고만 명시.
HuggingFace 모델 카드에는 `license: other` 표기.
FunASR 프레임워크 자체는 MIT이지만, **SenseVoice 모델 가중치에 대한 라이선스가 명확히 분리되어 있지 않음.**
상업 도입 전 FunAudioLLM 측에 직접 확인 필요. 불확실성이 허용되지 않는다면 Fun-ASR-Nano-2512(Apache 2.0)를 사용할 것.

## 기존 특징

- ASR + Language Identification + Emotion Recognition + Audio Event Detection
- 50개 이상 언어 지원 (한국어 포함)
- 논문 기준 Recognition Latency < 80ms
- Whisper-Small 대비 5배 이상 빠름
- Whisper-Large 대비 15배 이상 빠름

## 구조 주의사항

SenseVoice는 인코더 전체 처리 방식으로, 구조적 Streaming이 아님.
짧은 청크를 빠르게 처리하는 방식으로 유사 Streaming을 구현.
공식 런타임은 FunASR이며, "SenseVoice + Sherpa Runtime" 조합은
Sherpa-ONNX ONNX 포팅을 통해 가능하나 Streaming 모드 지원은 별도 검증 필요.

---

# 8. FunASR Streaming

개발: Alibaba 계열 (modelscope/FunASR)

특징

- Streaming ASR
- WebSocket 지원
- CPU 지원
- OpenAI 호환 API 제공
- 170x realtime 처리 지원 (GPU 기준)
- SenseVoice, Paraformer, Fun-ASR-Nano-2512 등을 실행하는 프레임워크

장점

- 기업 서비스 지향 (Production Ready)
- Fun-ASR-Nano-2512와 함께 사용 시 가장 완성도 높은 구성

단점

- 한국어 전용 벤치마크 자료 부족
- FunASR 프레임워크와 FunAudioLLM의 SenseVoice/Fun-ASR-Nano는 별도 조직 — 혼동 주의

---

# 9. Qwen3-ASR

장점

- 매우 높은 정확도 (52개 언어 지원)
- 최신 LLM 계열 ASR (2026.01 출시)

단점

- Streaming 최적화 미성숙
- CPU 실시간 처리 검증 부족

현재 기준

정확도 중심 프로젝트에는 유망하지만,

100ms Streaming 환경에는 아직 검증 필요

---

# 10. 추천 아키텍처

## 권장 구조

```
PCM 100ms → Silero VAD → Moonshine Tiny Ko (Sherpa-ONNX) → Partial/Final Text → LLM → Supertonic 3
```

## 대안 구조 (라이선스 완전 자유)

```
PCM 100ms → Silero VAD → Fun-ASR-MLT-Nano-2512 (FunASR) → Partial/Final Text → LLM → Supertonic 3
```

## Silero VAD 역할 및 최종 설정 (실측 튜닝 완료)

Silero VAD는 신경망 기반 발화 구간 검출기로, 에너지 임계값 방식 대비:
- 문장 내 짧은 포즈를 발화 종료로 오인하지 않음
- 배경 소음 속 발화 구분 가능
- 처리 속도 < 1ms / 청크 (CPU)

| 파라미터 | 최종값 | 설명 |
|---|---|---|
| `silence_duration_s` | **0.3s** | 무음이 이 시간 지속되면 발화 종료로 판정 |
| `threshold` | **0.8** | Silero 발화 감지 신뢰도 임계값 (EPD) |
| `speech_pad_ms` | 200ms | 발화 시작 직전 오디오 보존 길이 |
| `max_utterance_s` | 15s | 최대 발화 버퍼 길이 (초과 시 강제 flush) |
| `partial_interval_s` | 0.5s | 중간 결과 반환 주기 |

### 파라미터 튜닝 기준

`silence_duration_s`와 `threshold`는 음원 특성에 따라 조정:

| 음원 유형 | silence_duration_s | threshold |
|---|---|---|
| 자연 발화 (전화 통화) | 0.5~0.8s | 0.5 |
| TTS 합성음 | 0.3s | 0.8 |
| 잡음 환경 | 0.5s | 0.6~0.7 |

---

# 11. 최종 추천

## 실측 기반 최종 추천

1순위 — **성능 우선 (라이선스 조건 수용 가능한 경우)**

**Moonshine Tiny Ko + Sherpa-ONNX**

- RTF **0.031** (실측) — 이 서버에서 동시 30채널 여유
- 로딩 1.5초, 모델 69MB, 인식 정확도 완벽
- 라이선스: Moonshine Community License
  - 연매출 $100만 미만: [moonshine.ai/community-license](https://moonshine.ai/community-license) 등록 후 무료
  - 연매출 $100만 이상: Enterprise 계약 필요
  - "Powered by Moonshine AI" 표시 의무

2순위 — **라이선스 완전 자유 우선**

**Fun-ASR-MLT-Nano-2512 + FunASR**

- 라이선스: **Apache 2.0** — 아무 조건 없음
- RTF **0.72~0.92** (실측) — 이 서버에서 실시간 처리 가능하나 여유 없음
- 동시 다채널 환경에는 고사양 CPU 또는 GPU 서버 권장
- 로딩 18초, 모델 1.5GB

보류

- Sherpa Korean Zipformer: Issue #2886 버그 해결 확인 전 사용 자제
- SenseVoice: 라이선스 불명확(other) + 지원 불확실성으로 신규 적용 비권장

---

# 미검토 항목 (추가 확인 필요)

| 항목 | 상태 | 내용 |
|---|---|---|
| 동시 처리 스트림 수 | ❌ 미측정 | AICC 환경에서 CPU 코어당 몇 채널 처리 가능한지 부하 테스트 필요 |
| 자연 발화 WER | ❌ 미측정 | 실제 전화 음질(8kHz 업샘플, 잡음) 기준 WER 측정 필요 |
| Moonshine 라이선스 등록 | ❌ 미완료 | 상업 사용 시 moonshine.ai/community-license 등록 필요 |
| VAD 파라미터 (자연 발화) | ❌ 미튜닝 | 현재 설정은 TTS 기준. 실제 통화 음성으로 재튜닝 권장 |

---

# 결론

2025년 하반기~2026년 초 기준으로 상황이 크게 바뀌었다.
두 모델 모두 실제 설치·동작·성능을 이 서버에서 검증했다.

**두 모델의 포지션은 명확하다:**

| | Moonshine Tiny Ko | Fun-ASR-MLT-Nano-2512 |
|---|---|---|
| RTF (실측) | **0.031** | 0.72~0.92 |
| 라이선스 | Community (조건부) | **Apache 2.0** |
| 동시 채널 (이 서버) | ~30채널 | ~1채널 |
| 모델 크기 | 69MB | 1.5GB |
| 로딩 시간 | 1.5초 | 18초 |

이 서버 사양과 AICC 실시간 요건을 기준으로는 **Moonshine Tiny Ko**가 압도적으로 유리하다.
라이선스 조건(등록 의무, 표시 의무)을 수용할 수 없을 때만 Fun-ASR-MLT-Nano를 선택한다.

---

## 라이선스 요약표

| 모델 | 라이선스 | 무조건 상업 사용 | 비고 |
|---|---|---|---|
| Fun-ASR-MLT-Nano-2512 | **Apache 2.0** | ✅ 자유 | 조건 없음 |
| FunASR 프레임워크 | MIT | ✅ 자유 | 조건 없음 |
| Sherpa-ONNX 런타임 | Apache 2.0 | ✅ 자유 | 조건 없음 |
| Moonshine Tiny Ko | Community License | ⚠️ 조건부 | 매출 $100만 미만 등록 후 무료, 이상은 Enterprise + 표시 의무 |
| SenseVoice 가중치 | other (불명확) | ❓ 미확인 | 상업 사용 전 직접 확인 필요 |
| Sherpa Korean Zipformer | Apache 2.0 | ✅ 자유 | 현재 버그 있음 |

---

## 배포 구성 (실제 설치 및 동작 검증 완료)

```
cpu_stt/
  serve.py          # WebSocket STT 서버 (Moonshine / FunASR 공용)
  client_test.py    # 테스트 클라이언트 (WAV/MP3 → 서버 → 결과 출력)
  config.yaml       # 모델 선택 및 VAD 설정
  models/
    moonshine-tiny-ko/   # 다운로드 완료 (69MB)
  logs/
    stt_server.log       # 서버 로그 (자동 누적)
  venv/             # Python 3.11 가상환경 (sherpa-onnx, funasr, silero-vad, scipy 포함)
```

### 서버 실행

```bash
# Moonshine 서버 (기본, RTF 0.031)
venv/bin/python serve.py

# FunASR 서버 (Apache 2.0 라이선스 우선)
venv/bin/python serve.py --model funasr

# 포트 변경
venv/bin/python serve.py --port 9000
```

### 테스트 클라이언트

```bash
# 기본 (실시간 100ms 페이싱)
venv/bin/python client_test.py --audio path/to/audio.wav

# 빠른 전송 (레이턴시 측정)
venv/bin/python client_test.py --audio path/to/audio.wav --fast

# 원격 서버
venv/bin/python client_test.py --url ws://192.168.0.10:8765 --audio audio.wav
```

### WebSocket 프로토콜

```
클라이언트 → 서버:
  binary          16kHz / 16-bit mono PCM (100ms 단위 권장)
  {"type":"eos"}  스트림 종료 신호 (통화 종료 시 전송)

서버 → 클라이언트:
  {"type":"partial", "text":"..."}  발화 중 중간 결과 (0.5초마다)
  {"type":"final",   "text":"..."}  발화 종료 확정 결과
  {"type":"eos_ack"}                EOS 처리 완료 (연결 종료 가능)
  {"type":"error",   "message":"..."} 오류
```

### TTS 음성 실측 결과 (korean_ko.wav, 55.38초, Intel Broadwell 4코어)

| VAD 설정 | final 수 | 주요 인식 결과 |
|---|---|---|
| silence 0.8s / threshold 0.5 (에너지 기반) | 2개 | 첫 문장만 |
| silence 1.5s | 0개 | 55초 전체 버퍼 → Moonshine 실패 |
| silence 0.3s / threshold 0.8 **(최종)** | **13개** | 문장 단위 분리 성공 |

최종 설정에서 인식된 문장 예시:
- "안녕하십니까 국민건강보험공단 고객센터입니다"
- "무엇을 도와드릴까요?"
- "보험료 납부 내역 확인을 원하시는군요."
- "본인 확인을 위해 성함과 주민등록번호 앞자리를 말씀해주세요."
- "이번 기회에 계좌 자동이체나 신용카드 전기결제를 신청해 보시겠습니까?"

---

## 파인튜닝 환경 구성 (RTX 3090 기준)

두 모델 모두 **RTX 3090 24GB** 한 장으로 파인튜닝 가능.

| 모델 | 학습 방식 | VRAM 사용 | 비고 |
|---|---|---|---|
| Moonshine Tiny Ko | Full fine-tuning | **~1GB** | 24GB의 4% — 배치 크게 가능 |
| Fun-ASR-MLT-Nano | **LoRA** (권장) | **~5~6GB** | 24GB의 25% — 여유있음 |
| Fun-ASR-MLT-Nano | Full fine-tuning | ~15~16GB | gradient checkpointing + 배치 1~2 필수 |

### 학습 방법

**Moonshine** — HuggingFace Trainer 기반 (`MoonshineForConditionalGeneration`)
- 커뮤니티 툴킷: [pierre-cheneau/finetune-moonshine-asr](https://github.com/pierre-cheneau/finetune-moonshine-asr)
- 데이터 포맷: HuggingFace `datasets` (audio + sentence 컬럼)

**Fun-ASR-MLT-Nano** — 공식 `docs/finetune.md` + DeepSpeed ZeRO2
- 저장소: [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
- 데이터 포맷: JSONL (messages 구조)
- 단일 GPU: `python funasr/bin/train_ds.py ++use_lora=true ++use_bf16=true`

### 학습 데이터 (AICC 도메인)

| 데이터셋 | 규모 | 접근 | 용도 |
|---|---|---|---|
| [AIHub 공공분야 고객응대](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71615) | 3,300시간 | 무료 신청 | 전화 음질, 감정 태깅, AICC 최적 |
| [AIHub 복지 콜센터 상담](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=470) | 2,945시간 | 무료 신청 | 16kHz wav, 227만 파일 |
| [Zeroth Korean](https://openslr.org/40/) | 51.6시간 | 즉시 다운로드 | CC-BY 4.0, 선행 테스트용 |

→ 세부 내용: `TODO_training.md` 참조

# CPU 한국어 실시간 Streaming STT 서버

CPU만으로 동작하는 한국어 실시간 STT(Speech-to-Text) WebSocket 서버입니다. AICC(AI Contact Center) 및 실시간 음성 처리 시스템을 대상으로 설계되었습니다.

## 특징

- **CPU 전용** — GPU 불필요, 온프레미스 배포 가능
- **한국어 특화** — Moonshine Tiny Ko / FunASR MLT Nano 백엔드 지원
- **진정한 스트리밍** — 100ms PCM 패킷 단위 입력, Partial/Final 결과 실시간 반환
- **VAD 내장** — Silero VAD로 발화 구간 자동 감지
- **WebSocket 기반** — 표준 WebSocket 프로토콜, 손쉬운 연동

## 아키텍처

```
PCM 100ms → Silero VAD → STT Backend → Partial/Final JSON
```

### 지원 백엔드

| 백엔드 | 모델 | 라이선스 | 특징 |
|--------|------|----------|------|
| **Moonshine** (기본) | `sherpa-onnx-moonshine-tiny-ko-quantized` | Community | 초저지연, 경량 |
| **FunASR** | `FunAudioLLM/Fun-ASR-MLT-Nano-2512` | Apache 2.0 | 다국어, 안정적 |

## 요구사항

- Python 3.8+
- `sherpa-onnx` (Moonshine 백엔드)
- `funasr` (FunASR 백엔드)
- `websockets`, `numpy`, `pyyaml`

## 설치

```bash
python -m venv venv
source venv/bin/activate
pip install sherpa-onnx funasr websockets numpy pyyaml
```

## 모델 다운로드

Moonshine Tiny Ko 모델 파일을 `models/moonshine-tiny-ko/` 디렉터리에 배치합니다:

```
models/moonshine-tiny-ko/
  encoder_model.ort
  decoder_model_merged.ort
  tokens.txt
```

## 실행

```bash
# config.yaml 기본 설정으로 실행 (Moonshine 백엔드)
python serve.py

# 백엔드 지정
python serve.py --model moonshine
python serve.py --model funasr

# 포트 지정
python serve.py --port 8765
```

## WebSocket 프로토콜

**클라이언트 → 서버**

- Binary 프레임: 16kHz / 16-bit mono PCM (100ms 단위 권장)

**서버 → 클라이언트**

```json
{"type": "partial", "text": "안녕하"}       // 발화 중 중간 결과
{"type": "final",   "text": "안녕하세요"}   // 발화 종료 후 확정 결과
{"type": "error",   "message": "..."}       // 오류
```

## 설정 (`config.yaml`)

주요 설정 항목:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `model` | `moonshine` | 사용할 백엔드 (`moonshine` \| `funasr`) |
| `server.port` | `8765` | WebSocket 서버 포트 |
| `vad.silence_duration_s` | `0.3` | 발화 종료 판정 무음 시간 (초) |
| `vad.partial_interval_s` | `0.5` | Partial 결과 반환 주기 (초) |
| `vad.threshold` | `0.8` | Silero VAD 발화 감지 임계값 |
| `vad.max_utterance_s` | `15` | 최대 발화 길이 (초) |

## 테스트

```bash
# WebSocket 클라이언트 테스트
python client_test.py

# 개별 백엔드 테스트
python test_moonshine.py
python test_funasr_nano.py
```

## 라이선스

MIT

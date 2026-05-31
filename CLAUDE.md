# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project targets a **CPU-only Korean real-time streaming STT (Speech-to-Text)** system for AICC (AI Contact Center) and real-time voice processing. The primary requirements are:

- CPU-only inference (no GPU required)
- Korean language support
- Streaming ASR with 100ms PCM packet input
- Partial/interim result delivery in real-time
- WebSocket/TCP-based integration
- On-premise (local) deployment

## Architecture Decision

Based on the research in `plan.md`, the chosen architecture is:

```
PCM 100ms → Silero VAD → SenseVoice → Partial Text → LLM → Supertonic 3
```

**Primary recommendation:** SenseVoice + Sherpa Runtime  
**Fallback option:** Sherpa-ONNX Korean Zipformer (`sherpa-onnx-streaming-zipformer-korean-2024-06-16`)

### Key model characteristics

| Model | Latency | Korean | Streaming | CPU |
|---|---|---|---|---|
| SenseVoice (Alibaba FunAudioLLM) | <80ms | ✓ | ✓ | ✓ |
| Sherpa Korean Zipformer | Low | ✓ | ✓ | ✓ |
| Faster Whisper | Medium | ✓ | Pseudo | ✓ |

SenseVoice is >5x faster than Whisper-Small and >15x faster than Whisper-Large. Whisper-based models are **not** true streaming — they buffer and re-infer, making them unsuitable for 100ms chunk environments.

## Status

The repository is in the **research/planning phase**. `plan.md` contains the full model comparison report in Korean.

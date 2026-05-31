"""
CPU 한국어 Streaming STT WebSocket 서버

프로토콜:
  클라이언트 → 서버: binary  16kHz / 16-bit mono PCM (100ms 단위 권장)
  서버 → 클라이언트: JSON
    {"type": "partial", "text": "..."}   # 중간 결과
    {"type": "final",   "text": "..."}   # 확정 결과 (무음 감지 후)
    {"type": "error",   "message": "..."}

실행:
  python serve.py                     # config.yaml 기준
  python serve.py --model moonshine
  python serve.py --model funasr
  python serve.py --port 8765
"""

import argparse
import asyncio
import collections
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import websockets

LOG_FILE = Path(__file__).parent / "logs" / "stt_server.log"

def setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 콘솔 핸들러
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    # 파일 핸들러
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

setup_logging()
log = logging.getLogger("stt-server")

CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── 설정 로드 ─────────────────────────────────────────────────────────────────

def load_config(model_override=None, port_override=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if model_override:
        cfg["model"] = model_override
    if port_override:
        cfg["server"]["port"] = port_override
    return cfg


# ── 백엔드: Moonshine (sherpa-onnx) ──────────────────────────────────────────

class MoonshineBackend:
    def __init__(self, cfg):
        import sherpa_onnx
        mc = cfg["moonshine"]
        log.info("Moonshine 모델 로딩 중...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=mc["encoder"],
            decoder=mc["decoder"],
            tokens=mc["tokens"],
            num_threads=mc.get("num_threads", 4),
        )
        self.sample_rate = cfg["vad"]["sample_rate"]
        log.info("Moonshine 로딩 완료")

    def recognize(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip().lstrip("▁").strip()


# ── 백엔드: Fun-ASR-MLT-Nano ──────────────────────────────────────────────────

class FunASRBackend:
    def __init__(self, cfg):
        import soundfile as sf
        import tempfile, os
        from funasr import AutoModel
        fc = cfg["funasr"]
        log.info("Fun-ASR-MLT-Nano 모델 로딩 중...")
        self.model = AutoModel(
            model=fc["model_id"],
            hub=fc.get("hub", "hf"),
            device="cpu",
            llm_dtype="fp32",
            disable_update=True,
        )
        self.language = fc.get("language", "ko")
        self.sample_rate = cfg["vad"]["sample_rate"]
        self._sf = sf
        self._tempfile = tempfile
        self._os = os
        log.info("Fun-ASR-MLT-Nano 로딩 완료")

    def recognize(self, samples: np.ndarray) -> str:
        # funasr AutoModel은 파일 경로를 받으므로 임시 wav 생성
        with self._tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        try:
            self._sf.write(path, samples, self.sample_rate, subtype="PCM_16")
            result = self.model.generate(
                input=path,
                language=self.language,
                llm_dtype="fp32",
            )
            return result[0]["text"].strip() if result else ""
        finally:
            self._os.unlink(path)


# ── VAD 버퍼 (Silero VAD 기반) ───────────────────────────────────────────────

class SileroVadBuffer:
    """
    Silero VAD 기반 발화 구간 검출 및 오디오 버퍼.

    Silero VAD 는 16kHz 기준 512-sample(32ms) 단위로 처리.
    들어오는 100ms(1600-sample) 청크를 내부에서 분할해 처리하며,
    발화 시작/종료 이벤트를 기반으로 STT에 전달할 오디오를 누적한다.
    """
    SILERO_CHUNK = 512   # 16kHz 고정 요구사항

    def __init__(self, sr: int, silence_duration_s: float, partial_interval_s: float,
                 threshold: float = 0.5, speech_pad_ms: int = 200,
                 max_utterance_s: float = 20.0):
        import torch
        from silero_vad import load_silero_vad, VADIterator
        self._torch = torch
        model = load_silero_vad()
        self.vad = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sr,
            min_silence_duration_ms=int(silence_duration_s * 1000),
            speech_pad_ms=speech_pad_ms,
        )
        self.sr              = sr
        self.partial_samples = int(partial_interval_s * sr)
        self.max_samples     = int(max_utterance_s * sr)
        # 발화 시작 직전 오디오를 보존하는 pre-buffer
        pre_len = int(speech_pad_ms / 1000 * sr) + self.SILERO_CHUNK
        self._pre     = collections.deque(maxlen=pre_len)
        self._pending = np.array([], dtype=np.float32)  # reset()에서 유지됨
        self.reset()

    def reset(self):
        self._speech:   list[np.ndarray] = []
        # _pending 은 유지 — 아직 처리 안 된 sub-chunk 가 있을 수 있음
        self.in_speech  = False
        self._samples_since_partial = 0
        self._end_flagged = False
        self.vad.reset_states()

    # ── 외부 인터페이스 ──────────────────────────────────────────────────────

    @property
    def has_speech(self) -> bool:
        return bool(self._speech)

    @property
    def is_end_of_utterance(self) -> bool:
        return self._end_flagged

    @property
    def should_emit_partial(self) -> bool:
        return self.in_speech and self._samples_since_partial >= self.partial_samples

    def get_audio(self) -> np.ndarray:
        return np.concatenate(self._speech) if self._speech else np.array([], np.float32)

    def mark_partial_emitted(self):
        self._samples_since_partial = 0

    def flush_utterance(self) -> np.ndarray:
        audio = self.get_audio()
        self.reset()
        return audio

    # ── 오디오 입력 ──────────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray):
        """100ms 청크를 받아 Silero 단위(512 samples)로 처리."""
        self._end_flagged = False
        buf = np.concatenate([self._pending, chunk])
        i   = 0

        while i + self.SILERO_CHUNK <= len(buf):
            sub   = buf[i: i + self.SILERO_CHUNK]
            event = self.vad(self._torch.from_numpy(sub))

            # 발화 시작 감지
            if event and "start" in event:
                self.in_speech = True
                pre = np.array(list(self._pre), dtype=np.float32)
                if len(pre):
                    self._speech.append(pre)

            # 발화 중 오디오 누적
            if self.in_speech:
                self._speech.append(sub)
                self._samples_since_partial += self.SILERO_CHUNK

                # 최대 발화 길이 초과 → 강제 flush
                total = sum(len(s) for s in self._speech)
                if total >= self.max_samples:
                    log.info(f"max_utterance 초과 ({total/self.sr:.1f}s) → 강제 flush")
                    self._end_flagged = True
                    self.in_speech    = False
                    i += self.SILERO_CHUNK
                    break
            else:
                # 침묵 구간: pre-buffer 갱신
                for s in sub:
                    self._pre.append(s)

            # 발화 종료 감지
            if event and "end" in event:
                self._end_flagged = True
                self.in_speech    = False

            i += self.SILERO_CHUNK

        self._pending = buf[i:]


# ── WebSocket 핸들러 ──────────────────────────────────────────────────────────

def make_handler(backend, vad_cfg):
    sr           = vad_cfg["sample_rate"]
    silence_s    = vad_cfg.get("silence_duration_s", 0.8)
    partial_s    = vad_cfg.get("partial_interval_s", 0.5)
    threshold    = vad_cfg.get("threshold", 0.5)
    speech_pad   = vad_cfg.get("speech_pad_ms", 200)
    max_utt_s    = vad_cfg.get("max_utterance_s", 20.0)

    log.info(
        f"Silero VAD 설정 — silence: {silence_s}s / threshold: {threshold} / "
        f"speech_pad: {speech_pad}ms / max_utterance: {max_utt_s}s / partial: {partial_s}s"
    )
    log.info("Silero VAD 모델 로딩 중...")
    vad_proto = SileroVadBuffer(sr, silence_s, partial_s, threshold, speech_pad, max_utt_s)
    log.info("Silero VAD 로딩 완료")

    async def handler(websocket):
        addr = websocket.remote_address
        log.info(f"연결: {addr}")
        vad = SileroVadBuffer(sr, silence_s, partial_s, threshold, speech_pad, max_utt_s)
        last_partial = ""   # 현재 발화 구간의 가장 최근 partial 텍스트

        async def flush_final(trigger: str = "eos"):
            """버퍼 음성을 처리해 final 반환.
            Moonshine이 빈 결과를 내면 last_partial을 fallback으로 사용."""
            nonlocal last_partial
            text = ""

            if vad.has_speech:
                audio = vad.flush_utterance()
                t0 = time.perf_counter()
                text = (await asyncio.get_event_loop().run_in_executor(
                    None, backend.recognize, audio
                )).strip()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if not text:
                    # Moonshine 빈 결과 → 직전 partial로 대체
                    if last_partial:
                        text = last_partial
                        log.info(f"[final/{trigger}] (빈 결과 → partial fallback) {text!r}  ({elapsed_ms:.0f}ms)")
                    else:
                        log.info(f"[final/{trigger}] (빈 결과 — 전송 생략)  ({elapsed_ms:.0f}ms)")
                else:
                    log.info(f"[final/{trigger}] {text!r}  ({elapsed_ms:.0f}ms)")
            elif last_partial:
                # 버퍼가 이미 비어있지만 미전송 partial이 있는 경우 (EOS 직전 VAD flush 실패)
                text = last_partial
                log.info(f"[final/{trigger}] (버퍼 비어있음 → partial fallback) {text!r}")

            if text:
                last_partial = ""   # 사용한 partial은 초기화
                await websocket.send(json.dumps({"type": "final", "text": text}, ensure_ascii=False))

        try:
            async for message in websocket:
                # ── EOS 신호 (텍스트 JSON) ──────────────────────────────────
                if isinstance(message, str):
                    try:
                        ctrl = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if ctrl.get("type") == "eos":
                        log.info("EOS 수신 → 버퍼 flush")
                        await flush_final()
                        await websocket.send(json.dumps({"type": "eos_ack"}))
                    continue

                # ── PCM 오디오 (바이너리) ────────────────────────────────────
                pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                vad.push(pcm)

                if not vad.has_speech:
                    continue

                if vad.is_end_of_utterance:
                    await flush_final(trigger="vad")

                elif vad.should_emit_partial:
                    audio = vad.get_audio()
                    vad.mark_partial_emitted()
                    t0 = time.perf_counter()
                    text = (await asyncio.get_event_loop().run_in_executor(
                        None, backend.recognize, audio
                    )).strip()
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    if not text:
                        continue
                    last_partial = text   # 발화 구간의 최신 partial 저장
                    log.info(f"[partial] {text!r}  ({elapsed_ms:.0f}ms)")
                    await websocket.send(json.dumps({"type": "partial", "text": text}, ensure_ascii=False))

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception as e:
            log.error(f"오류: {e}")
            try:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass
        finally:
            log.info(f"연결 종료: {addr}")

    return handler


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPU Korean STT WebSocket Server")
    parser.add_argument("--model", choices=["moonshine", "funasr"], help="사용할 모델")
    parser.add_argument("--port", type=int, help="WebSocket 포트 (기본: config.yaml)")
    args = parser.parse_args()

    cfg = load_config(model_override=args.model, port_override=args.port)
    model_name = cfg["model"]
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]

    log.info(f"모델: {model_name}")

    if model_name == "moonshine":
        backend = MoonshineBackend(cfg)
    elif model_name == "funasr":
        backend = FunASRBackend(cfg)
    else:
        sys.exit(f"알 수 없는 모델: {model_name}")

    handler = make_handler(backend, cfg["vad"])

    async def serve():
        async with websockets.serve(handler, host, port):
            log.info(f"서버 시작: ws://{host}:{port}  (Ctrl+C로 종료)")
            await asyncio.Future()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        log.info("서버 종료")


if __name__ == "__main__":
    main()

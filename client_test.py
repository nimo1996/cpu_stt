"""
STT WebSocket 테스트 클라이언트

사용법:
  # 기본 (내장 테스트 음성, 실시간 속도)
  python client_test.py

  # 음성 파일 지정
  python client_test.py --audio path/to/audio.wav

  # 빠른 전송 (실시간 대기 없이 즉시 전송)
  python client_test.py --fast

  # 서버 주소 지정
  python client_test.py --url ws://192.168.0.10:8765

  # 청크 크기 변경 (기본 100ms)
  python client_test.py --chunk-ms 200
"""

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

# ── 기본값 ────────────────────────────────────────────────────────────────────

DEFAULT_URL      = "ws://127.0.0.1:8765"
DEFAULT_AUDIO    = str(Path(__file__).parent / "models/moonshine-tiny-ko/test_wavs/0.wav")
TARGET_SR        = 16000   # 서버가 기대하는 샘플레이트
CHUNK_MS         = 100


# ── 오디오 로드 ───────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """WAV / MP3 → 16kHz mono float32 ndarray"""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[오류] 파일 없음: {path}")

    if p.suffix.lower() == ".wav":
        with wave.open(path, "rb") as wf:
            sr      = wf.getframerate()
            n_ch    = wf.getnchannels()
            sw      = wf.getsampwidth()
            frames  = wf.readframes(wf.getnframes())

        if sw == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            sys.exit(f"[오류] 지원되지 않는 샘플 폭: {sw}bytes")

        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
    else:
        try:
            import torchaudio
            wav, sr = torchaudio.load(path)
            samples = wav.mean(0).numpy().astype(np.float32)
        except Exception as e:
            sys.exit(f"[오류] WAV 아닌 파일은 torchaudio 필요: {e}")

    # 리샘플링
    if sr != TARGET_SR:
        try:
            # scipy: anti-aliasing 포함 고품질 리샘플링
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, TARGET_SR)
            samples = resample_poly(samples, TARGET_SR // g, sr // g).astype(np.float32)
        except ImportError:
            try:
                import torch
                import torchaudio
                tensor = torch.from_numpy(samples).unsqueeze(0)
                tensor = torchaudio.functional.resample(tensor, sr, TARGET_SR)
                samples = tensor.squeeze(0).numpy().astype(np.float32)
            except Exception:
                # 최후 수단: 선형 보간 (품질 낮음)
                ratio   = TARGET_SR / sr
                n_new   = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, n_new),
                    np.arange(len(samples)), samples
                ).astype(np.float32)
        print(f"  리샘플링: {sr}Hz → {TARGET_SR}Hz")

    return samples.astype(np.float32)


def to_pcm16(samples: np.ndarray) -> bytes:
    """float32 → 16-bit PCM bytes"""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


# ── 결과 출력 헬퍼 ────────────────────────────────────────────────────────────

_last_partial_len = 0

def print_partial(text: str, elapsed_ms: float):
    global _last_partial_len
    tag  = f"\033[33m[partial {elapsed_ms:5.0f}ms]\033[0m"  # 노란색
    line = f"  {tag}  {text}"
    # 이전 partial 줄을 덮어씀
    print(f"\r{line:<100}", end="", flush=True)
    _last_partial_len = len(line)

def print_final(text: str, elapsed_ms: float):
    global _last_partial_len
    # partial 줄 지우고 final 출력
    print(f"\r{' ' * 100}\r", end="")
    tag = f"\033[32m[final   {elapsed_ms:5.0f}ms]\033[0m"   # 초록색
    print(f"  {tag}  {text}")
    _last_partial_len = 0

def print_error(msg: str):
    print(f"\r  \033[31m[error]\033[0m  {msg}")


# ── 메인 클라이언트 ───────────────────────────────────────────────────────────

async def run(url: str, audio_path: str, chunk_ms: int, fast: bool):
    print(f"\n{'='*60}")
    print(f"  STT 테스트 클라이언트")
    print(f"{'='*60}")
    print(f"  서버   : {url}")
    print(f"  파일   : {audio_path}")
    print(f"  청크   : {chunk_ms}ms  |  모드: {'빠른 전송' if fast else '실시간'}")

    samples  = load_audio(audio_path)
    duration = len(samples) / TARGET_SR
    print(f"  오디오 : {duration:.2f}초  ({len(samples):,} samples @ {TARGET_SR}Hz)")
    print(f"{'='*60}\n")

    chunk_samples = int(TARGET_SR * chunk_ms / 1000)
    n_chunks      = (len(samples) + chunk_samples - 1) // chunk_samples

    try:
        async with websockets.connect(url) as ws:
            print(f"  서버 연결 성공\n")

            send_done  = asyncio.Event()
            results    = []          # (type, text, recv_ms)
            session_t0 = time.perf_counter()

            # ── 수신 태스크 ──────────────────────────────────────────────────
            async def receiver():
                async for raw in ws:
                    recv_ms = (time.perf_counter() - session_t0) * 1000
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    mtype = msg.get("type")
                    text  = msg.get("text", "")
                    err   = msg.get("message", "")

                    if mtype == "partial":
                        print_partial(text, recv_ms)
                        results.append(("partial", text, recv_ms))
                    elif mtype == "final":
                        print_final(text, recv_ms)
                        results.append(("final", text, recv_ms))
                    elif mtype == "error":
                        print_error(err)
                        results.append(("error", err, recv_ms))
                    elif mtype == "eos_ack":
                        # 서버가 EOS 처리 완료 → 수신 종료
                        return

            # ── 전송 태스크 ──────────────────────────────────────────────────
            async def sender():
                print(f"  전송 시작 ({n_chunks}개 청크) ...\n")
                chunk_interval = chunk_ms / 1000.0

                for i in range(n_chunks):
                    chunk = samples[i * chunk_samples : (i + 1) * chunk_samples]
                    await ws.send(to_pcm16(chunk))

                    if not fast:
                        await asyncio.sleep(chunk_interval)

                # EOS 신호 전송 → 서버가 버퍼를 즉시 flush해 final 반환
                await ws.send(json.dumps({"type": "eos"}))

                send_done.set()
                print(f"\n  EOS 전송 완료. 최종 결과 대기 중 ...\n")
                # final 수신까지 최대 15초 대기
                await asyncio.sleep(15)

            recv_task = asyncio.create_task(receiver())
            await sender()
            try:
                await asyncio.wait_for(recv_task, timeout=12)
            except asyncio.TimeoutError:
                recv_task.cancel()

    except (ConnectionRefusedError, OSError):
        sys.exit(f"\n[오류] 서버에 연결할 수 없습니다: {url}\n서버가 실행 중인지 확인하세요.")
    except websockets.exceptions.WebSocketException as e:
        sys.exit(f"\n[오류] WebSocket 오류: {e}")

    # ── 요약 ────────────────────────────────────────────────────────────────
    total_elapsed = (time.perf_counter() - session_t0) * 1000
    finals  = [r for r in results if r[0] == "final"]
    partials = [r for r in results if r[0] == "partial"]

    print(f"\n{'='*60}")
    print(f"  결과 요약")
    print(f"{'='*60}")
    print(f"  오디오 길이     : {duration:.2f}초")
    print(f"  총 경과 시간    : {total_elapsed:.0f}ms")
    print(f"  partial 수신    : {len(partials)}회")
    print(f"  final   수신    : {len(finals)}회")
    if finals:
        last_final_ms = finals[-1][2]
        lag = last_final_ms - duration * 1000
        print(f"  최종 결과 수신  : {last_final_ms:.0f}ms  (오디오 종료 후 +{lag:.0f}ms)")
        print(f"\n  최종 텍스트:")
        for _, text, ms in finals:
            print(f"    [{ms:.0f}ms]  {text}")
    print(f"{'='*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="STT WebSocket 테스트 클라이언트")
    ap.add_argument("--url",      default=DEFAULT_URL,   help="서버 주소 (기본: ws://127.0.0.1:8765)")
    ap.add_argument("--audio",    default=DEFAULT_AUDIO, help="입력 오디오 파일 경로")
    ap.add_argument("--chunk-ms", type=int, default=CHUNK_MS, dest="chunk_ms",
                    help="청크 크기 ms (기본: 100)")
    ap.add_argument("--fast",     action="store_true",
                    help="실시간 대기 없이 즉시 전송 (레이턴시 측정용)")
    args = ap.parse_args()

    asyncio.run(run(args.url, args.audio, args.chunk_ms, args.fast))


if __name__ == "__main__":
    main()

import time
import os
import warnings
warnings.filterwarnings("ignore")

from funasr import AutoModel

MODEL_ID = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
# 번들 한국어 예시 음성 (모델 캐시에서 가져옴)
CACHE_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--FunAudioLLM--Fun-ASR-MLT-Nano-2512"
)

def find_ko_wav():
    for root, _, files in os.walk(CACHE_DIR):
        for f in files:
            if f == "ko.mp3":
                return os.path.join(root, f)
    # 없으면 Moonshine 테스트용 wav 재활용
    return "/home/aicc/cpu_stt/models/moonshine-tiny-ko/test_wavs/0.wav"

print("=" * 55)
print("Fun-ASR-MLT-Nano-2512 한국어 테스트")
print("=" * 55)

# ── 1. 모델 로드 ──────────────────────────────────────────
print("\n[1] 모델 로딩 중...")
t0 = time.perf_counter()
model = AutoModel(
    model=MODEL_ID,
    hub="hf",
    device="cpu",
    llm_dtype="fp32",
    disable_update=True,
)
load_time = time.perf_counter() - t0
print(f"    로딩 완료: {load_time:.1f}초")

# ── 2. 오디오 파일 확인 ───────────────────────────────────
audio_path = find_ko_wav()
print(f"\n[2] 테스트 오디오: {audio_path}")

# 오디오 길이 측정
try:
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    duration = wav.shape[-1] / sr
    print(f"    길이: {duration:.2f}초 / {sr}Hz")
except Exception as e:
    duration = None
    print(f"    길이 측정 실패: {e}")

# ── 3. Offline 인식 ───────────────────────────────────────
print("\n[3] Offline 인식 테스트...")
t0 = time.perf_counter()
result = model.generate(
    input=audio_path,
    language="ko",
    llm_dtype="fp32",
)
elapsed = time.perf_counter() - t0

text = result[0]["text"] if result else "(결과 없음)"
print(f"    인식 결과: {text}")
print(f"    처리 시간: {elapsed*1000:.0f}ms")
if duration:
    rtf = elapsed / duration
    print(f"    RTF: {rtf:.3f}  ({'실시간 가능 ✓' if rtf < 1.0 else '실시간 불가 — CPU 부담 큼'})")

# ── 4. 반복 실행으로 안정 속도 측정 ─────────────────────
print("\n[4] 반복 3회 속도 측정...")
times = []
for i in range(3):
    t0 = time.perf_counter()
    model.generate(input=audio_path, language="ko", llm_dtype="fp32")
    times.append(time.perf_counter() - t0)
    print(f"    {i+1}회: {times[-1]*1000:.0f}ms")

avg = sum(times) / len(times)
print(f"    평균: {avg*1000:.0f}ms")
if duration:
    print(f"    평균 RTF: {avg/duration:.3f}")

print("\n" + "=" * 55)
print("Moonshine Tiny Ko 비교 (이전 테스트)")
print("  Offline RTF : 0.031  (217ms / 6.92초)")
print("  모델 크기   : 69MB (27M params)")
print("=" * 55)

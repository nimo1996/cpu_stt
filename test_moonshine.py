import time
import wave
import numpy as np
import sherpa_onnx

MODEL_DIR = "/home/aicc/cpu_stt/models/moonshine-tiny-ko"
TEST_WAV  = f"{MODEL_DIR}/test_wavs/0.wav"

def load_wav(path):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1, "mono only"
        assert f.getsampwidth() == 2, "16-bit only"
        sample_rate = f.getframerate()
        raw = f.readframes(f.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

def test_offline():
    print("=== Offline (일괄) 인식 테스트 ===")
    recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
        encoder=f"{MODEL_DIR}/encoder_model.ort",
        decoder=f"{MODEL_DIR}/decoder_model_merged.ort",
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=4,
        debug=False,
    )
    samples, sr = load_wav(TEST_WAV)
    duration = len(samples) / sr
    print(f"오디오 길이: {duration:.2f}초 / 샘플레이트: {sr}Hz")

    stream = recognizer.create_stream()
    t0 = time.perf_counter()
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    elapsed = time.perf_counter() - t0

    result = stream.result.text.strip()
    rtf = elapsed / duration
    print(f"인식 결과: {result}")
    print(f"처리 시간: {elapsed*1000:.1f}ms  /  RTF: {rtf:.3f}  (1.0 미만이면 실시간 가능)")
    return result

def test_streaming_sim(chunk_ms=100):
    print(f"\n=== Streaming 시뮬레이션 ({chunk_ms}ms 청크) ===")
    recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
        encoder=f"{MODEL_DIR}/encoder_model.ort",
        decoder=f"{MODEL_DIR}/decoder_model_merged.ort",
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=4,
        debug=False,
    )
    samples, sr = load_wav(TEST_WAV)
    chunk_size = int(sr * chunk_ms / 1000)
    total_chunks = (len(samples) + chunk_size - 1) // chunk_size
    print(f"총 {total_chunks}개 청크 ({chunk_ms}ms 단위)")

    latencies = []
    all_samples = []

    for i in range(total_chunks):
        chunk = samples[i * chunk_size: (i + 1) * chunk_size]
        all_samples.append(chunk)

        # 매 청크마다 누적 음성을 재인식 (유사 streaming)
        t0 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, np.concatenate(all_samples))
        recognizer.decode_stream(stream)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        partial = stream.result.text.strip()
        print(f"  청크 {i+1:2d}: {latency:6.1f}ms  →  {partial}")

    print(f"\n청크별 평균 지연: {np.mean(latencies):.1f}ms")
    print(f"청크별 최대 지연: {np.max(latencies):.1f}ms")
    print(f"100ms 실시간 기준 초과 청크: {sum(1 for l in latencies if l > 100)}/{len(latencies)}")

if __name__ == "__main__":
    result = test_offline()
    if not result:
        print("\n[경고] 인식 결과가 비어 있습니다. 모델 파일이나 API를 확인하세요.")
    else:
        test_streaming_sim(chunk_ms=100)

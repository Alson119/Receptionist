"""Offline speech for the receptionist.

  * TTS: Piper neural voice -> played with ffplay
  * ASR: microphone via ffmpeg (PulseAudio, works under WSLg) -> faster-whisper

Both go through ffmpeg/ffplay so no PortAudio / espeak system packages are needed.
"""
import os
import subprocess
import tempfile
import time
import wave

import numpy as np

RATE = 16000
CHUNK = 480  # 30 ms at 16 kHz


class TTS:
    def __init__(self, voice="weights/piper/en_US-lessac-medium.onnx"):
        from piper import PiperVoice

        if not os.path.exists(voice):
            raise FileNotFoundError(
                f"{voice} missing - run: python3 -m piper.download_voices "
                f"{os.path.basename(voice)[:-5]} --data-dir {os.path.dirname(voice)}")
        self.voice = PiperVoice.load(voice)

    def say(self, text):
        """Speak and block until playback ends (so the mic doesn't hear the robot)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                self.voice.synthesize_wav(text, w)
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                           stdin=subprocess.DEVNULL, check=False)
        finally:
            os.remove(path)


class ASR:
    def __init__(self, model="small.en", device="cuda", mic="default"):
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(model, device=device, compute_type="float16",
                                      download_root="weights/whisper")
        except Exception as e:  # no usable GPU / CUDA libs
            print(f"(Whisper on {device} failed: {e}; using CPU)")
            self.model = WhisperModel(model, device="cpu", compute_type="int8",
                                      download_root="weights/whisper")
        self.mic = mic
        self._transcribe(np.zeros(RATE, np.float32))  # warm up, first call is slow

    def record(self, max_wait=10.0, max_len=15.0, silence_end=1.0):
        """Record one utterance: wait for speech, stop after `silence_end` s of quiet."""
        # stdin=DEVNULL: ffmpeg otherwise reads the terminal and eats typed answers
        proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "quiet", "-f", "pulse", "-i", self.mic,
             "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        chunks, started, quiet, t0 = [], False, 0.0, time.time()
        noise = []
        try:
            while True:
                raw = proc.stdout.read(CHUNK * 2)
                if len(raw) < CHUNK * 2:
                    break
                x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
                rms = float(np.sqrt(np.mean(x ** 2)))
                elapsed = time.time() - t0
                if len(noise) < 10:  # first 0.3 s: measure the room's noise floor
                    noise.append(rms)
                    chunks.append(x)
                    continue
                thr = max(0.01, 3 * float(np.median(noise)))
                if not started:
                    chunks = chunks[-10:] + [x]  # keep a little audio before speech starts
                    if rms > thr:
                        started = True
                        print("   (listening...)")
                    elif elapsed > max_wait:
                        return None
                    continue
                chunks.append(x)
                quiet = quiet + CHUNK / RATE if rms < thr else 0.0
                if quiet >= silence_end or len(chunks) * CHUNK / RATE > max_len:
                    break
        finally:
            proc.terminate()
            proc.wait()
        return np.concatenate(chunks) if started else None

    def _transcribe(self, audio):
        segs, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()

    def listen(self):
        audio = self.record()
        return self._transcribe(audio) if audio is not None else ""

"""Live microphone level meter - talk and watch the bar.

  python3 mic_test.py            # 10 s on the default (Windows) mic
  python3 mic_test.py 20         # run longer
"""
import subprocess
import sys
import time

import numpy as np

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
proc = subprocess.Popen(
    ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "pulse", "-i", "default",
     "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
print(f"Talk now ({secs:.0f} s). Speech should reach roughly -30 dB or louder.\n")
t0, peak = time.time(), -100.0
try:
    while time.time() - t0 < secs:
        raw = proc.stdout.read(3200)  # 0.1 s
        if not raw:
            print("no audio stream - is WSLg running? try `wsl --shutdown` from Windows")
            break
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        peak = max(peak, db)
        bar = "#" * int(max(0, db + 70) / 1.5)
        print(f"\r{db:6.1f} dB |{bar:<47}|", end="", flush=True)
finally:
    proc.terminate()
print(f"\n\nloudest: {peak:.1f} dB -> "
      + ("mic is working" if peak > -35 else "too quiet: mic muted, wrong device, or blocked"))

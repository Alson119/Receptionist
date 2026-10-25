# Receptionist simulation (YOLO-World + Gemma 4)

A desktop prototype of the RoboCup@Home receptionist task:

- **YOLO-World** (open-vocabulary) finds guests at the door and chairs/sofas in the living room, and works out which seats are free (sofas are split into per-person slots).
- **Gemma 4 12B via Ollama** pulls name / favourite drink / interest out of what the guest says, describes their appearance from a camera crop, and writes the host-guest introduction.
- Speech is typed in the terminal for now; swap `listen()` / `say()` in `receptionist.py` for ASR/TTS later.

| File | What it does |
|---|---|
| `perception.py` | YOLO-World wrapper, free-seat logic, drawing |
| `llm.py` | Ollama client and the three prompts |
| `receptionist.py` | The task state machine |
| `test_setup.py` | Checks Ollama, Gemma 4 and YOLO-World end to end |

## Recommended setup: Python on Windows, Ollama in WSL

Your Windows `Yolo` venv already has ultralytics and a CUDA build of torch, and Windows Python can use the webcam directly (WSL2 can't without extra USB passthrough). WSL forwards ports to Windows, so Windows reaches the WSL Ollama at `http://localhost:11434`.

**1. In WSL**, make sure Ollama is serving and check the exact Gemma tag:

```bash
ollama serve        # skip if it's already running as a service
ollama list         # note the gemma4 tag, e.g. gemma4:12b
```

**2. In PowerShell:**

```powershell
cd $HOME\VScode\receptionist_sim
..\Yolo\Scripts\Activate.ps1
pip install requests
$env:RECEPTIONIST_MODEL = "gemma4:12b"   # only if your tag differs
python test_setup.py
```

**3. Run the task:**

```powershell
python receptionist.py                    # webcam 0 for door and room
python receptionist.py --door 1 --room 0  # two cameras
python receptionist.py --door door.jpg --room room.jpg --guests 1   # still images
```

## Voice (WSL)

`speech.py` gives offline text-to-speech (Piper) and speech-to-text (faster-whisper). Audio goes through ffmpeg/ffplay and WSLg's PulseAudio, so no `apt` packages are needed.

```bash
python3 -m pip install --user faster-whisper piper-tts
python3 -m piper.download_voices en_US-lessac-medium --data-dir weights/piper
python3 receptionist.py --door 2 --room 4 --tts --asr
```

- `--tts` only / `--asr` only also work. If nothing is heard twice, it falls back to typing.
- `--whisper base.en` is faster; `medium.en` is more accurate with unusual names.
- Other voices: https://huggingface.co/rhasspy/piper-voices (pass with `--voice`).

## Known snags

- **No preview window / `cv2.error: The function is not implemented`**: ultralytics installed `opencv-python-headless` over `opencv-python`. The script falls back to saving frames in `output/`. To get the window back:
  `pip uninstall -y opencv-python-headless opencv-python; pip install opencv-python`
- **Running inside WSL**: webcams passed through to WSL don't show up as index 0. Probe them with `cv2.VideoCapture(i, cv2.CAP_V4L2)` for `i` in 0-5 (e.g. `--door 2 --room 4`). If CLIP installs as a package called `UNKNOWN`, the system pip is too old: `python3 -m pip install --user --upgrade pip`, then reinstall.
- **First YOLO-World run** downloads the weights and installs OpenAI CLIP from GitHub, which needs `git` on PATH.
- **Windows can't reach Ollama**: in WSL run `OLLAMA_HOST=0.0.0.0 ollama serve`, or set `OLLAMA_URL` to the WSL IP (`hostname -I` in WSL).
- **Slow appearance description**: the first call loads the model into VRAM. If YOLO-World and Gemma together run out of the 16 GB, use `--weights yolov8s-worldv2.pt`.

## Tuning

- Classes: edit `SEAT_CLASSES` in `perception.py` (e.g. add `"beanbag"`).
- Guest-at-door trigger: `min_height` in `wait_for_guest` (fraction of frame height).
- Host details: `--host-name`, `--host-drink`, `--host-interest`.

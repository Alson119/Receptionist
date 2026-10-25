"""RoboCup@Home receptionist simulation on a desktop.

Flow per guest:
  wait at door -> guest steps up -> greet, get name/drink/interest -> describe them
  -> "guide" to living room -> find a free seat -> offer it -> introduce to host
  (and, for later guests, describe the earlier guests to them).

Speech is typed in the terminal by default; --tts speaks with Piper and --asr listens
with Whisper (see speech.py), falling back to typing if nothing is heard.
Camera: webcam index or an image/video path, separately for door and living room.

  python receptionist.py                              # one webcam for both
  python receptionist.py --door door.jpg --room room.jpg
  python receptionist.py --door 2 --room 4 --tts --asr   # full voice, WSL cameras
"""
import argparse
import os
import re
import time

import cv2

import llm
from perception import Perception, crop, draw, iou, largest_person, seat_slots, where

OUT_DIR = "output"


# ------------------------------------------------------------------ I/O stubs

def say(text):
    print(f"\n🤖 ROBOT: {text}")
    if TTS:
        TTS.say(text)


def listen(prompt="guest"):
    if ASR:
        for attempt in range(2):
            text = ASR.listen()
            if text:
                print(f"🧑 {prompt.upper()} (heard): {text}")
                return text
            if attempt == 0:
                say("Sorry, I didn't catch that. Could you say it again?")
        print("   (still nothing heard - type it instead)")
    return input(f"🧑 {prompt.upper()} (type what they say): ").strip()


class Source:
    """Webcam index, video file, or still image, read on demand."""

    def __init__(self, spec):
        self.image = None
        self.cap = None
        if spec.isdigit():
            self.cap = cv2.VideoCapture(int(spec), cv2.CAP_DSHOW if os.name == "nt" else 0)
        elif spec.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            self.image = cv2.imread(spec)
            if self.image is None:
                raise FileNotFoundError(spec)
        else:
            self.cap = cv2.VideoCapture(spec)
        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"cannot open source {spec}")

    def read(self):
        if self.image is not None:
            return self.image.copy()
        for _ in range(3):  # drop stale buffered frames from webcams
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame


class Viewer:
    """cv2.imshow when available, else save annotated frames to ./output."""

    def __init__(self):
        self.gui = True
        os.makedirs(OUT_DIR, exist_ok=True)
        self.n = 0

    def show(self, name, img, save=False):
        if self.gui:
            try:
                cv2.imshow("receptionist", img)
                cv2.waitKey(1)
            except cv2.error:
                self.gui = False
                print("(no OpenCV GUI available - saving frames to ./output instead)")
        if save or not self.gui:
            self.n += 1
            cv2.imwrite(os.path.join(OUT_DIR, f"{self.n:02d}_{name}.jpg"), img)


# ------------------------------------------------------------------ task steps

def wait_for_guest(src, per, view, min_height=0.4, frames_needed=5, timeout=120):
    """Block until a person fills enough of the frame for several frames."""
    say("Waiting at the door for a guest...")
    streak, t0 = 0, time.time()
    while time.time() - t0 < timeout:
        frame = src.read()
        dets = per.detect(frame)
        view.show("door", draw(frame, dets))
        p = largest_person(dets)
        if p and p.h > min_height * frame.shape[0]:
            streak += 1
            if streak >= frames_needed or src.image is not None:
                return frame, p
        else:
            streak = 0
            if src.image is not None:  # a still image won't change, so don't spin until timeout
                raise RuntimeError("no guest (large enough person) in the door image")
    raise TimeoutError("no guest arrived")


def get_guest_info():
    say("Hello and welcome! May I have your name, your favourite drink, "
        "and something you're interested in?")
    info = llm.extract_guest_info(listen())
    for key, question in (("name", "Sorry, what is your name?"),
                          ("drink", "What is your favourite drink?"),
                          ("interest", "And what are you interested in?")):
        tries = 0
        while not info.get(key) and tries < 2:
            say(question)
            info[key] = llm.extract_guest_info(listen()).get(key) or None
            tries += 1
    info["name"] = info["name"] or "friend"
    parts = [f"Nice to meet you, {info['name']}."]
    if info["drink"]:
        parts.append(f"Your favourite drink is {info['drink']}.")
    if info["interest"]:
        parts.append(f"You're into {info['interest']}.")
    say(" ".join(parts) + " Is that right?")
    if is_no(listen()):
        say("Sorry about that, let's try again.")
        return get_guest_info()
    return info


NO_WORDS = {"no", "nope", "nah", "wrong", "incorrect", "not"}


def is_no(answer):
    words = re.findall(r"[a-z']+", answer.lower())
    return bool(words) and (words[0] in NO_WORDS or "wrong" in words or "incorrect" in words)


def likes(person):
    """'who likes cola and is into football', skipping anything unknown."""
    bits = []
    if person.get("drink"):
        bits.append(f"likes {person['drink']}")
    if person.get("interest"):
        bits.append(f"is into {person['interest']}")
    return f", who {' and '.join(bits)}" if bits else ""


def find_seat(src, per, view, given=()):
    """Pick a free seat, skipping seats already offered to earlier guests."""
    frame = src.read()
    dets = per.detect(frame)
    slots = seat_slots(dets)
    for s in slots:
        if any(iou(s.box, g) > 0.5 for g in given):
            s.occupied = True
    free = [s for s in slots if not s.occupied]
    # prefer single chairs, then larger slots
    free.sort(key=lambda s: (s.seat in ("sofa", "couch", "bench"),
                             -(s.box[2] - s.box[0]) * (s.box[3] - s.box[1])))
    chosen = free[0] if free else None
    view.show("room", draw(frame, dets, slots, chosen), save=True)
    return frame, chosen, dets


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--door", default="0", help="webcam index or image/video path")
    ap.add_argument("--room", default=None, help="defaults to the door source")
    ap.add_argument("--guests", type=int, default=2)
    ap.add_argument("--weights", default="weights/yolov8s-worldv2.pt",
                    help="yolov8l-worldv2.pt is more accurate if the GPU has room")
    ap.add_argument("--host-name", default="John")
    ap.add_argument("--host-drink", default="milk")
    ap.add_argument("--host-interest", default="robotics")
    ap.add_argument("--tts", action="store_true", help="speak out loud with Piper")
    ap.add_argument("--voice", default="weights/piper/en_US-lessac-medium.onnx")
    ap.add_argument("--asr", action="store_true", help="listen on the mic with Whisper")
    ap.add_argument("--whisper", default="small.en", help="tiny.en / base.en / small.en / medium.en")
    ap.add_argument("--mic", default="default", help="PulseAudio source name")
    args = ap.parse_args()

    global TTS, ASR
    if args.tts or args.asr:
        import speech
    if args.tts:
        TTS = speech.TTS(args.voice)
    if args.asr:
        print("Loading Whisper...")
        ASR = speech.ASR(args.whisper, mic=args.mic)

    print(f"Using Ollama model: {llm.resolve_model()} at {llm.OLLAMA_URL}")
    print("Loading the model into the GPU (about a minute the first time)...")
    t = time.time()
    llm.warm_up()
    print(f"   ready in {time.time() - t:.1f}s")
    per = Perception(args.weights)
    door = Source(args.door)
    room = door if args.room in (None, args.door) else Source(args.room)
    view = Viewer()
    host = {"name": args.host_name, "drink": args.host_drink, "interest": args.host_interest}
    guests = []
    given_seats = []  # boxes of seats already offered, so no two guests get the same one

    for g in range(1, args.guests + 1):
        print(f"\n======== GUEST {g} ========")
        frame, person = wait_for_guest(door, per, view)
        info = get_guest_info()

        t = time.time()
        info["looks"] = llm.describe_person(crop(frame, person.box))
        print(f"   [appearance in {time.time() - t:.1f}s] {info['looks']}")
        view.show(f"guest{g}", crop(frame, person.box), save=True)

        say(f"Please follow me to the living room, {info['name']}.")
        if room is door and door.image is None:
            input("   >> Point the camera at the living room, then press Enter...")

        room_frame, seat, _ = find_seat(room, per, view, given_seats)
        if seat:
            given_seats.append(seat.box)
            say(f"{info['name']}, please take the {seat.seat} "
                f"{where(seat.box, room_frame.shape[1])}.")
        else:
            say(f"I'm sorry {info['name']}, I can't see a free seat. Please wait here a moment.")

        say(llm.common_ground(host, info))
        for prev in guests:
            say(f"{info['name']}, this is {prev['name']}{likes(prev)}. "
                f"{prev['looks'].get('summary', '')}")
        guests.append(info)

        if g < args.guests and door.image is None:
            input("   >> Point the camera back at the door for the next guest, then press Enter...")

    say("All guests are seated. Enjoy the party!")
    if view.gui:
        cv2.destroyAllWindows()


TTS = None
ASR = None

if __name__ == "__main__":
    main()

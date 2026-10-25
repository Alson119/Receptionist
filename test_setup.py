"""Quick check that Ollama (in WSL), Gemma 4 and YOLO-World all work.

  python test_setup.py
"""
import time

import cv2

import llm
from perception import Perception, crop, draw, largest_person, seat_slots

print("1) Ollama")
print("   installed models:", llm.list_models())
print("   using:", llm.resolve_model())

print("2) Language: extracting guest info")
t = time.time()
print("  ", llm.extract_guest_info("Hey, I'm Mei Ling, I'll have an iced lemon tea, "
                                    "and I'm really into badminton"),
      f"({time.time() - t:.1f}s)")

print("3) YOLO-World (downloads weights + CLIP on first run)")
from ultralytics.utils import ASSETS  # noqa: E402

img = cv2.imread(str(ASSETS / "bus.jpg"))
per = Perception()
t = time.time()
dets = per.detect(img)
print(f"   {len(dets)} detections in {time.time() - t:.2f}s:",
      [(d.label, round(d.conf, 2)) for d in dets])
print("   seat slots:", seat_slots(dets))
cv2.imwrite("test_detections.jpg", draw(img, dets))

print("4) Vision: describing the largest person")
p = largest_person(dets)
if p:
    t = time.time()
    print("  ", llm.describe_person(crop(img, p.box)), f"({time.time() - t:.1f}s)")

print("\nAll good. Annotated image saved to test_detections.jpg")

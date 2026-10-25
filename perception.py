"""Open-vocabulary detection with YOLO-World, plus free-seat reasoning."""
from dataclasses import dataclass

import cv2

PERSON = "person"
SEAT_CLASSES = ["chair", "armchair", "sofa", "couch", "stool", "bench"]
MULTI_SEATS = {"sofa", "couch", "bench"}
CLASSES = [PERSON] + SEAT_CLASSES


@dataclass
class Det:
    label: str
    box: tuple  # x1, y1, x2, y2
    conf: float

    @property
    def w(self):
        return self.box[2] - self.box[0]

    @property
    def h(self):
        return self.box[3] - self.box[1]

    @property
    def cx(self):
        return (self.box[0] + self.box[2]) / 2


@dataclass
class Slot:
    seat: str
    box: tuple
    occupied: bool


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def overlap_frac(slot, other):
    """Fraction of the slot's area covered by `other`."""
    x1, y1 = max(slot[0], other[0]), max(slot[1], other[1])
    x2, y2 = min(slot[2], other[2]), min(slot[3], other[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (slot[2] - slot[0]) * (slot[3] - slot[1])
    return inter / area if area > 0 else 0.0


def dedupe_seats(seats, thr=0.6):
    """YOLO-World often fires 'chair' and 'armchair' on the same object."""
    keep = []
    for s in sorted(seats, key=lambda d: -d.conf):
        if all(iou(s.box, k.box) < thr for k in keep):
            keep.append(s)
    return keep


def seat_slots(dets):
    """Split sofas into per-person slots and mark each slot occupied or free."""
    people = [d for d in dets if d.label == PERSON]
    seats = dedupe_seats([d for d in dets if d.label in SEAT_CLASSES])
    slots = []
    for s in seats:
        x1, y1, x2, y2 = s.box
        n = 1
        if s.label in MULTI_SEATS and s.h > 0:
            n = max(2, min(4, round(s.w / (s.h * 0.9))))
        step = (x2 - x1) / n
        for i in range(n):
            box = (x1 + i * step, y1, x1 + (i + 1) * step, y2)
            occupied = False
            for p in people:
                # a seated person's hips sit roughly 60% down their box
                hx, hy = p.cx, p.box[1] + 0.6 * p.h
                hip_inside = box[0] <= hx <= box[2] and box[1] - 0.2 * s.h <= hy <= box[3]
                if hip_inside or overlap_frac(box, p.box) > 0.4:
                    occupied = True
                    break
            slots.append(Slot(s.label, box, occupied))
    return slots


def where(box, frame_w):
    cx = (box[0] + box[2]) / 2
    if cx < frame_w / 3:
        return "on your left"
    if cx > 2 * frame_w / 3:
        return "on your right"
    return "in front of you"


class Perception:
    def __init__(self, weights="weights/yolov8s-worldv2.pt", conf=0.2, device=None):
        from ultralytics import YOLOWorld  # imported here so seat logic is testable without torch

        self.model = YOLOWorld(weights)
        self.model.set_classes(CLASSES)
        self.conf = conf
        self.device = device

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        ):
            out.append(Det(r.names[int(cls)], tuple(float(v) for v in box), float(conf)))
        return out


def largest_person(dets):
    people = [d for d in dets if d.label == PERSON]
    return max(people, key=lambda d: d.w * d.h) if people else None


def crop(frame, box, pad=0.05):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return frame[y1:y2, x1:x2].copy()


def draw(frame, dets, slots=(), chosen=None):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        colour = (255, 160, 0) if d.label == PERSON else (180, 180, 180)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    for s in slots:
        x1, y1, x2, y2 = map(int, s.box)
        colour = (0, 0, 255) if s.occupied else (0, 200, 0)
        thick = 4 if s is chosen else 1
        cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), colour, thick)
        cv2.putText(img, "TAKEN" if s.occupied else "FREE", (x1 + 6, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return img

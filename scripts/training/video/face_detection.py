from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float | None
    frame_index: int
    timestamp_seconds: float | None


class FaceDetector:
    def detect(self, rgb, frame_index: int, timestamp_seconds=None) -> list[FaceDetection]:
        raise NotImplementedError


class HaarFaceDetector(FaceDetector):
    """Local OpenCV Haar fallback; weaker than modern learned detectors."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(24, 24), max_detect_dimension: int = 640):
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(cascade)
        if self.classifier.empty():
            raise RuntimeError("OpenCV Haar cascade unavailable")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.max_detect_dimension = max_detect_dimension

    def detect(self, rgb, frame_index: int, timestamp_seconds=None):
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        longest = max(height, width)
        scale = 1.0
        if self.max_detect_dimension and longest > self.max_detect_dimension:
            scale = self.max_detect_dimension / float(longest)
            target_w = max(1, round(width * scale))
            target_h = max(1, round(height * scale))
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        boxes = self.classifier.detectMultiScale(
            gray,
            self.scale_factor,
            self.min_neighbors,
            minSize=self.min_size,
        )
        detections: list[FaceDetection] = []
        inv_scale = 1.0 / scale
        for bx, by, bw, bh in boxes:
            x = max(0, min(width - 1, int(round(bx * inv_scale))))
            y = max(0, min(height - 1, int(round(by * inv_scale))))
            w = max(1, min(width - x, int(round(bw * inv_scale))))
            h = max(1, min(height - y, int(round(bh * inv_scale))))
            if valid_bbox((x, y, w, h), width, height):
                detections.append(FaceDetection((x, y, w, h), None, frame_index, timestamp_seconds))
        return detections


def valid_bbox(box, width, height):
    x, y, w, h = map(int, box)
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x < width and y < height and x + w <= width and y + h <= height

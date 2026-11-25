"""OpenCV Haar face/eye detection and conservative crop alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    crop_rgb: np.ndarray
    alignment_status: str

    def as_dict(self) -> dict[str, Any]:
        x, y, width, height = self.box
        return {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "alignment_status": self.alignment_status,
        }


class FaceDetector:
    """Uses the cascades bundled with OpenCV; no network weights are fetched."""

    def __init__(self, min_size: int = 32, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        root = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(str(root + "haarcascade_frontalface_default.xml"))
        self.eyes = cv2.CascadeClassifier(str(root + "haarcascade_eye.xml"))
        if self.face.empty() or self.eyes.empty():
            raise RuntimeError("OpenCV bundled Haar cascades could not be loaded")
        self.min_size = max(16, int(min_size))
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)

    def detect(self, rgb: np.ndarray, margin_ratio: float = 0.15) -> list[FaceResult]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Face detection expects an RGB image")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.face.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        results: list[FaceResult] = []
        image_height, image_width = rgb.shape[:2]
        for x, y, width, height in boxes:
            margin_x = int(width * margin_ratio)
            margin_y = int(height * margin_ratio)
            left = max(0, int(x) - margin_x)
            top = max(0, int(y) - margin_y)
            right = min(image_width, int(x + width + margin_x))
            bottom = min(image_height, int(y + height + margin_y))
            crop = rgb[top:bottom, left:right].copy()
            roi = gray[int(y):int(y + height), int(x):int(x + width)]
            upper = roi[: max(1, int(roi.shape[0] * 0.62)), :]
            eye_boxes = self.eyes.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4, minSize=(8, 8))
            eye_centers = sorted(
                [(int(ex + ew / 2), int(ey + eh / 2)) for ex, ey, ew, eh in eye_boxes],
                key=lambda point: point[0],
            )
            aligned = False
            if len(eye_centers) >= 2:
                # Select the widest plausible pair in the upper face region.
                pair = max(
                    ((a, b) for i, a in enumerate(eye_centers) for b in eye_centers[i + 1:]),
                    key=lambda item: item[1][0] - item[0][0],
                )
                (x1, y1), (x2, y2) = pair
                if x2 > x1:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    center = ((x1 + x2) / 2 + (int(x) - left), (y1 + y2) / 2 + (int(y) - top))
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    crop = cv2.warpAffine(
                        crop,
                        matrix,
                        (crop.shape[1], crop.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    aligned = True
            results.append(
                FaceResult(
                    box=(int(x), int(y), int(width), int(height)),
                    crop_rgb=crop,
                    alignment_status="eyes_aligned" if aligned else "crop_only",
                )
            )
        return results

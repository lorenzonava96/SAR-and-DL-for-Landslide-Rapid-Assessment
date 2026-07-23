from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class Window:
    x0: int; y0: int; x1: int; y1: int


def sliding_windows(image: np.ndarray, size: int, step: int) -> Iterator[tuple[Window, np.ndarray]]:
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    height, width = image.shape[:2]
    if height < size or width < size:
        return
    for y in range(0, height - size + 1, step):
        for x in range(0, width - size + 1, step):
            yield Window(x, y, x + size, y + size), image[y:y + size, x:x + size]


def non_max_suppression(boxes: np.ndarray, overlap_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.int32)
    boxes = boxes.astype(float, copy=False)
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)
    selected: list[int] = []
    while len(indices):
        i = indices[-1]
        selected.append(int(i))
        remaining = indices[:-1]
        xx1 = np.maximum(x1[i], x1[remaining]); yy1 = np.maximum(y1[i], y1[remaining])
        xx2 = np.minimum(x2[i], x2[remaining]); yy2 = np.minimum(y2[i], y2[remaining])
        width = np.maximum(0, xx2 - xx1 + 1); height = np.maximum(0, yy2 - yy1 + 1)
        overlap = (width * height) / area[remaining]
        indices = remaining[overlap <= overlap_threshold]
    return boxes[selected].astype(np.int32)

import numpy as np
from sort import Sort

class PlayerTracker:
    def __init__(self):
        self.tracker = Sort()

    def track(self, detections):
        if len(detections) == 0:
            return []
        dets = np.array(detections)
        tracked = self.tracker.update(dets)
        return tracked  # Each row: [x1, y1, x2, y2, id]

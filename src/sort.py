import numpy as np

class Sort:
    def __init__(self):
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        results = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            results.append([x1, y1, x2, y2, self.next_id])
            self.next_id += 1

        return np.array(results)
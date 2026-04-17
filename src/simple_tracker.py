#!/usr/bin/env python3
"""Simple IoU-based tracker for demo purposes."""
from collections import deque

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = boxAArea + boxBArea - inter
    return inter / union if union > 0 else 0

class Track:
    def __init__(self, tid, bbox):
        self.id = tid
        self.bbox = bbox
        self.hits = 1
        self.missed = 0
        self.trace = deque(maxlen=50)
        self.trace.append(bbox)

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missed=5):
        self.next_id = 1
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2,score,cls]
        matches = []
        if len(self.tracks) == 0:
            for det in detections:
                tr = Track(self.next_id, det[:4])
                self.next_id += 1
                self.tracks.append(tr)
            return [{'id':t.id, 'bbox':t.bbox} for t in self.tracks]
        # compute IoU matrix
        iou_matrix = [[iou(tr.bbox, det[:4]) for det in detections] for tr in self.tracks]
        matched_t = set(); matched_d = set()
        # greedy matching
        for t_idx, row in enumerate(iou_matrix):
            if len(row) == 0: continue
            best_d = max(range(len(row)), key=lambda x: row[x])
            if row[best_d] >= self.iou_threshold and best_d not in matched_d and t_idx not in matched_t:
                matched_t.add(t_idx); matched_d.add(best_d)
                tr = self.tracks[t_idx]
                det = detections[best_d]
                tr.bbox = det[:4]
                tr.hits += 1
                tr.missed = 0
                tr.trace.append(det[:4])
        # handle unmatched tracks
        for idx in reversed(range(len(self.tracks))):
            if idx not in matched_t:
                self.tracks[idx].missed += 1
                if self.tracks[idx].missed > self.max_missed:
                    self.tracks.pop(idx)
        # create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_d:
                tr = Track(self.next_id, det[:4])
                self.next_id += 1
                self.tracks.append(tr)
        return [{'id':t.id, 'bbox':t.bbox} for t in self.tracks]

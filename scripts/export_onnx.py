#!/usr/bin/env python3
"""Export YOLO model to ONNX for deployment."""
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='yolov8n.pt')
parser.add_argument('--out', default='models/yolov8.onnx')
args = parser.parse_args()
model = YOLO(args.weights)
print('Exporting to ONNX... this may take a while')
model.export(format='onnx', imgsz=640, simplify=True)
print('Export complete')

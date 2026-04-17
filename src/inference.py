#!/usr/bin/env python3
"""Run detection over an image sequence/folder and apply simple IoU tracker, saving visualized results."""
import argparse
import os
import cv2
import glob
from ultralytics import YOLO
from src.simple_tracker import SimpleTracker


def yolo_txt_to_xyxy(txt_path, img_w, img_h):
    dets = []
    if not os.path.exists(txt_path):
        return dets
    with open(txt_path,'r') as f:
        for line in f:
            parts=line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0]); xc=float(parts[1]); yc=float(parts[2]); w=float(parts[3]); h=float(parts[4])
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            dets.append([x1,y1,x2,y2,1.0,cls])
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolov8n.pt')
    parser.add_argument('--source', default='data/toy/images/val')
    parser.add_argument('--save-dir', default='runs/detect/track')
    args = parser.parse_args()
    model = YOLO(args.weights)
    os.makedirs(args.save_dir, exist_ok=True)
    # run YOLO predict and save txt labels
    print('Running YOLO detection (this will create runs/detect/predict outputs)')
    model.predict(source=args.source, save=True, save_txt=True)
    # find predicted images and labels
    predict_dir = 'runs/detect/predict'
    images = sorted(glob.glob(os.path.join(args.source, '*')))
    tracker = SimpleTracker()
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        h,w = img.shape[:2]
        txt_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        txt_path = os.path.join(predict_dir, 'labels', txt_name)
        dets = yolo_txt_to_xyxy(txt_path, w, h)
        tracks = tracker.update(dets)
        # draw
        for det in dets:
            x1,y1,x2,y2,_,cls = det
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        for tr in tracks:
            x1,y1,x2,y2 = tr['bbox']
            tid = tr['id']
            cv2.putText(img, f'ID:{tid}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        out_path = os.path.join(args.save_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
    print('Saved tracked visualizations to', args.save_dir)

if __name__ == '__main__':
    main()

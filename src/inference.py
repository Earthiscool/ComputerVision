#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/train/weights/best.pt')
    parser.add_argument('--source', default='data/toy/images/val')
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed. Install with: pip install ultralytics')
        return
    model = YOLO(args.weights)
    results = model.predict(source=args.source, save=True)
    print('Saved predictions. See ./runs/detect/predict')

if __name__ == '__main__':
    main()

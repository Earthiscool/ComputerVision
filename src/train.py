"""Train script using Ultralytics YOLOv8 API with sensible defaults for a tiny dataset."""
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/toy/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--model', default='yolov8n.pt')
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics not installed. Install with: pip install ultralytics')
        return
    model = YOLO(args.model)
    print('Starting short training on', args.data)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=4)

if __name__ == '__main__':
    main()

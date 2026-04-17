#!/usr/bin/env python3
"""Generate a tiny synthetic dataset (images + YOLO labels) for quick training runs."""
import os
from PIL import Image, ImageDraw
import random

def make_image(path, w=640, h=480, n_boxes=3):
    img = Image.new('RGB', (w, h), (255,255,255))
    draw = ImageDraw.Draw(img)
    labels = []
    for i in range(n_boxes):
        x1 = random.randint(0, w-60)
        y1 = random.randint(0, h-60)
        x2 = x1 + random.randint(20, 100)
        y2 = y1 + random.randint(20, 100)
        color = tuple(random.randint(0,255) for _ in range(3))
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        # YOLO format: class x_center y_center width height (normalized)
        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        cls = random.randint(0,1)  # two classes: 0 or 1
        labels.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    img.save(path)
    return labels


def ensure_dirs(base):
    for split in ('train','val'):
        os.makedirs(os.path.join(base,'images',split), exist_ok=True)
        os.makedirs(os.path.join(base,'labels',split), exist_ok=True)


def main():
    base = 'data/toy'
    ensure_dirs(base)
    # create 20 train, 5 val
    for i in range(20):
        img_path = os.path.join(base,'images','train', f"img_{i:03d}.jpg")
        labels = make_image(img_path)
        with open(os.path.join(base,'labels','train', f"img_{i:03d}.txt"), 'w') as f:
            f.write('\n'.join(labels))
    for i in range(5):
        img_path = os.path.join(base,'images','val', f"img_val_{i:03d}.jpg")
        labels = make_image(img_path)
        with open(os.path.join(base,'labels','val', f"img_val_{i:03d}.txt"), 'w') as f:
            f.write('\n'.join(labels))
    # create data.yaml for Ultralytics
    data_yaml = {
        'train': os.path.abspath(os.path.join(base,'images','train')),
        'val': os.path.abspath(os.path.join(base,'images','val')),
        'nc': 2,
        'names': ['class0','class1']
    }
    import yaml
    with open(os.path.join(base,'data.yaml'), 'w') as f:
        yaml.dump(data_yaml)
    print('Created toy dataset at', base)

if __name__ == '__main__':
    main()

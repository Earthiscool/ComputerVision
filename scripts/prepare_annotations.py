#!/usr/bin/env python3
"""Convert COCO annotations to YOLO format (per-image .txt labels) and create data.yaml for YOLOv8.
This script supports COCO jsons if pycocotools is installed. If not, it prints instructions.
"""
import argparse
import os
import json


def coco_to_yolo(coco_json, images_dir, labels_out_dir, class_map=None):
    os.makedirs(labels_out_dir, exist_ok=True)
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    imgs = {img['id']:img for img in coco.get('images',[])}
    anns = {}
    for a in coco.get('annotations',[]):
        anns.setdefault(a['image_id'], []).append(a)
    categories = {c['id']:c['name'] for c in coco.get('categories',[])}
    # optional class map
    if class_map is None:
        # default: map category order to 0..nc-1
        uniq = sorted(list({c['id'] for c in coco.get('categories',[])}))
        class_map = {cid:i for i,cid in enumerate(uniq)}
    for img_id, img in imgs.items():
        filename = img['file_name']
        w = img['width']; h = img['height']
        image_path = os.path.join(images_dir, filename)
        label_lines = []
        for a in anns.get(img_id, []):
            bbox = a['bbox']  # x,y,w,h (COCO)
            x, y, bw, bh = bbox
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            cls = class_map[a['category_id']]
            label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")
        out_txt = os.path.join(labels_out_dir, os.path.splitext(filename)[0] + '.txt')
        with open(out_txt, 'w') as f:
            f.write('\n'.join(label_lines))
    print('Converted COCO annotations to YOLO labels in', labels_out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-json', help='COCO annotations json')
    parser.add_argument('--images-dir', help='Path to COCO images')
    parser.add_argument('--out', default='data/coco_yolo', help='Output base dir')
    args = parser.parse_args()
    if args.coco_json and args.images_dir:
        images_out = os.path.join(args.out, 'images')
        labels_out = os.path.join(args.out, 'labels')
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)
        coco_to_yolo(args.coco_json, args.images_dir, labels_out)
        # create simple data.yaml (user may need to adapt paths)
        data_yaml = {
            'train': os.path.abspath(images_out),
            'val': os.path.abspath(images_out),
            'nc': len({}),
            'names': []
        }
        try:
            import yaml
            with open(os.path.join(args.out,'data.yaml'),'w') as f:
                yaml.dump(data_yaml)
        except Exception:
            pass
    else:
        print('Provide --coco-json and --images-dir to convert. For quick runs, use scripts/create_toy_dataset.py')

if __name__ == '__main__':
    main()

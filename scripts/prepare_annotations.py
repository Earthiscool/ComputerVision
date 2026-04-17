#!/usr/bin/env python3
"""Annotation conversion and dataset split utilities (stub)."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-dir', default='data/coco')
    args = parser.parse_args()
    print('Stub: convert COCO annotations to YOLO format or desired layout')

if __name__ == '__main__':
    main()

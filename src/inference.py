"""Inference pipeline: detection -> tracking -> visualization (stub)."""
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='models/best.pt')
    args = parser.parse_args()
    print('Stub: run detection and tracking on video or camera')

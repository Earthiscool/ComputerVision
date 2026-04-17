#!/usr/bin/env python3
"""Evaluate tracking results using motmetrics if ground truth is provided in MOTChallenge format.
This is a helper stub — adapt paths to your GT and hypothesis files.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt', help='Ground truth text directory (MOT format)')
parser.add_argument('--hyp', help='Hypothesis text directory (MOT format)')
args = parser.parse_args()

try:
    import motmetrics as mm
except Exception:
    print('motmetrics not installed. Install with: pip install motmetrics')
    exit(0)

print('This script is a helper. Use motmetrics to load GT/hyp and compute metrics. See motmetrics docs for examples.')

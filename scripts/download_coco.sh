#!/usr/bin/env bash
set -e
mkdir -p ../data/coco
cd ../data/coco
echo "Downloading COCO 2017 annotations (subset as needed)"
[ -f annotations_trainval2017.zip ] || wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# note: train2017 and val2017 are large; consider downloading subset or using AWS/gs:// mirrors
echo "Downloaded (or found) COCO annotation zip. Unzip and subset as needed."

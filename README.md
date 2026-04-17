# ComputerVision

End-to-end object detection + multi-object tracking portfolio project.

Quickstart:
1. python3 -m pip install -r requirements.txt
2. python scripts/create_toy_dataset.py
3. python src/train.py --data data/toy/data.yaml --epochs 3
4. python src/inference.py --weights yolov8n.pt --source data/toy/images/val

Demo:
- Run demo/streamlit_app.py with streamlit: streamlit run demo/streamlit_app.py

Exports:
- Use scripts/export_onnx.py to export to ONNX (requires ultralytics export support)

CI:
- Basic CI runs pytest on push and PRs.

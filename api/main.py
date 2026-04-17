from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

app = FastAPI(title='CV Demo API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        try:
            from ultralytics import YOLO
            MODEL = YOLO('yolov8n.pt')
        except Exception as e:
            MODEL = e
    return MODEL

@app.get('/health')
async def health():
    return {'status':'ok'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    model = load_model()
    if isinstance(model, Exception):
        return JSONResponse(status_code=500, content={'error': str(model)})
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        arr = np.array(img)
    except Exception as e:
        return JSONResponse(status_code=400, content={'error':'invalid image', 'detail': str(e)})
    # run prediction
    results = model.predict(source=arr, conf=0.25, imgsz=640)
    out = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for b in boxes:
            # b may be a ultralytics object; access attributes carefully
            try:
                xyxy = b.xyxy[0].tolist()
            except Exception:
                xyxy = [float(x) for x in b.xyxy]
            try:
                conf = float(b.conf[0])
            except Exception:
                conf = None
            try:
                cls = int(b.cls[0])
            except Exception:
                cls = None
            out.append({'xyxy': [float(x) for x in xyxy], 'conf': conf, 'class': cls})
    return {'predictions': out}

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import os

app = FastAPI(title='ONNX CV Demo API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join('models','model.onnx')
SESSION = None

def load_session():
    global SESSION
    if SESSION is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError('ONNX model not found at '+MODEL_PATH)
        SESSION = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return SESSION

def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.size  # PIL: (w,h)
    w,h = shape
    new_w, new_h = new_shape
    scale = min(new_w/w, new_h/h)
    nw, nh = int(w*scale), int(h*scale)
    im_resized = im.resize((nw, nh), Image.BILINEAR)
    new_im = Image.new('RGB', (new_w,new_h), color)
    paste_x = (new_w - nw) // 2
    paste_y = (new_h - nh) // 2
    new_im.paste(im_resized, (paste_x, paste_y))
    return new_im, scale, paste_x, paste_y


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_w, orig_h = img.size
    img_letter, scale, pad_x, pad_y = letterbox(img, new_shape=(640,640))
    arr = np.array(img_letter).astype(np.float32) / 255.0
    # HWC to CHW
    arr = arr.transpose(2,0,1)
    arr = np.expand_dims(arr, axis=0)
    return arr, (orig_w, orig_h), scale, pad_x, pad_y


def xywh_to_xyxy(x, y, w, h):
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    # prediction: (N,85) -> x,y,w,h,obj_conf,cls_scores...
    boxes = []
    if prediction.size == 0:
        return []
    scores = prediction[:,4:5] * prediction[:,5:]
    classes = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), classes]
    mask = confidences > conf_thres
    if not mask.any():
        return []
    filtered = prediction[mask]
    classes = classes[mask]
    confidences = confidences[mask]
    # boxes in xywh (normalized or absolute depending on model)
    detections = []
    for i, det in enumerate(filtered):
        x, y, w, h = det[:4]
        detections.append([x, y, w, h, float(confidences[i]), int(classes[i])])
    # convert to xyxy absolute (we will handle scaling outside)
    # simple greedy NMS
    out = []
    dets = np.array(detections)
    if dets.size == 0:
        return out
    x1 = dets[:,0] - dets[:,2]/2
    y1 = dets[:,1] - dets[:,3]/2
    x2 = dets[:,0] + dets[:,2]/2
    y2 = dets[:,1] + dets[:,3]/2
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-dets[:,4])
    while order.size > 0:
        i = order[0]
        out.append([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(dets[i,4]), int(dets[i,5])])
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[rest] - inter)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return out

@app.post('/onnx_predict')
async def onnx_predict(file: UploadFile = File(...)):
    try:
        sess = load_session()
    except FileNotFoundError as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    contents = await file.read()
    try:
        img_arr, (orig_w, orig_h), scale, pad_x, pad_y = preprocess(contents)
    except Exception as e:
        return JSONResponse(status_code=400, content={'error': 'invalid image', 'detail': str(e)})
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: img_arr})
    # Attempt to find detection tensor
    # Commonly outputs[0] has shape (1,N,85)
    pred = None
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[2] >= 5:
            pred = out[0]
            break
    if pred is None:
        return JSONResponse(status_code=500, content={'error': 'unexpected ONNX output format'})
    dets = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    results = []
    for x1,y1,x2,y2,conf,cls in dets:
        # x1..y2 are relative to 640x640 letterboxed image; convert back to original image coords
        # undo padding and scale
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        # clamp
        x1 = max(0.0, min(x1, orig_w))
        y1 = max(0.0, min(y1, orig_h))
        x2 = max(0.0, min(x2, orig_w))
        y2 = max(0.0, min(y2, orig_h))
        results.append({'xyxy':[x1,y1,x2,y2],'conf':conf,'class':cls})
    return {'predictions': results}

# health route
@app.get('/health')
async def health():
    return {'status':'ok'}

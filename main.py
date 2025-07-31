from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel

# Registrar Conv como safe global para torch.load
Conv = None
try:
    from ultralytics.nn.modules.conv import Conv
except ImportError:
    try:
        from ultralytics.nn.common import Conv
    except ImportError:
        try:
            from ultralytics.nn.modules import Conv
        except ImportError:
            try:
                from ultralytics.nn import Conv
            except ImportError:
                Conv = None

if Conv is None:
    raise ImportError("No se pudo importar Conv desde ninguna ruta conocida de Ultralytics.")

safe_globals = [
    DetectionModel,
    torch.nn.modules.container.Sequential,
    Conv,
]

torch.serialization.add_safe_globals(safe_globals)

app = FastAPI()
model = YOLO("best.pt")  # El modelo está en la raíz

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class": results[0].names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "box": [x1, y1, x2, y2]
        })
    return JSONResponse(detections)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
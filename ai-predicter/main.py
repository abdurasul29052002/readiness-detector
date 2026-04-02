"""
FastAPI AI server — individual o'quvchi rasmlarini classification qilish.

3 xil model qo'llab-quvvatlanadi:
  - YOLOv8 detect  (best.pt)
  - YOLOv8 classify (best_cls.pt)
  - ResNet50 PyTorch (best_resnet50.pt)

Endpoints:
  POST /classify       — bitta crop rasmni classification qiladi
  POST /classify/batch — bir nechta crop rasmlarni batch classification qiladi
  GET  /health         — server holati
  GET  /models         — mavjud modellar ro'yxati
  POST /models/switch  — faol modelni almashtirish
  POST /models/upload  — yangi model yuklash
"""

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent / "models"
METADATA_FILE = MODELS_DIR / "metadata.json"

CLASS_NAMES = {
    0: "hand-raising",
    1: "read",
    2: "write",
    3: "discuss",
    4: "bow-head",
    5: "turn-head",
    6: "standing",
}

# ResNet50 klass tartibi (ImageFolder alifbo bo'yicha)
RESNET_CLASS_NAMES = [
    "bow-head", "discuss", "hand-raising", "read",
    "standing", "turn-head", "write",
]

# ResNet50 index -> API class_id mapping
RESNET_TO_API = {
    0: 4,  # bow-head -> 4
    1: 3,  # discuss -> 3
    2: 0,  # hand-raising -> 0
    3: 1,  # read -> 1
    4: 6,  # standing -> 6
    5: 5,  # turn-head -> 5
    6: 2,  # write -> 2
}

# Diqqatli (attentive) va chalg'igan (distracted) guruhlari
ATTENTIVE_CLASSES = {0, 1, 2}
DISTRACTED_CLASSES = {3, 4, 5, 6}

# ResNet50 uchun inference transform
RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Student Behavior Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model registry
models_registry: dict[str, dict] = {}
active_model_version: str | None = None
model = None  # YOLO yoki nn.Module
model_task: str | None = None  # "classify", "detect", "resnet50"


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata():
    data = {}
    for ver, info in models_registry.items():
        data[ver] = info["metadata"]
    METADATA_FILE.write_text(json.dumps(data, indent=2, default=str))


def _detect_model_type(path: Path) -> str:
    """Model tipini aniqlash: fayl nomiga va formatiga qarab."""
    name = path.stem.lower()
    if "resnet" in name:
        return "resnet50"
    # YOLO model ekanligini tekshirish
    try:
        yolo_model = YOLO(str(path))
        task = yolo_model.task or "detect"
        return task  # "classify" yoki "detect"
    except Exception:
        return "unknown"


def _load_resnet50(path: Path) -> nn.Module:
    """ResNet50 modelni yuklash."""
    resnet = models.resnet50()
    resnet.fc = nn.Linear(resnet.fc.in_features, 7)
    state = torch.load(str(path), map_location=DEVICE, weights_only=True)
    resnet.load_state_dict(state)
    resnet.to(DEVICE)
    resnet.eval()
    return resnet


def _activate_model(version: str):
    """Modelni yuklash va faollashtirish."""
    global model, active_model_version, model_task

    entry = models_registry[version]

    if entry["model"] is None:
        path = entry["path"]
        detected_type = entry["metadata"].get("task") or _detect_model_type(path)
        entry["metadata"]["task"] = detected_type

        if detected_type == "resnet50":
            entry["model"] = _load_resnet50(path)
        else:
            entry["model"] = YOLO(str(path))
            detected_type = entry["model"].task or "detect"
            entry["metadata"]["task"] = detected_type

    model = entry["model"]
    model_task = entry["metadata"]["task"]
    active_model_version = version
    print(f"Model activated: {version} (task={model_task})")


@app.on_event("startup")
def load_models():
    metadata = load_metadata()

    for pt_file in sorted(MODELS_DIR.glob("*.pt")):
        version = pt_file.stem
        meta = metadata.get(version, {})

        # Fayl nomidan model tipini oldindan aniqlash (lazy load uchun)
        task_hint = meta.get("task")
        if not task_hint and "resnet" in version.lower():
            task_hint = "resnet50"

        models_registry[version] = {
            "path": pt_file,
            "model": None,
            "metadata": {
                "version": version,
                "training_date": meta.get("training_date"),
                "accuracy": meta.get("accuracy"),
                "description": meta.get("description"),
                "filename": pt_file.name,
                "task": task_hint,
            },
        }

    # Default: best_cls, agar yo'q bo'lsa birinchi topilgan model
    default_version = "best_cls" if "best_cls" in models_registry else next(iter(models_registry), None)
    if default_version:
        _activate_model(default_version)
    else:
        print("WARNING: No models found in models/ directory")


def run_resnet50_classification(pil_image: Image.Image, confidence: float) -> dict:
    """ResNet50 bilan bitta rasmni classify qilish."""
    input_tensor = RESNET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    resnet_cls_id = int(probs.argmax())
    conf = float(probs[resnet_cls_id])

    if conf < confidence:
        return {
            "class_id": -1,
            "class_name": "unknown",
            "confidence": round(conf, 3),
            "group": "unknown",
        }

    api_cls_id = RESNET_TO_API[resnet_cls_id]
    cls_name = CLASS_NAMES.get(api_cls_id, "unknown")
    group = "attentive" if api_cls_id in ATTENTIVE_CLASSES else "distracted"

    return {
        "class_id": api_cls_id,
        "class_name": cls_name,
        "confidence": round(conf, 3),
        "group": group,
    }


def run_classification(img_array, confidence: float) -> dict:
    """YOLOv8 bilan bitta crop rasmni classification qiladi."""
    results = model.predict(img_array, conf=confidence, verbose=False)
    result = results[0]

    probs = result.probs
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)

    cls_name = CLASS_NAMES.get(cls_id, "unknown")
    group = "attentive" if cls_id in ATTENTIVE_CLASSES else "distracted"

    return {
        "class_id": cls_id,
        "class_name": cls_name,
        "confidence": round(conf, 3),
        "group": group,
    }


def run_detection(img_array, confidence: float) -> list[dict]:
    """Rasmda detection — har bir topilgan o'quvchi uchun natija."""
    results = model.predict(img_array, conf=confidence, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, "unknown"),
            "confidence": round(conf, 3),
            "group": "attentive" if cls_id in ATTENTIVE_CLASSES else "distracted",
            "bbox": {
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            },
        })
    return detections


def make_summary(results: list) -> dict:
    """Natijalaridan summary hisoblash."""
    attentive_count = sum(1 for r in results if r["group"] == "attentive")
    distracted_count = sum(1 for r in results if r["group"] == "distracted")
    total = len(results)
    return {
        "total": total,
        "attentive": attentive_count,
        "distracted": distracted_count,
        "attentive_percent": round(attentive_count / total * 100, 1) if total > 0 else 0,
        "distracted_percent": round(distracted_count / total * 100, 1) if total > 0 else 0,
    }


@app.post("/classify")
async def classify_single(
    file: UploadFile = File(...),
    confidence: float = 0.5,
):
    """Bitta rasmni predict qiladi — model tipiga qarab classify yoki detect."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if model_task == "resnet50":
        return run_resnet50_classification(image, confidence)
    elif model_task == "classify":
        img_array = np.array(image)
        return run_classification(img_array, confidence)
    else:
        img_array = np.array(image)
        detections = run_detection(img_array, confidence)
        return {"detections": detections, "summary": make_summary(detections)}


@app.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    confidence: float = 0.5,
):
    """Bir nechta rasmlarni batch predict qiladi — model tipiga qarab."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    results = []
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if model_task == "resnet50":
            results.append(run_resnet50_classification(image, confidence))
        elif model_task == "classify":
            img_array = np.array(image)
            results.append(run_classification(img_array, confidence))
        else:
            img_array = np.array(image)
            detections = run_detection(img_array, confidence)
            results.extend(detections)

    return {
        "results": results,
        "summary": make_summary(results),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_task": model_task,
        "active_model_version": active_model_version,
        "available_models": len(models_registry),
    }


# --- Model Versioning ---

@app.get("/models")
def list_models_endpoint():
    return {
        "active_version": active_model_version,
        "models": [
            {
                "version": ver,
                "filename": info["metadata"]["filename"],
                "loaded": info["model"] is not None,
                "training_date": info["metadata"].get("training_date"),
                "accuracy": info["metadata"].get("accuracy"),
                "description": info["metadata"].get("description"),
                "task": info["metadata"].get("task"),
            }
            for ver, info in models_registry.items()
        ],
    }


@app.post("/models/switch")
def switch_model(version: str):
    if version not in models_registry:
        raise HTTPException(404, f"Model version '{version}' not found")

    _activate_model(version)
    return {"status": "switched", "active_version": active_model_version, "task": model_task}


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    version: str = Form(...),
    training_date: str = Form(None),
    accuracy: float = Form(None),
    description: str = Form(None),
):
    dest = MODELS_DIR / f"{version}.pt"
    content = await file.read()
    dest.write_bytes(content)

    task_hint = "resnet50" if "resnet" in version.lower() else None

    models_registry[version] = {
        "path": dest,
        "model": None,
        "metadata": {
            "version": version,
            "training_date": training_date,
            "accuracy": accuracy,
            "description": description,
            "filename": f"{version}.pt",
            "task": task_hint,
        },
    }
    save_metadata()

    return {"status": "uploaded", "version": version}

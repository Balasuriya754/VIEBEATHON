# ==========================================================
# vision.py — Vision + Math Integration (Final)
# ==========================================================

import os
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

# Import MathEngine for OCR-to-Math solving
try:
    from math_engine import MathEngine

    MATH_AVAILABLE = True
    math_engine = MathEngine()
except Exception as e:
    MATH_AVAILABLE = False
    math_engine = None
    print(f"⚠️ MathEngine unavailable: {e}")

# Windows Tesseract path
if os.name == "nt":
    tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_path):
        pytesseract.pytesseract.tesseract_cmd = tess_path

# Lazy imports
cv2 = None
np = None
YOLO = None
requests = None


def _init_dependencies():
    """Initialize optional dependencies"""
    global cv2, np, YOLO, requests
    try:
        import cv2 as cv2_module
        cv2 = cv2_module
    except ImportError:
        pass
    try:
        import numpy as numpy_module
        np = numpy_module
    except ImportError:
        pass
    try:
        from ultralytics import YOLO as YOLO_module
        YOLO = YOLO_module
    except ImportError:
        pass
    try:
        import requests as requests_module
        requests = requests_module
    except ImportError:
        pass


_init_dependencies()

# ==========================================================
# OCR FUNCTIONS
# ==========================================================
def _enhance_image(img: Image.Image, aggressive: bool = False) -> Image.Image:
    """Enhance image for better OCR"""
    if img.mode != "L":
        img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.6)
    if aggressive and np is not None:
        arr = np.array(img)
        threshold = max(0, arr.mean() - 10)
        arr = (arr > threshold).astype("uint8") * 255
        img = Image.fromarray(arr)
    return img


def ocr_image(image_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract text from image using OCR"""
    metadata = {"engine": "tesseract", "attempts": []}
    try:
        img = Image.open(image_path)
    except Exception as e:
        return f"[Error: {e}]", metadata

    best_text, best_len = "", 0
    for mode, config in [("normal", "--oem 3 --psm 6"), ("aggressive", "--oem 3 --psm 4")]:
        try:
            enhanced = _enhance_image(img, aggressive=(mode == "aggressive"))
            text = pytesseract.image_to_string(enhanced, lang="eng", config=config)
            metadata["attempts"].append({mode: len(text)})
            if len(text.strip()) > best_len:
                best_text, best_len = text.strip(), len(text.strip())
        except Exception as e:
            metadata["attempts"].append({mode: f"error: {e}"})

    # Fallback EasyOCR
    if best_len < 60:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(image_path)
            text = " ".join([r[1] for r in results if len(r) >= 2])
            if len(text.strip()) > best_len:
                best_text = text.strip()
                best_len = len(best_text)
        except Exception:
            pass

    text = re.sub(r"[ \t]+", " ", best_text.replace("\r", "\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Detect if it’s mathematical text
    is_math = bool(re.search(r"[0-9xXyY=+\-*/^]", text))
    if is_math and MATH_AVAILABLE:
        try:
            math_result = math_engine.solve_query(f"solve {text}")
            metadata["math_detected"] = True
            metadata["math_solution"] = math_result
        except Exception as e:
            metadata["math_error"] = str(e)

    return text or "[No readable text]", metadata


# ==========================================================
# OBJECT DETECTION
# ==========================================================
_yolo_model = None

def _load_yolo(model_name: str = "yolov8n.pt"):
    global _yolo_model
    if _yolo_model is None and YOLO is not None:
        try:
            _yolo_model = YOLO(model_name)
        except Exception:
            _yolo_model = None


def detect_objects(image_path: str, confidence: float = 0.25) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Detect objects using YOLOv8"""
    if YOLO is None:
        return [{"warning": "YOLO not available"}], None
    _load_yolo()
    if _yolo_model is None:
        return [{"warning": "YOLO model unavailable"}], None

    try:
        results = _yolo_model.predict(image_path, conf=confidence, verbose=False)
        detections, annotated_path = [], None
        result = results[0]
        for box in result.boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            bbox = box.xyxy.cpu().numpy()[0].tolist()
            detections.append({
                "label": result.names.get(class_id, str(class_id)),
                "conf": round(conf, 3),
                "bbox": bbox
            })
        if cv2 is not None and detections:
            img = cv2.imread(image_path)
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, det["label"], (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            annotated_path = str(Path(image_path).with_suffix(".detected.jpg"))
            cv2.imwrite(annotated_path, img)
        return detections, annotated_path
    except Exception as e:
        return [{"error": str(e)}], None


# ==========================================================
# LLAVA INTEGRATION (optional)
# ==========================================================
def llava_describe(image_path: str, prompt: str,
                   ollama_url: str = "http://127.0.0.1:11434/api/generate",
                   model: str = "llava:13b") -> Tuple[str, Dict[str, Any]]:
    """Describe image using LLaVA via Ollama"""
    if requests is None:
        return "[Error: requests unavailable]", {}
    try:
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
        payload = {"model": model, "prompt": prompt, "stream": False, "images": [b64_image]}
        r = requests.post(ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip(), {"model": model}
    except Exception as e:
        return f"[Error: {e}]", {"model": model}


# ==========================================================
# MASTER FUNCTION
# ==========================================================
def analyze_image(image_path: str,
                  do_ocr: bool = True,
                  do_detect: bool = False,
                  do_llava: bool = False,
                  llava_prompt: str = "Describe this image.") -> Dict[str, Any]:
    """Comprehensive vision + math analyzer"""
    result = {"source": Path(image_path).name}

    if do_ocr:
        text, meta = ocr_image(image_path)
        result["ocr_text"] = text
        result["ocr_meta"] = meta

        # Auto-solve math if detected
        if meta.get("math_detected"):
            result["math_solution"] = meta["math_solution"]

    if do_detect:
        dets, ann = detect_objects(image_path)
        result["detections"] = dets
        result["annotated_image"] = ann

    if do_llava:
        desc, meta = llava_describe(image_path, llava_prompt)
        result["llava_answer"] = desc
        result["llava_meta"] = meta

    return result

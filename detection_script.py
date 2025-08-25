#!/usr/bin/env python3
r"""
Gloved vs. Ungloved Hand Detection (YOLOv8)

- Reads all images in --input (jpg/jpeg/png)
- Saves annotated images to --output
- Writes per-image JSON logs to --logs in the required format

Usage (PowerShell from submission/Part_1_Glove_Detection):
  python detection_script.py `
    --input .\data\test\images `
    --output .\output `
    --logs .\logs `
    --model .\weights\best.pt `
    --confidence 0.5 `
    --imgsz 640 `
    --device cpu `
    --workers 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    print("ERROR: Ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
    raise e

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Map dataset names/aliases -> required labels for the assessment
CANONICAL_MAP = {
    # gloved
    "gloved": "gloved_hand",
    "glove": "gloved_hand",
    "gloved_hand": "gloved_hand",
    "with_glove": "gloved_hand",
    "withglove": "gloved_hand",

    # bare / ungloved
    "not-gloved": "bare_hand",
    "no-glove": "bare_hand",
    "bare_hand": "bare_hand",
    "barehand": "bare_hand",
    "without_glove": "bare_hand",
    "withoutglove": "bare_hand",
    "ungloved": "bare_hand",
    "no_glove": "bare_hand",
}


def list_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_xyxy(box) -> List[int]:
    """Ultralytics Boxes.xyxy array/tensor -> [x1, y1, x2, y2] ints"""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    return [x1, y1, x2, y2]


def canonical_label(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return CANONICAL_MAP.get(key, name)  # fallback to original if unknown


def draw_annotations(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label_txt = f'{det["label"]} {det["confidence"]:.2f}'

        # box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # label background
        (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 8, th + 4)
        cv2.rectangle(
            annotated,
            (x1, y_text - th - 6),
            (x1 + tw + 6, y_text + baseline - 6),
            (0, 255, 0),
            thickness=-1
        )
        # label text (black on green bg)
        cv2.putText(
            annotated,
            label_txt,
            (x1 + 3, y_text - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_inference(
    model_path: str,
    input_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    workers: int,
    iou: float,
    max_det: int,
) -> None:
    ensure_dir(output_dir)
    ensure_dir(logs_dir)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}. Supported: {', '.join(sorted(IMG_EXTS))}")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # print model class map once for sanity
    class_map = getattr(getattr(model, "model", None), "names", getattr(model, "names", {}))
    print("Model class map:", class_map)

    # Batched prediction (Ultralytics handles batching internally)
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        device=device,
        workers=workers,
        verbose=False,
        stream=False,
    )

    # get class name mapping from model
    names = None
    if hasattr(model, "model") and hasattr(model.model, "names"):
        names = model.model.names
    elif hasattr(model, "names"):
        names = model.names
    if names is None:
        names = {}

    for img_path, res in zip(images, results):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"WARNING: Could not read image {img_path}, skipping.")
            continue

        detections: List[Dict[str, Any]] = []
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            for (box, score, cls_id) in zip(xyxy, confs, clss):
                raw_label = names.get(int(cls_id), f"class_{int(cls_id)}")
                label = canonical_label(str(raw_label))
                detections.append(
                    {
                        "label": label,
                        "confidence": float(round(float(score), 6)),
                        "bbox": to_xyxy(box),
                    }
                )

        # save annotated image
        annotated = draw_annotations(img_bgr, detections)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        # save JSON log
        json_record = {
            "filename": img_path.name,
            "detections": detections,
        }
        json_path = logs_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_record, f, indent=2)

        print(f"[OK] {img_path.name}: {len(detections)} detections -> {out_path.name}, {json_path.name}")


def parse_args():
    p = argparse.ArgumentParser(description="Gloved vs. Ungloved Hand Detection (YOLOv8)")
    p.add_argument("--input", type=str, required=True, help="Folder with input images (.jpg/.jpeg/.png)")
    p.add_argument("--output", type=str, default="./output", help="Folder to save annotated images")
    p.add_argument("--logs", type=str, default="./logs", help="Folder to save per-image JSON logs")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="Path to weights (e.g., .\\weights\\best.pt or runs\\...\\best.pt)")
    p.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', '0', '0,1', ...")
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (lower = stricter)")
    p.add_argument("--max_det", type=int, default=100, help="Max detections per image")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    logs_dir = Path(args.logs)

    if not input_dir.exists():
        print(f"ERROR: --input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    run_inference(
        model_path=args.model,
        input_dir=input_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        conf=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        iou=args.iou,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()

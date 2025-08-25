# Gloved vs. Ungloved Hand Detection (YOLOv8)

Detects gloved vs ungloved (bare) hands in images using a trained YOLOv8 model, saves annotated images, and writes per‑image JSON logs for evaluation.

**Inference script:** `detection_script.py`  
**Verification script:** `verify_submission.py`  
**Model weights (you provide):** `weights/best.pt`

## 1) Project Structure

```
glove-detection/
├─ detection_script.py
├─ verify_submission.py
├─ weights/               # put best.pt here (not committed by default)
│  └─ best.pt
├─ data/
│  ├─ test/
│  │  └─ images/         # your test images (.jpg/.jpeg/.png)
│  └─ valid/
│     └─ images/         # optional validation images
├─ output/                # annotated outputs (created at runtime)
├─ logs/                  # JSON logs per image (created at runtime)
├─ output_test/           # if you run on test set
└─ logs_test/             # if you run on test set
```

## 2) Environment Setup

### Windows (PowerShell)

```powershell
# from the repo root
python -m venv .venv
.venv\Scripts\Activate.ps1

# install deps
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

If you prefer, add a `requirements.txt`:

```
ultralytics
opencv-python
numpy
```

then install via `pip install -r requirements.txt`.

## 3) Put the Model Weights in Place

Place your trained YOLOv8 weights at:

```
weights/best.pt
```

(If your weights live elsewhere, pass the path via `--model`.)

## 4) Run Inference

### PowerShell (Windows)

The backtick ` is the line‑continuation character:

```powershell
python detection_script.py `
  --input .\data\test\images `
  --output .\output_test `
  --logs .\logs_test `
  --model .\weights\best.pt `
  --confidence 0.30 `
  --imgsz 416 `
  --iou 0.50 `
  --max_det 20 `
  --device cpu `
  --workers 0
```

### Bash (macOS/Linux)

Use `\` for line continuation:

```bash
python detection_script.py \
  --input ./data/test/images \
  --output ./output_test \
  --logs ./logs_test \
  --model ./weights/best.pt \
  --confidence 0.30 \
  --imgsz 416 \
  --iou 0.50 \
  --max_det 20 \
  --device cpu \
  --workers 0
```

### Key flags

- `--input` : folder with .jpg/.jpeg/.png
- `--output` : annotated images folder (auto‑created)
- `--logs` : JSON logs folder (auto‑created)
- `--model` : YOLO weights (e.g., weights/best.pt)
- `--confidence` : score threshold (0–1)
- `--imgsz` : inference image size (e.g., 416 / 640)
- `--iou` : NMS IoU threshold
- `--max_det` : max detections per image
- `--device` : cpu, 0 (GPU 0), 0,1 (multi‑GPU)
- `--workers` : dataloader workers (set 0 on Windows/CPU‑only)

### Outputs produced per image

- Annotated image in `--output`
- JSON log in `--logs`, e.g.:

```json
{
  "filename": "image123.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

### Label Canonicalization

- Model classes like `gloved`, `glove`, `gloved_hand`, `with_glove` → `gloved_hand`
- `not-gloved`, `bare_hand`, `ungloved`, `no_glove`, etc. → `bare_hand`

## 5) Verify the Submission Logs

After running inference:

```bash
python .\verify_submission.py --images .\data\test\images --logs .\logs_test
```

Expected output example:

```
Images: 98 | JSON logs: 98
✅ All logs look good.
```


## 6) Tips & Troubleshooting

- **Performance on CPU:** prefer smaller `--imgsz` (e.g., 416), `--workers 0`, lower `--max_det`.
- **Freezing on Windows:** set `--workers 0`.
- **CUDA/GPU:** install compatible CUDA toolchain and set `--device 0`.
- **No images found:** check `--input` path and supported extensions.
- **OpenCV errors on write:** verify write permissions for `--output`.
- **Ultralytics missing:** `pip install ultralytics`.

## 7) License

Add your preferred license (e.g., MIT) here.

## 8) Acknowledgments

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
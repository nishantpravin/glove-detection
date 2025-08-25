import argparse, json
from pathlib import Path
import cv2

VALID = {"gloved_hand", "bare_hand"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--logs", required=True)
    args = ap.parse_args()

    img_dir = Path(args.images)
    log_dir = Path(args.logs)

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    logs = sorted([p for p in log_dir.iterdir() if p.suffix.lower()==".json"])

    print(f"Images: {len(imgs)} | JSON logs: {len(logs)}")

    img_set = {p.stem for p in imgs}
    log_set = {p.stem for p in logs}
    missing = img_set - log_set
    extra   = log_set - img_set
    if missing: print(f"Missing JSONs for {len(missing)} images (first 5): {sorted(list(missing))[:5]}")
    if extra:   print(f"Extra JSONs without images: {sorted(list(extra))[:5]}")

    bad = 0
    for jp in logs:
        try:
            js = json.loads(jp.read_text(encoding="utf-8"))
            fn = js.get("filename","")
            dets = js.get("detections",[])
            # Optional: image size check for bbox bounds
            imgp = img_dir / fn
            im = cv2.imread(str(imgp))
            h = im.shape[0] if im is not None else None
            w = im.shape[1] if im is not None else None

            for d in dets:
                if d.get("label") not in VALID:
                    raise ValueError(f"Bad label {d.get('label')}")
                c = d.get("confidence")
                if not (isinstance(c,float) or isinstance(c,int)) or not (0 <= float(c) <= 1):
                    raise ValueError(f"Bad confidence {c}")
                bbox = d.get("bbox")
                if not (isinstance(bbox, list) and len(bbox)==4 and all(isinstance(x,int) for x in bbox)):
                    raise ValueError(f"Bad bbox {bbox}")
                if w and h:
                    x1,y1,x2,y2 = bbox
                    if not (0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h):
                        raise ValueError(f"Out-of-bounds bbox {bbox} for {fn} ({w}x{h})")
        except Exception as e:
            bad += 1
            print(f"[BAD] {jp.name}: {e}")

    if bad==0 and not missing and not extra:
        print("✅ All logs look good.")
    else:
        print(f"⚠️ Issues found: {bad} bad files.")
if __name__ == "__main__":
    main()
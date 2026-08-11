#!/usr/bin/env python3
"""Decode BaDLAD paper-test with Ultralytics YOLOv8-seg → COCO preds JSON.

Hyps stay private under --out-dir. Resume-safe (skips done image_ids).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def instances_to_coco(result, image_id: int) -> list:
    import pycocotools.mask as mask_util

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    has_mask = result.masks is not None
    masks = result.masks.data.cpu().numpy() if has_mask else None

    out = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].tolist()
        item = {
            "image_id": int(image_id),
            "category_id": int(classes[i]),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(scores[i]),
        }
        if has_mask:
            # masks are at model imgsz; resize to original via result.orig_shape
            m = masks[i].astype(np.uint8)
            if m.shape != result.orig_shape:
                import cv2

                m = cv2.resize(
                    m, (result.orig_shape[1], result.orig_shape[0]), interpolation=cv2.INTER_NEAREST
                )
            rle = mask_util.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            item["segmentation"] = rle
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    from ultralytics import YOLO

    gold = json.loads(args.gold.read_text())
    name_to_id = {im["file_name"]: int(im["id"]) for im in gold["images"]}
    images = sorted(args.images.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]
    missing = [p.name for p in images if p.name not in name_to_id]
    if missing:
        raise SystemExit(f"{len(missing)} images not in gold, e.g. {missing[:3]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = args.out_dir / "preds_coco.json"
    done_path = args.out_dir / "done_image_ids.json"

    preds: list = []
    done: set[int] = set()
    if preds_path.exists():
        preds = json.loads(preds_path.read_text())
    if done_path.exists():
        done = set(json.loads(done_path.read_text()))

    print(f"device={args.device} already_done={len(done)} total={len(images)}", flush=True)
    model = YOLO(str(args.weights))

    pending = [p for p in images if name_to_id[p.name] not in done]
    for n, path in enumerate(tqdm(pending, desc="decode-yolo"), 1):
        image_id = name_to_id[path.name]
        results = model.predict(
            source=str(path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            retina_masks=True,
        )
        preds.extend(instances_to_coco(results[0], image_id))
        done.add(image_id)
        if n % args.save_every == 0 or n == len(pending):
            preds_path.write_text(json.dumps(preds))
            done_path.write_text(json.dumps(sorted(done)))

    preds_path.write_text(json.dumps(preds))
    done_path.write_text(json.dumps(sorted(done)))
    meta = {
        "weights": str(args.weights),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "n_images": len(done),
        "n_instances": len(preds),
        "preds": str(preds_path),
    }
    (args.out_dir / "decode_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decode BaDLAD paper-test with Detectron2 Mask R-CNN → COCO preds JSON.

Hyps stay private under --out-dir. Resume-safe (skips done image_ids).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def build_predictor(
    weights: Path,
    score_thresh: float,
    device: str,
    arch: str = "mask",
    out_dir: Path | None = None,
):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg_name = {
        "mask": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "faster": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    }[arch]

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = device

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frozen_cfg.yaml").write_text(cfg.dump())

    return DefaultPredictor(cfg)


def instances_to_coco(instances, image_id: int) -> list:
    from detectron2.structures import Boxes, BoxMode
    from detectron2.utils.visualizer import GenericMask
    import pycocotools.mask as mask_util

    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []
    has_mask = instances.has("pred_masks")
    results = []
    for i in range(len(scores)):
        bbox_xyxy = boxes[i].tolist()
        bbox_xywh = BoxMode.convert(bbox_xyxy, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        item = {
            "image_id": int(image_id),
            "category_id": int(classes[i]),
            "bbox": [float(x) for x in bbox_xywh],
            "score": float(scores[i]),
        }
        if has_mask:
            mask = instances.pred_masks[i].numpy().astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("ascii")
            item["segmentation"] = rle
        results.append(item)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True, help="Dir of paper_test PNGs")
    p.add_argument("--gold", type=Path, required=True, help="COCO gold (for image_id map)")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument(
        "--arch",
        choices=("mask", "faster"),
        default="mask",
        help="mask = Mask R-CNN; faster = Faster R-CNN (bbox only)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N images (smoke)")
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    gold = json.loads(args.gold.read_text())
    # map file_name -> image_id
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
    predictor = build_predictor(
        args.weights, args.score_thresh, args.device, args.arch, args.out_dir
    )

    pending = [p for p in images if name_to_id[p.name] not in done]
    for n, path in enumerate(tqdm(pending, desc="decode"), 1):
        image_id = name_to_id[path.name]
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"failed to read {path}")
        with torch.no_grad():
            out = predictor(img)
        preds.extend(instances_to_coco(out["instances"], image_id))
        done.add(image_id)
        if n % args.save_every == 0 or n == len(pending):
            preds_path.write_text(json.dumps(preds))
            done_path.write_text(json.dumps(sorted(done)))

    preds_path.write_text(json.dumps(preds))
    done_path.write_text(json.dumps(sorted(done)))
    meta = {
        "weights": str(args.weights),
        "arch": args.arch,
        "score_thresh": args.score_thresh,
        "device": args.device,
        "n_images": len(done),
        "n_instances": len(preds),
        "preds": str(preds_path),
    }
    (args.out_dir / "decode_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

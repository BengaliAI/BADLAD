#!/usr/bin/env python3
"""Score BaDLAD paper-test preds (COCO instances) vs private gold.

Publishes metrics only — never upload hyps/gold.
Primary metric: COCO mask AP@[.5:.95] (mask_map).
Optional --domains CSV (file_name,domain) → group_by domain (paper Table 3 shape).
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from copy import deepcopy
from datetime import date
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _shift_cat_ids(coco_dict: dict, delta: int = 1) -> dict:
    """BaDLAD cats are 0-based; pycocotools expects positive ids."""
    out = deepcopy(coco_dict)
    for c in out.get("categories", []):
        c["id"] = int(c["id"]) + delta
    for a in out.get("annotations", []):
        a["category_id"] = int(a["category_id"]) + delta
    return out


def _shift_pred_cats(preds: list, delta: int = 1) -> list:
    out = []
    for p in preds:
        q = dict(p)
        q["category_id"] = int(q["category_id"]) + delta
        out.append(q)
    return out


def _run_eval(coco_gt: COCO, preds: list, iou_type: str) -> dict:
    if not preds:
        return {
            "overall": {k: float("nan") for k in ["AP", "AP50", "AP75", "APs", "APm", "APl"]},
            "by_class": {},
        }
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    overall = {n: float(ev.stats[i]) for i, n in enumerate(names)}
    per_cat = {}
    cat_ids = coco_gt.getCatIds()
    precision = ev.eval.get("precision")
    if precision is not None:
        for i, cid in enumerate(cat_ids):
            p = precision[:, :, i, 0, -1]
            p = p[p > -1]
            ap = float(p.mean()) if p.size else float("nan")
            name = coco_gt.loadCats([cid])[0]["name"]
            per_cat[name] = ap
    return {"overall": overall, "by_class": per_cat}


def _subset_gold(gold: dict, keep_ids: set[int]) -> dict:
    imgs = [im for im in gold["images"] if int(im["id"]) in keep_ids]
    anns = [a for a in gold["annotations"] if int(a["image_id"]) in keep_ids]
    return {
        "info": gold.get("info", {}),
        "licenses": gold.get("licenses", []),
        "categories": gold["categories"],
        "images": imgs,
        "annotations": anns,
    }


def _load_domains(path: Path) -> dict[str, str]:
    """file_name -> domain."""
    out: dict[str, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["file_name"]] = row["domain"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", required=True, type=Path, help="COCO gold JSON")
    p.add_argument("--preds", required=True, type=Path, help="COCO results JSON list")
    p.add_argument(
        "--domains",
        type=Path,
        default=None,
        help="CSV with columns file_name,domain (paper Table 3 grouping)",
    )
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-url", default="")
    p.add_argument("--backend", default="detectron2_mask_rcnn")
    p.add_argument("--scorer-commit", default="")
    p.add_argument("--decode-commit", default="")
    p.add_argument("--requested-by", default="maintainer")
    p.add_argument("--notes", default="")
    p.add_argument("--out", type=Path, default=Path("metrics.json"))
    p.add_argument("--quiet", action="store_true", help="suppress COCOeval summarize spam")
    args = p.parse_args()

    gold_raw = json.loads(args.gold.read_text())
    preds_raw = json.loads(args.preds.read_text())
    if not isinstance(preds_raw, list):
        raise SystemExit("preds must be a JSON list of COCO result dicts")

    gold_shift = _shift_cat_ids(gold_raw, 1)
    preds_shift = _shift_pred_cats(preds_raw, 1)

    name_to_id = {im["file_name"]: int(im["id"]) for im in gold_shift["images"]}
    id_to_name = {v: k for k, v in name_to_id.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def eval_slice(gold_sub: dict, preds_sub: list) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(gold_sub, tf)
            tmp = Path(tf.name)
        try:
            coco_gt = COCO(str(tmp))
            img_ids = set(coco_gt.getImgIds())
            preds_f = [x for x in preds_sub if int(x["image_id"]) in img_ids]
            # silence summarize if quiet: redirect via monkeypatch of print? keep simple
            segm = _run_eval(coco_gt, preds_f, "segm")
            bbox = _run_eval(coco_gt, preds_f, "bbox")
        finally:
            tmp.unlink(missing_ok=True)
        return {
            "mask_map": segm["overall"]["AP"],
            "mask_map50": segm["overall"]["AP50"],
            "mask_map_by_class": segm["by_class"],
            "bbox_map": bbox["overall"]["AP"],
            "bbox_map50": bbox["overall"]["AP50"],
            "bbox_map_by_class": bbox["by_class"],
            "n_images": len(img_ids),
            "n_pred_instances": len(preds_f),
        }

    overall = eval_slice(gold_shift, preds_shift)

    by_domain = {}
    if args.domains:
        dom_map = _load_domains(args.domains)
        # domain -> set of image_ids
        domain_ids: dict[str, set[int]] = {}
        missing = 0
        for fname, iid in name_to_id.items():
            d = dom_map.get(fname)
            if d is None:
                missing += 1
                continue
            domain_ids.setdefault(d, set()).add(iid)
        if missing:
            raise SystemExit(f"domains CSV missing {missing} gold file_names")
        for domain, ids in sorted(domain_ids.items()):
            gold_sub = _subset_gold(gold_shift, ids)
            preds_sub = [x for x in preds_shift if int(x["image_id"]) in ids]
            by_domain[domain] = eval_slice(gold_sub, preds_sub)

    row = {
        "model_id": args.model_id,
        "model_url": args.model_url,
        "mask_map": overall["mask_map"],
        "mask_map50": overall["mask_map50"],
        "mask_map_by_class": overall["mask_map_by_class"],
        "bbox_map": overall["bbox_map"],
        "bbox_map50": overall["bbox_map50"],
        "bbox_map_by_class": overall["bbox_map_by_class"],
        "mask_map_by_domain": {
            d: {
                "mask_map": v["mask_map"],
                "mask_map_by_class": v["mask_map_by_class"],
                "n_images": v["n_images"],
            }
            for d, v in by_domain.items()
        },
        "backend": args.backend,
        "scorer_commit": args.scorer_commit,
        "decode_commit": args.decode_commit,
        "evaluated_at": date.today().isoformat(),
        "requested_by": args.requested_by,
        "notes": args.notes,
        "n_images_gold": overall["n_images"],
        "n_pred_instances": overall["n_pred_instances"],
    }
    args.out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

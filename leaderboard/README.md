# BaDLAD maintainer-run leaderboard tooling

Private eval against `badlad-test` (paper COCO hidden test, local/Drive only).

## Environment (uv)

Pinned to recreate the 2026-07-25 Mask R-CNN baseline:

```bash
cd leaderboard
uv sync          # creates .venv from uv.lock (Python 3.12, torch cu124)
```

| Pin | Value |
| --- | --- |
| Python | 3.12 (see `.python-version`) |
| Lock | `uv.lock` |
| Torch | 2.5.1+cu124 |
| Detectron2 | 0.6 (from source) |
| pycocotools | 2.0.11 |
| opencv | 4.10.0 |

Baseline used conda `pytorch_env` before this lock existed; versions above match that run.
New evals should use `uv sync` + `uv run`.

## Paths

| Role | Path |
| --- | --- |
| Images | `/workspace/Datasets/badlad-test-package/images/paper_test/` |
| Gold | `/workspace/Datasets/badlad-test-package/labels/coco/BDLAD_test_coco.json` |
| Weights (v1) | `/workspace/Datasets/badlad-paper-models/maskrcnn_scratch_10k_iter/model_final.pth` |
| Domain map | `/workspace/Datasets/badlad-test-package/meta/paper_test_domains.csv` |
| Run dir | `/workspace/Datasets/badlad-bench-runs/<date>_<slug>/` |

## Decode (private hyps)

```bash
uv run python decode_detectron_maskrcnn.py \
  --images /workspace/Datasets/badlad-test-package/images/paper_test \
  --gold /workspace/Datasets/badlad-test-package/labels/coco/BDLAD_test_coco.json \
  --weights /workspace/Datasets/badlad-paper-models/maskrcnn_scratch_10k_iter/model_final.pth \
  --out-dir /workspace/Datasets/badlad-bench-runs/YYYYMMDD_maskrcnn_scratch \
  --score-thresh 0.05 --device cuda
```

Writes private `preds_coco.json` + `frozen_cfg.yaml` (never push to Hub/git).

## Score (never publish hyps)

```bash
uv run python score.py \
  --gold /workspace/Datasets/badlad-test-package/labels/coco/BDLAD_test_coco.json \
  --preds /workspace/Datasets/badlad-bench-runs/.../preds_coco.json \
  --domains /workspace/Datasets/badlad-test-package/meta/paper_test_domains.csv \
  --model-id badlad-mrcnn-paper \
  --backend detectron2_mask_rcnn_R50_FPN \
  --out /workspace/Datasets/badlad-bench-runs/.../metrics.json
```

Primary: `mask_map` (COCO mask AP@[.5:.95]). Also reports `bbox_map`, per-class, and
`mask_map_by_domain` (paper Table 3 shape) when `--domains` given.

Domain map built from tracker `original_name` prefixes (same rules as
`BadLad_domainwise_test_set_metric_calculator.ipynb`):
govtdoc / property_dalil / Purbasha / ijhj|image000|kjgj|njhjghb /
বাংলাদেশের_স্বাধীনতা_যুদ্ধ… / else → Magazine and Books.

## Baseline vs paper (2026-07-25)

Local re-evaluation with `score_thresh=0.05` on the full 13 328-image paper hidden
test. Performance is slightly higher than paper Table 3 on two domains; the remaining
four match to < 0.1 AP.

**Mask AP ×100 — local @0.05 / paper (M-RCNN | ImgNet | Mask)**

| Domain | n | P | Tx | I | Tb |
| --- | ---: | --- | --- | --- | --- |
| Historical Newspapers | 345 | 60.3 / 60.3 | 18.3 / 18.3 | 57.3 / 57.3 | 0.0 / 0.0 |
| New Newspapers | 65 | 41.4 / 41.4 | 13.1 / 13.2 | 45.2 / 45.2 | 1.9 / 1.9 |
| Magazine and Books | 11674 | 61.8 / 61.8 | 25.3 / 25.3 | 44.9 / 44.9 | 2.3 / 2.3 |
| Liberation War Documents | 402 | 71.1 / 71.2 | 26.8 / 26.8 | 1.1 / 1.0 | 40.1 / 40.1 |
| Government Documents | 514 | **49.4** / 39.1 | **23.7** / 18.7 | **26.1** / 19.4 | **5.1** / 3.7 |
| Property Deeds | 328 | **38.0** / 0.6 | **14.2** / 0.7 | **13.3** / 2.1 | **3.2** / 0.6 |

Government Documents and Property Deeds are consistently higher locally. Possible
causes: different image subset, annotation version drift, or notebook data-loading
bug in the original paper eval. Performance has seemingly improved on these domains
in the local run — needs investigation.

## Notes

- `train_finetuned_10k_iter` is Faster R-CNN (no masks) — not for mask_map.
- Pred format: COCO instances JSON (bbox + RLE segmentation).
- `frozen_cfg.yaml` saved per run captures full Detectron config (input size, NMS, anchors, etc.).

# BaDLAD leaderboard (maintainer-run)

Eval mode: `maintainer_eval` on private dataset `badlad-test` (paper COCO).

## Paths

| Role | Path |
| --- | --- |
| Images | `/workspace/Datasets/badlad-test-package/images/paper_test/` |
| Gold | `/workspace/Datasets/badlad-test-package/labels/coco/BDLAD_test_coco.json` |
| Weights (v1) | `/workspace/Datasets/badlad-paper-models/maskrcnn_scratch_10k_iter/model_final.pth` |
| Run dir | `/workspace/Datasets/badlad-bench-runs/<date>_<slug>/` |

## Decode

```bash
conda activate pytorch_env
python leaderboard/decode_detectron_maskrcnn.py \
  --images /workspace/Datasets/badlad-test-package/images/paper_test \
  --gold /workspace/Datasets/badlad-test-package/labels/coco/BDLAD_test_coco.json \
  --weights /workspace/Datasets/badlad-paper-models/maskrcnn_scratch_10k_iter/model_final.pth \
  --out-dir /workspace/Datasets/badlad-bench-runs/YYYYMMDD_maskrcnn_scratch \
  --score-thresh 0.5 --device cuda
```

Writes private `preds_coco.json` (never push to Hub/git).

## Score

```bash
python leaderboard/score.py \
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

## Notes

- `train_finetuned_10k_iter` is Faster R-CNN (no masks) — not for mask_map.
- Pred format: COCO instances JSON (bbox + RLE segmentation).

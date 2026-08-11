---
name: BaDLAD evaluation request
about: Ask maintainers to score a model on the private BaDLAD paper hidden test
title: "[eval] "
labels: ["evaluation"]
---

## Model

- **model_id** (Hub id or slug):
- **model_url** (downloadable weights):
- **Framework / backend** (Detectron2 Mask R-CNN, YOLOv8-seg, other — describe):
- **License**:

## Notes for maintainers

- Expected pred format: COCO instances JSON (bbox + RLE masks when applicable).
- Primary metric: COCO mask AP@[.5:.95] (`mask_map`) on private `badlad-test`.
- Do **not** attach predictions or ask for the test set.

## Checklist

- [ ] Weights are publicly downloadable
- [ ] I understand hypotheses will not be returned
- [ ] Contact (GitHub / email):

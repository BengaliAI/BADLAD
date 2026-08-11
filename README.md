**BaDLAD: A Large Multi-Domain Bengali Document Layout Analysis Dataset**
https://arxiv.org/abs/2303.05325

While strides have been made in deep learning based Bengali Optical Character Recognition (OCR) in the past decade, the absence of large Document Layout Analysis (DLA) datasets has hindered the application of OCR in document transcription, e.g., transcribing historical documents and newspapers. Moreover, rule-based DLA systems that are currently being employed in practice are not robust to domain variations and out-of-distribution layouts. To this end, we present the first multidomain large Bengali Document Layout Analysis Dataset: BaDLAD. This dataset contains 33,695 human annotated document samples from six domains - i) books and magazines, ii) public domain govt. documents, iii) liberation war documents, iv) newspapers, v) historical newspapers, and vi) property deeds, with 710K polygon annotations for four unit types: text-box, paragraph, image, and table. Through preliminary experiments benchmarking the performance of existing state-of-the-art deep learning architectures for English DLA, we demonstrate the efficacy of our dataset in training deep learning based Bengali document digitization models.

Dataset: https://www.kaggle.com/datasets/reasat/badlad-train

Kaggle Competition: https://www.kaggle.com/competitions/dlsprint2/data

## Models (inventory — verified facts)

Paper Table 3 has 7 configs. The files below are what we currently have on disk /
public hosts. Table 3 membership is labeled cautiously.

### BaDLAD-trained (or production) weights

| ID / path | What | sha256 (prefix) | Table 3 |
| --- | --- | --- | --- |
| `maskrcnn_scratch_10k_iter/model_final.pth` ([HF](https://huggingface.co/bengaliAI/badlad-mrcnn-paper)) | Detectron Mask R-CNN R50 (mask head) | `d3b663…` | **likely** M-RCNN\|ImgNet\|Mask (domain AP match; not proven identical to original run) |
| `train_finetuned_10k_iter/model_final.pth` | Detectron Faster R-CNN (bbox only) | `bd507a…` | **unverified** (some F-RCNN\|BBox row) |
| `yolo-best` / `badlad-yolov8-best` | YOLOv8m-seg (Ultralytics; args in ckpt) | `890e666…` | **unverified** (paper says 100 ep / bs8; this ckpt is 20 ep / bs32) |

Detectron finetunes live under Drive `DLA_Project/badlad_trained/`. YOLO also
ships as `layoutparserweights/yolo-best.pt.zip` and is the bbocr layout stage.
The same Drive folder’s `PubLayNet_*` files are official LayoutParser PubLayNet
**base** init weights (sha256-matched to HF `layoutparser/detectron2`), not
BaDLAD finetunes / Table 3 scored Detectron checkpoints.

Training docs we have: paper §experiments (high level); Detectron notebooks in
this repo (recipes, no frozen `config.yaml` beside weights); YOLO
`ckpt['train_args']` (authoritative for that file).

## Leaderboard

Maintainer-run evaluation on the private paper hidden test (13 328 images, COCO mask AP).
Decode + score scripts live in [`leaderboard/`](leaderboard/).

```bash
uv sync                          # installs everything except detectron2
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'  # needs CUDA toolkit
```

Full pinned versions in `requirements.txt`; `uv.lock` covers the uv-resolvable subset.

### Local replay vs paper (2026-07-25) — likely M-RCNN\|ImgNet\|Mask

Checkpoint: `maskrcnn_scratch_10k_iter`, `score_thresh=0.05`. Four domains match
paper to < 0.1 AP; Government Documents and Property Deeds are higher locally —
needs investigation. This is a **likely** Table 3 match, not a claim of bit-identical
reproduction of the original paper run.

**Mask AP ×100 — local / paper (M-RCNN | ImgNet | Mask)**

| Domain | n | P | Tx | I | Tb |
| --- | ---: | --- | --- | --- | --- |
| Historical Newspapers | 345 | 60.3 / 60.3 | 18.3 / 18.3 | 57.3 / 57.3 | 0.0 / 0.0 |
| New Newspapers | 65 | 41.4 / 41.4 | 13.1 / 13.2 | 45.2 / 45.2 | 1.9 / 1.9 |
| Magazine and Books | 11674 | 61.8 / 61.8 | 25.3 / 25.3 | 44.9 / 44.9 | 2.3 / 2.3 |
| Liberation War Documents | 402 | 71.1 / 71.2 | 26.8 / 26.8 | 1.1 / 1.0 | 40.1 / 40.1 |
| Government Documents | 514 | **49.4** / 39.1 | **23.7** / 18.7 | **26.1** / 19.4 | **5.1** / 3.7 |
| Property Deeds | 328 | **38.0** / 0.6 | **14.2** / 0.7 | **13.3** / 2.1 | **3.2** / 0.6 |

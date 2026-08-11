**BaDLAD: A Large Multi-Domain Bengali Document Layout Analysis Dataset**
https://arxiv.org/abs/2303.05325

While strides have been made in deep learning based Bengali Optical Character Recognition (OCR) in the past decade, the absence of large Document Layout Analysis (DLA) datasets has hindered the application of OCR in document transcription, e.g., transcribing historical documents and newspapers. Moreover, rule-based DLA systems that are currently being employed in practice are not robust to domain variations and out-of-distribution layouts. To this end, we present the first multidomain large Bengali Document Layout Analysis Dataset: BaDLAD. This dataset contains 33,695 human annotated document samples from six domains - i) books and magazines, ii) public domain govt. documents, iii) liberation war documents, iv) newspapers, v) historical newspapers, and vi) property deeds, with 710K polygon annotations for four unit types: text-box, paragraph, image, and table. Through preliminary experiments benchmarking the performance of existing state-of-the-art deep learning architectures for English DLA, we demonstrate the efficacy of our dataset in training deep learning based Bengali document digitization models.

Dataset: https://www.kaggle.com/datasets/reasat/badlad-train

Kaggle Competition: https://www.kaggle.com/competitions/dlsprint2/data

## Models

BaDLAD-trained checkpoints (Hugging Face):

| Model | Hub | Notes |
| --- | --- | --- |
| Mask R-CNN R50 (paper mask baseline) | [`bengaliAI/badlad-mrcnn-paper`](https://huggingface.co/bengaliAI/badlad-mrcnn-paper) | Instance segmentation; leaderboard `mask_map` |
| Faster R-CNN R50 (paper bbox baseline) | [`bengaliAI/badlad-frcnn-paper`](https://huggingface.co/bengaliAI/badlad-frcnn-paper) | Bounding boxes only |
| YOLOv8m-seg | [`bengaliAI/badlad-yolov8m-seg`](https://huggingface.co/bengaliAI/badlad-yolov8m-seg) | Layout seg; also used in bbocr |

PubLayNet Detectron weights that appear beside older Drive dumps are LayoutParser
**base** init checkpoints (not BaDLAD finetunes).

## Leaderboard

Maintainer-run evaluation on the private paper hidden test (13 328 images, COCO mask AP).
Decode + score scripts live in [`leaderboard/`](leaderboard/).

```bash
uv sync                          # installs everything except detectron2
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'  # needs CUDA toolkit
```

Full pinned versions in `requirements.txt`; `uv.lock` covers the uv-resolvable subset.

### Mask R-CNN vs paper Table 3 (2026-07-25)

Checkpoint: `bengaliAI/badlad-mrcnn-paper`, `score_thresh=0.05`. Domain-wise mask AP
(×100) compared to paper row M-RCNN | ImgNet | Mask:

| Domain | n | P | Tx | I | Tb |
| --- | ---: | --- | --- | --- | --- |
| Historical Newspapers | 345 | 60.3 / 60.3 | 18.3 / 18.3 | 57.3 / 57.3 | 0.0 / 0.0 |
| New Newspapers | 65 | 41.4 / 41.4 | 13.1 / 13.2 | 45.2 / 45.2 | 1.9 / 1.9 |
| Magazine and Books | 11674 | 61.8 / 61.8 | 25.3 / 25.3 | 44.9 / 44.9 | 2.3 / 2.3 |
| Liberation War Documents | 402 | 71.1 / 71.2 | 26.8 / 26.8 | 1.1 / 1.0 | 40.1 / 40.1 |
| Government Documents | 514 | 49.4 / 39.1 | 23.7 / 18.7 | 26.1 / 19.4 | 5.1 / 3.7 |
| Property Deeds | 328 | 38.0 / 0.6 | 14.2 / 0.7 | 13.3 / 2.1 | 3.2 / 0.6 |

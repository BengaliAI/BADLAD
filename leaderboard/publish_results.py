#!/usr/bin/env python3
"""Publish a scored run to the public BaDLAD results dataset.

Reads private metrics.json (from score.py) and upserts one row into
bengaliAI/badlad-results results.csv. Only score fields leave the
machine — never preds, gold, or images.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RESULTS_REPO = "bengaliAI/badlad-results"
COLUMNS = [
    "rank",
    "model_id",
    "model_url",
    "mask_map",
    "mask_map_by_domain",
    "bbox_map",
    "backend",
    "scorer_commit",
    "decode_commit",
    "evaluated_at",
    "requested_by",
    "notes",
]
REQUIRED = ["model_id", "mask_map", "evaluated_at"]


def row_from_metrics(m: dict) -> dict:
    missing = [k for k in REQUIRED if k not in m]
    if missing:
        raise SystemExit(f"metrics.json missing fields: {missing}")
    by_dom = m.get("mask_map_by_domain") or {}
    # Compact: domain -> mask_map only for the public CSV
    if isinstance(by_dom, dict):
        compact = {
            d: (v.get("mask_map") if isinstance(v, dict) else v) for d, v in by_dom.items()
        }
    else:
        compact = {}
    return {
        "rank": "",
        "model_id": m["model_id"],
        "model_url": m.get("model_url", ""),
        "mask_map": f"{float(m['mask_map']):.6f}",
        "mask_map_by_domain": json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        "bbox_map": f"{float(m['bbox_map']):.6f}" if m.get("bbox_map") is not None else "",
        "backend": m.get("backend", ""),
        "scorer_commit": m.get("scorer_commit", ""),
        "decode_commit": m.get("decode_commit", ""),
        "evaluated_at": m["evaluated_at"],
        "requested_by": m.get("requested_by", "maintainer"),
        "notes": m.get("notes", ""),
    }


def load_existing(repo: str) -> list[dict]:
    try:
        path = hf_hub_download(repo, "results.csv", repo_type="dataset")
    except Exception as e:
        print(f"no existing results.csv ({e}); starting empty", file=sys.stderr)
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def render(rows: list[dict]) -> str:
    # Higher mask_map is better
    rows = sorted(rows, key=lambda r: float(r["mask_map"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = str(i)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in COLUMNS})
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--repo", default=RESULTS_REPO)
    p.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite an existing row with the same model_id",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Print CSV, upload nothing")
    g.add_argument("--publish", action="store_true", help="Upload to the Hub")
    args = p.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    new = row_from_metrics(metrics)

    rows = load_existing(args.repo)
    dupes = [r for r in rows if r["model_id"] == new["model_id"]]
    if dupes and not args.replace:
        raise SystemExit(
            f"{new['model_id']} already on the board (mask_map={dupes[0]['mask_map']}); "
            "use --replace to overwrite"
        )
    rows = [r for r in rows if r["model_id"] != new["model_id"]]
    rows.append(new)

    text = render(rows)
    print(text)

    if args.dry_run:
        print("dry run — nothing uploaded", file=sys.stderr)
        return

    HfApi().upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="results.csv",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"Add score for {new['model_id']} (mask_map={new['mask_map']})",
    )
    print(f"published {new['model_id']} to {args.repo}", file=sys.stderr)


if __name__ == "__main__":
    main()

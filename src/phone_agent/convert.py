"""Turn AndroidControl into training and test data.

Two stages, so the slow part only runs once:

1. `steps`  (slow, ~15 min): read the 50 GB of demos once. For every screen, save a
   shrunk JPEG and one "step record": goal, earlier actions, the human's action
   (the label), and the element boxes the scorer needs. Output:
       data/processed/android_control/images/<episode>/<step>.jpg
       data/processed/android_control/steps/{train,validation,test}.jsonl
       data/processed/android_control/summary.json

2. `sft`  (fast): build chat rows for the training script from the step records,
   with each image embedded as text (base64). Output:
       data/processed/android_control/sft/<split>.jsonl

Usage:
    python -m phone_agent.convert steps [--shards 0 1] [--max-image-tokens 1000] [--workers 6]
    python -m phone_agent.convert sft --split train [--no-done] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from phone_agent import a11y
from phone_agent.actions import Action, to_norm
from phone_agent.android_control import SHARD_GLOB, convert_action, is_point, iter_episodes, load_splits
from phone_agent.prompt import DEFAULT_MAX_HISTORY, image_data_url, training_row
from phone_agent.screens import DEFAULT_MAX_IMAGE_TOKENS, resize_for_model, to_jpeg

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "android_control"
OUT_DIR = ROOT / "data" / "processed" / "android_control"
SPLITS = ("train", "validation", "test")
DONE = Action("done", status="complete")


def _norm_box(e: a11y.Element, width: int, height: int) -> list[int]:
    left, top = to_norm(e.left, e.top, width, height)
    right, bottom = to_norm(e.right, e.bottom, width, height)
    return [left, top, right, bottom]


def _target_boxes(elements: list[a11y.Element], x_px: float, y_px: float, w: int, h: int) -> list[dict]:
    """Boxes under the human's finger, smallest first, up to the first clickable one.

    The scorer will usually take the smallest clickable box (e.g. the whole list row,
    not just the text label inside it). Without any clickable box, keep the 5 smallest.
    """
    hits = a11y.boxes_at(elements, x_px, y_px)
    chain = []
    for e in hits:
        chain.append(e)
        if e.interactive:
            break
    else:
        chain = hits[:5]
    return [{"box": _norm_box(e, w, h), "interactive": e.interactive} for e in chain]


def _interactive_boxes(elements: list[a11y.Element], w: int, h: int) -> list[list[int]]:
    """Every visible clickable/typeable element on screen (for the optional snap add-on)."""
    seen, boxes = set(), []
    for e in elements:
        if not (e.visible and e.interactive and e.area > 0):
            continue
        box = _norm_box(e, w, h)
        if box[2] > box[0] and box[3] > box[1] and tuple(box) not in seen:
            seen.add(tuple(box))
            boxes.append(box)
    return boxes


def episode_steps(
    ep, split: str, groups: list[str], out_dir: Path, max_image_tokens: int
) -> tuple[list[dict], Counter]:
    """All step records for one demo, plus counts of anything skipped. Saves the images."""
    counts: Counter = Counter()
    n = len(ep.actions)
    if n == 0:
        counts["skipped_episode_no_actions"] += 1
        return [], counts
    if not (len(ep.screenshots) == len(ep.widths) == len(ep.heights) == len(ep.trees) == n + 1):
        counts["skipped_episode_bad_lengths"] += 1
        return [], counts
    try:
        labels = [convert_action(a, ep.widths[i], ep.heights[i]) for i, a in enumerate(ep.actions)]
    except ValueError:
        counts["skipped_episode_bad_action"] += 1
        return [], counts

    image_dir = out_dir / "images" / str(ep.episode_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    history: list[str] = []
    for i in range(n + 1):
        w, h = ep.widths[i], ep.heights[i]
        image = resize_for_model(ep.screenshots[i], max_image_tokens)
        (image_dir / f"{i}.jpg").write_bytes(to_jpeg(image))
        elements = a11y.parse_forest(ep.trees[i])
        final = i == n
        label = DONE if final else labels[i]
        record = {
            "id": f"e{ep.episode_id}-s{i}",
            "episode_id": ep.episode_id,
            "step": i,
            "num_actions": n,
            "split": split,
            "groups": groups,
            "goal": ep.goal,
            "step_instruction": None if final else ep.step_instructions[i],
            "history": list(history),
            "label": label.to_json(),
            # "human" = the human's recorded action. "end_of_demo" = we added "done"
            # because the human stopped here; report it separately from human steps.
            "label_source": "end_of_demo" if final else "human",
            "raw_action": None if final else ep.actions[i],
            "image": f"images/{ep.episode_id}/{i}.jpg",
            "screen": [w, h],
            "image_size": [image.width, image.height],
            "target_boxes": [],
            "interactive_boxes": _interactive_boxes(elements, w, h),
        }
        if not final and is_point(label):
            raw = ep.actions[i]
            record["target_boxes"] = _target_boxes(elements, raw["x"], raw["y"], w, h)
            counts["point_steps"] += 1
            counts["point_steps_without_box"] += not record["target_boxes"]
        records.append(record)
        counts[f"label_{label.action}"] += 1
        history.append(label.to_json())
    counts["episodes"] += 1
    return records, counts


def convert_shard(task: tuple[str, str, str, int]) -> tuple[list[dict], Counter]:
    """Worker: convert one shard file. Runs in its own process."""
    shard, raw_dir, out_dir, max_image_tokens = task
    split_of, groups_of = load_splits(raw_dir)
    records, counts = [], Counter()
    for ep in iter_episodes(shard):
        split = split_of.get(ep.episode_id)
        if split is None:
            counts["skipped_episode_no_split"] += 1
            continue
        recs, c = episode_steps(ep, split, groups_of.get(ep.episode_id, []), Path(out_dir), max_image_tokens)
        records.extend(recs)
        counts.update(c)
    return records, counts


def build_steps(raw_dir: Path, out_dir: Path, shards: list[int] | None, max_image_tokens: int, workers: int) -> int:
    paths = sorted(raw_dir.glob(SHARD_GLOB))
    if shards is not None:
        paths = [p for p in paths if int(p.name.split("-")[1]) in shards]
    if not paths:
        print(f"No shard files found in {raw_dir}. Run scripts/download_android_control.sh first.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(raw_dir), str(out_dir), max_image_tokens) for p in paths]
    records: list[dict] = []
    counts: Counter = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, (recs, c) in zip(paths, pool.map(convert_shard, tasks)):
            records.extend(recs)
            counts.update(c)
            print(f"{path.name}: {c['episodes']} demos, {len(recs)} steps", flush=True)

    records.sort(key=lambda r: (r["episode_id"], r["step"]))
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    per_split = Counter()
    for split in SPLITS:
        with open(steps_dir / f"{split}.jsonl", "w") as f:
            for r in records:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    per_split[split] += 1
    summary = {
        "shards": [p.name for p in paths],
        "max_image_tokens": max_image_tokens,
        "steps_per_split": dict(per_split),
        "demos_per_split": {
            s: len({r["episode_id"] for r in records if r["split"] == s}) for s in SPLITS
        },
        "counts": dict(sorted(counts.items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def build_sft(
    out_dir: Path, split: str, include_done: bool, limit: int | None, max_history: int, use_step_instruction: bool
) -> int:
    src = out_dir / "steps" / f"{split}.jsonl"
    if not src.exists():
        print(f"{src} not found. Run the 'steps' stage first.")
        return 1
    dst_dir = out_dir / "sft"
    dst_dir.mkdir(exist_ok=True)
    suffix = "" if include_done else ".no_done"
    dst = dst_dir / f"{split}{suffix}.jsonl"
    written = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            step = json.loads(line)
            if not include_done and step["label_source"] != "human":
                continue
            row = training_row(
                step,
                image_data_url(out_dir / step["image"]),
                max_history=max_history,
                use_step_instruction=use_step_instruction,
            )
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    print(f"wrote {written} rows to {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phone_agent.convert", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    p_steps = sub.add_parser("steps", help="demos -> images + step records (slow, run once)")
    p_steps.add_argument("--raw", type=Path, default=RAW_DIR)
    p_steps.add_argument("--out", type=Path, default=OUT_DIR)
    p_steps.add_argument("--shards", type=int, nargs="*", help="only these shard numbers (default: all)")
    p_steps.add_argument("--max-image-tokens", type=int, default=DEFAULT_MAX_IMAGE_TOKENS)
    p_steps.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))

    p_sft = sub.add_parser("sft", help="step records -> chat rows for training/testing")
    p_sft.add_argument("--out", type=Path, default=OUT_DIR)
    p_sft.add_argument("--split", choices=SPLITS, required=True)
    p_sft.add_argument("--no-done", action="store_true", help="leave out the added end-of-demo 'done' rows")
    p_sft.add_argument("--limit", type=int, help="stop after this many rows (for small tests)")
    p_sft.add_argument("--max-history", type=int, default=DEFAULT_MAX_HISTORY)
    p_sft.add_argument("--step-instruction", action="store_true", help="also show the human's step note")

    args = parser.parse_args(argv)
    if args.stage == "steps":
        return build_steps(args.raw, args.out, args.shards, args.max_image_tokens, args.workers)
    return build_sft(args.out, args.split, not args.no_done, args.limit, args.max_history, args.step_instruction)


if __name__ == "__main__":  # required on macOS: worker processes re-import this file
    sys.exit(main())

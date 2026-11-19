"""Render test: run sample rows through Glimmer's real renderer and check what training sees.

The training script (cookbook sft_loop) turns each chat row into tokens with the
`muse_glimmer` renderer. This runs that same code locally, for free, and checks:

1. the screenshot survived: exactly one image in the tokens, with the token count
   screens.py predicts (one token per 28x28 square);
2. only the answer is trained: the tokens with training weight > 0 spell out the
   human's action and nothing from the prompt;
3. length: tokens per row, and that every row fits max_seq_len (sft_loop silently
   drops longer rows).

Downloads only Glimmer's tokenizer and processor settings from Hugging Face (a few MB,
never the weights). The repo has no Python files, so nothing from it runs here.

Usage:
    python -m phone_agent.render_check [--diverse 20] [--random 480] [--max-seq-len 4096]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from phone_agent.prompt import image_data_url, training_row
from phone_agent.screens import PATCH_PX

TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
RENDERER = "muse_glimmer"
STEPS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "android_control"


def pick_rows(records: list[dict], n_diverse: int, n_random: int, seed: int = 0) -> list[dict]:
    """A few rows of every action type plus the awkward cases, then a random sample."""
    rng = random.Random(seed)
    by_action = defaultdict(list)
    for r in records:
        by_action[json.loads(r["label"])["action"]].append(r)
    picks: dict[str, dict] = {}

    def add(r: dict) -> None:
        picks.setdefault(r["id"], r)

    add(max(records, key=lambda r: len(r["history"])))  # longest history (gets cut to 8)
    add(max(records, key=lambda r: len(r["goal"])))  # longest goal
    add(next(r for r in records if r["screen"] != [1080, 2400]))  # another phone size
    add(next((r for r in by_action["type"] if not r["label"].isascii()), by_action["type"][0]))
    kinds = sorted(by_action)
    while len(picks) < n_diverse and any(by_action[k] for k in kinds):
        for kind in kinds:
            if by_action[kind] and len(picks) < n_diverse:
                add(by_action[kind].pop(rng.randrange(len(by_action[kind]))))
    rest = [r for r in records if r["id"] not in picks]
    for r in rng.sample(rest, min(n_random, len(rest))):
        add(r)
    return list(picks.values())


def _load_renderer(max_seq_len: int):
    from training.utils.supervised import populate_render_worker_state

    state: dict = {}
    populate_render_worker_state(
        state,
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER,
        max_seq_len=max_seq_len,
        tokenizer_trust_remote_code=False,
    )
    return state["tokenizer"], state["renderer"]


def check_row(step: dict, row: dict, tokenizer, renderer, max_seq_len: int) -> dict:
    """Render one row like sft_loop does and return measurements + any problems."""
    from training.utils.supervised import render_messages_to_datums, rendered_chunk_spans

    rendered_list = render_messages_to_datums(row["messages"], renderer=renderer, reduction="mean")
    problems = []
    if len(rendered_list) != 1:
        problems.append(f"rendered into {len(rendered_list)} sequences, expected 1")
    rendered = rendered_list[0]
    ids, weights = rendered.token_ids, rendered.token_weights

    images = [s for s in rendered_chunk_spans(rendered) if s.kind == "image"]
    w, h = step["image_size"]
    expected_image_tokens = (w // PATCH_PX) * (h // PATCH_PX)
    if len(images) != 1:
        problems.append(f"{len(images)} images in the tokens, expected 1")
    elif images[0].token_count != expected_image_tokens:
        problems.append(f"image is {images[0].token_count} tokens, expected {expected_image_tokens}")
    image_tokens = sum(s.token_count for s in images)

    trained_ids = [t for t, wt in zip(ids, weights) if wt > 0]
    trained_text = tokenizer.decode(trained_ids)
    if step["label"] not in trained_text:
        problems.append(f"answer not in trained tokens: {trained_text!r}")
    leftover = trained_text.replace(step["label"], "", 1)
    if any(word in leftover for word in ("Goal", "Previous", "action", "Reasoning")):
        problems.append(f"prompt text is being trained: {leftover!r}")
    if any(wt > 0 for s in images for wt in weights[s.token_start : s.token_start + s.token_count]):
        problems.append("image tokens are being trained")
    if len(ids) > max_seq_len:
        problems.append(f"{len(ids)} tokens > max_seq_len {max_seq_len}: training would DROP this row")

    return {
        "id": step["id"],
        "total": len(ids),
        "image": image_tokens,
        "text": len(ids) - image_tokens,
        "trained": len(trained_ids),
        "trained_text": trained_text,
        "leftover": leftover,
        "problems": problems,
        "rendered": rendered,
    }


def _pct(values: list[int], q: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phone_agent.render_check")
    parser.add_argument("--split", default="train")
    parser.add_argument("--diverse", type=int, default=20)
    parser.add_argument("--random", type=int, default=480)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--data", type=Path, default=STEPS_DIR)
    args = parser.parse_args(argv)

    with open(args.data / "steps" / f"{args.split}.jsonl") as f:
        records = [json.loads(line) for line in f]
    sample = pick_rows(records, args.diverse, args.random)
    print(f"Loading the {RENDERER} renderer and {TOKENIZER_MODEL} tokenizer (small files only)...", flush=True)
    tokenizer, renderer = _load_renderer(args.max_seq_len)

    results = []
    for step in sample:
        row = training_row(step, image_data_url(args.data / step["image"]))
        results.append(check_row(step, row, tokenizer, renderer, args.max_seq_len))

    # Show one full example: the exact text the model trains on (image shortened).
    first = results[0]
    ids = first["rendered"].token_ids
    weights = first["rendered"].token_weights
    from training.utils.supervised import rendered_chunk_spans

    print(f"\n===== example row {first['id']} as the model sees it =====")
    for span in rendered_chunk_spans(first["rendered"]):
        if span.kind == "image":
            print(f"[IMAGE: {span.token_count} tokens]", end="")
        else:
            piece = ids[span.token_start : span.token_start + span.token_count]
            w = weights[span.token_start : span.token_start + span.token_count]
            # Mark the trained part with >>> <<< so it stands out.
            untrained = [t for t, x in zip(piece, w) if x <= 0]
            trained = [t for t, x in zip(piece, w) if x > 0]
            print(tokenizer.decode(untrained), end="")
            if trained:
                print(">>>" + tokenizer.decode(trained) + "<<<", end="")
    print("\n===== end (only the part between >>> <<< is trained) =====\n")

    bad = [r for r in results if r["problems"]]
    for r in bad[:10]:
        print("PROBLEM", r["id"], r["problems"])
    leftovers = sorted({r["leftover"] for r in results})
    totals = [r["total"] for r in results]
    print(f"rows checked:            {len(results)} ({args.diverse} hand-picked + random)")
    print(f"rows with problems:      {len(bad)}")
    print(f"trained text besides the answer: {leftovers}")
    print(f"image tokens per row:    {sorted({r['image'] for r in results})}")
    print(f"text tokens per row:     mean {statistics.mean(r['text'] for r in results):.0f}, max {max(r['text'] for r in results)}")
    print(f"trained tokens per row:  mean {statistics.mean(r['trained'] for r in results):.1f}")
    print(f"total tokens per row:    mean {statistics.mean(totals):.0f}, p50 {_pct(totals, .5)}, p95 {_pct(totals, .95)}, max {max(totals)}")
    print(f"max_seq_len {args.max_seq_len}: {'all rows fit' if max(totals) <= args.max_seq_len else 'SOME ROWS WOULD BE DROPPED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

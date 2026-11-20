"""Launch LoRA SFT of Muse Glimmer 30B on Fireworks Serverless Training.

Serverless LoRA SFT of Glimmer 30B is free for jobs started before
2026-12-01 00:00 Pacific. This script refuses anything outside that: another model,
full-parameter training (rank 0), or a launch after the deadline. It never samples
from the model, never deploys, and never computes a check-set loss.

Trials (small, run before the full job):
  overfit  32 rows of every action type, 25 passes. The loss must fall close to zero.
           Proves the answers reach the trainer and the model can learn them.
  speed    1,000 random train rows, one pass, full-run settings. Measures speed for
           the full-run time estimate, and checks that a checkpoint save works.

Both trials close their training session on exit (frees the account's one Glimmer
slot) and save nothing to the account. Everything goes to runs/<trial>-<time>/
(gitignored): the rows used, settings.json, train.log, metrics.jsonl, summary.json.

Usage (the training loop runs on this Mac: keep it awake and plugged in):
    caffeinate -is python main.py train overfit
    caffeinate -is python main.py train speed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from phone_agent.prompt import image_data_url, training_row
from phone_agent.render_check import RENDERER, TOKENIZER_MODEL, pick_rows

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "android_control"
RUNS_DIR = ROOT / "runs"
BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
FREE_UNTIL = datetime(2026, 12, 1, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
FULL_TRAIN_ROWS = 88_308


@dataclass(frozen=True)
class Trial:
    rows: int
    pick: str  # "diverse" (every action type) or "random"
    epochs: int
    batch_size: int
    schedule: dict
    save_checkpoint: bool
    render_workers: int
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    max_seq_len: int = 2048
    grad_clip_norm: float = 1.0
    seed: int = 0


FULL_SCHEDULE = {"type": "cosine", "warmup_ratio": 0.02, "min_lr_ratio": 0.1}
TRIALS = {
    # Constant learning rate: the point is to see the loss reach ~0, not a good model.
    "overfit": Trial(rows=32, pick="diverse", epochs=25, batch_size=8,
                     schedule={"type": "constant"}, save_checkpoint=False, render_workers=2),
    "speed": Trial(rows=1000, pick="random", epochs=1, batch_size=32,
                   schedule=FULL_SCHEDULE, save_checkpoint=True, render_workers=4),
}
OVERFIT_PASS_LOSS = 0.1


def check_free(trial: Trial, now: datetime | None = None) -> None:
    """Raise unless this launch is inside the free Glimmer LoRA SFT offer."""
    now = now or datetime.now(tz=FREE_UNTIL.tzinfo)
    if now >= FREE_UNTIL:
        raise SystemExit(f"Refusing: the free offer covers jobs started before {FREE_UNTIL.isoformat()}.")
    if trial.lora_rank <= 0:
        raise SystemExit("Refusing: only LoRA (rank > 0) is free; rank 0 means full-parameter training.")


def select_rows(records: list[dict], trial: Trial) -> list[dict]:
    if trial.pick == "diverse":
        return pick_rows(records, n_diverse=trial.rows, n_random=0, seed=trial.seed)[: trial.rows]
    return random.Random(trial.seed).sample(records, trial.rows)


def write_rows(steps: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for step in steps:
            row = training_row(step, image_data_url(DATA_DIR / step["image"]))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_config(trial: Trial, run_dir: Path, dataset: Path):
    import training.recipes.sft_loop as sft_loop
    from fireworks.training.sdk.training_spec import parse_lr_scheduler_spec
    from training.utils import WandBConfig
    from training.utils.runner import RunnerConfig

    return sft_loop.Config(
        serverless=True,
        close_session_on_exit=True,  # trials only: free the slot, nothing needs these checkpoints
        log_path=str(run_dir / "logs"),
        base_model=BASE_MODEL,
        tokenizer_model=TOKENIZER_MODEL,
        renderer_name=RENDERER,
        tokenizer_trust_remote_code=False,
        dataset=str(dataset),
        train_on_what="all_assistant_messages",
        lora_rank=trial.lora_rank,
        lora_alpha=trial.lora_alpha,
        learning_rate=trial.learning_rate,
        lr_scheduler=parse_lr_scheduler_spec(trial.schedule),
        epochs=trial.epochs,
        batch_size=trial.batch_size,
        max_seq_len=trial.max_seq_len,
        grad_clip_norm=trial.grad_clip_norm,
        seed=trial.seed,
        save_final_checkpoint=trial.save_checkpoint,
        dcp_save_interval=None if trial.save_checkpoint else 0,
        checkpoint_interval_minutes=None if trial.save_checkpoint else 0,
        output_model_id=None,  # never promote a trial
        generation_eval=False,  # sampling is paid
        render_workers=trial.render_workers,
        wandb=WandBConfig(entity=None),
        runner=RunnerConfig(
            status_file=str(run_dir / "status.json"),
            metadata_file=str(run_dir / "metadata.json"),
            metrics_file=str(run_dir / "metrics.jsonl"),
        ),
    )


# The log prints rounded losses, sometimes "-0.0000"; only its timestamps are used.
_STEP_LINE = re.compile(r"^(\S+ \S+) .*Step (\d+)/\d+ \| Loss: ")


def summarize(run_dir: Path, trial_name: str, trial: Trial, wall_s: float) -> dict:
    """Loss curve (exact values from metrics.jsonl) and speed (log timestamps).

    Speed is in rows, not tokens: the recipe's token counter skips image tokens
    (~987 of ~1,400 per row), so a token rate would understate the work.
    """
    loss_at = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        rec = json.loads(line) if line.strip() else {}
        if "train/ce_loss" in rec:
            loss_at[rec["step"]] = rec["train/ce_loss"]
    time_at = {}
    for line in (run_dir / "train.log").read_text().splitlines():
        if m := _STEP_LINE.match(line):
            time_at[int(m.group(2))] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
    out: dict = {"trial": trial_name, "rows": trial.rows, "epochs": trial.epochs, "batch_size": trial.batch_size,
                 "steps": len(loss_at), "wall_minutes": round(wall_s / 60, 1)}
    if not loss_at:
        return out
    losses = [max(0.0, loss_at[s]) for s in sorted(loss_at)]  # tiny negatives are float noise
    per_epoch = -(-trial.rows // trial.batch_size)
    epoch_means = [round(statistics.mean(losses[i : i + per_epoch]), 4) for i in range(0, len(losses), per_epoch)]
    out.update(first_loss=round(losses[0], 4), last_loss=round(losses[-1], 4), epoch_mean_loss=epoch_means)
    if len(time_at) >= 2:
        # From step 1 to the last step: excludes start-up, includes the shared pool's ups and downs.
        first, last = min(time_at), max(time_at)
        sec_per_step = (time_at[last] - time_at[first]) / (last - first)
        rows_per_s = trial.batch_size / sec_per_step
        out.update(sec_per_step=round(sec_per_step, 2), rows_per_sec=round(rows_per_s, 2))
        if trial_name == "speed":
            out["full_run_hours_estimate"] = round(FULL_TRAIN_ROWS / rows_per_s / 3600, 1)
    if trial_name == "overfit":
        out["passed"] = epoch_means[-1] < OVERFIT_PASS_LOSS
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python main.py train")
    parser.add_argument("trial", choices=sorted(TRIALS))
    args = parser.parse_args(argv)
    trial = TRIALS[args.trial]
    check_free(trial)
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("FIREWORKS_API_KEY is not set")
        return 1
    from phone_agent import fireworks_status

    if fireworks_status.main() != 0:
        print("Not launching: an open session or job exists (or the check failed).")
        return 3

    run_dir = RUNS_DIR / f"{args.trial}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    with open(DATA_DIR / "steps" / "train.jsonl") as f:
        records = [json.loads(line) for line in f]
    chosen = select_rows(records, trial)
    dataset = run_dir / "train.jsonl"
    write_rows(chosen, dataset)
    (run_dir / "row_ids.json").write_text(json.dumps([s["id"] for s in chosen]))
    (run_dir / "settings.json").write_text(json.dumps({"trial": args.trial, **trial.__dict__}, indent=2))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)],
    )
    print(f"run folder: {run_dir.relative_to(ROOT)}  rows: {len(chosen)}", flush=True)

    import training.recipes.sft_loop as sft_loop

    started = time.monotonic()
    result = sft_loop.main(build_config(trial, run_dir, dataset))
    summary = summarize(run_dir, args.trial, trial, time.monotonic() - started)
    summary["optimizer_steps_reported"] = result.get("steps")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0 if summary.get("passed", True) else 4


if __name__ == "__main__":
    sys.exit(main())

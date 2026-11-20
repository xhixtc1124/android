# Project history

What was done, in order, with the results. Newest entries at the bottom.
Money spent so far: **$0**.

## 2026-11-19: Setup

- Git repo with a `.gitignore` written before the first commit. It keeps data, model files, run logs and secrets out of git.
- The Fireworks training cookbook is pinned at commit `74e103ab` (`scripts/setup_cookbook.sh`). It runs from its own Python 3.11 environment.
- `source scripts/env.sh` turns off telemetry (Fireworks skill analytics, Hugging Face, Weights & Biases) and activates that environment.
- `python main.py check` runs read-only setup checks. `python main.py sessions` is a read-only check for open Fireworks training sessions, because the account allows only one Glimmer training run at a time.
- Decisions:
  - **Model:** Meta's Muse Glimmer 30B, LoRA fine-tuned on Fireworks Serverless Training. That is free for jobs started before 2026-12-01 00:00 Pacific.
  - **Testing:** always at full precision (BF16) on Fireworks. This Mac (16 GB) cannot run a 30B model, and a 4-bit local copy would blur the results.
  - **Action format:** taps use a 0–1000 grid, so one label format works on every screen size. Every action is one JSON object, for example `{"action": "tap", "x": 951, "y": 87}`.

## 2026-11-19: Data

- Downloaded **AndroidControl** (Google, Apache 2.0) from Google's official storage: 20 files, 49.9 GB, plus the official split lists (`scripts/download_android_control.sh`).
- Wrote readers that open the files without TensorFlow:
  - `wire.py` decodes the protobuf format by hand;
  - `tfrecord.py` reads the record files;
  - `a11y.py` reads each screen's accessibility tree (the boxes of buttons and text on screen).
- Wrote the converter (`convert.py`). Each step of a human demo becomes one row: screenshot + goal + the last 8 actions → the human's next action.
- Checks against the real data:
  - **Split counts match the official numbers:** 13,594 / 137 / 1,543 demos for train / validation / test. The 9 train demos missing from the 13,603 had no actions at all.
  - **Scroll direction:** `scroll down` means "show what is further down". The free-text step notes sometimes say the opposite; the action labels are consistent.
  - **Taps land on something:** 99.9% of human taps fall inside a box on screen. For scoring, the target is the smallest *clickable* box under the tap. The smallest box of any kind is clickable only 46% of the time, so it would make a poor target.
  - **Four screen sizes** appear (1080×2400 is most common). Labels are checked to come out the same on every size.
- Full conversion: about 15 minutes, 8.7 GB of screenshots.

  | Split | Rows | Of which human steps | Of which added "done" rows |
  |---|---:|---:|---:|
  | train | 88,308 | 74,714 | 13,594 |
  | validation | 827 | | |
  | test | 9,987 | 8,444 | |

  Each demo's last screen gets an added `{"action": "done", "status": "complete"}` row so the model learns to stop. These rows are marked `end_of_demo` and scored separately from human steps.

- Choices that can still change:
  - screenshots shrunk to at most 1,000 image tokens (1080×2400 → 588×1316);
  - the last 8 actions in the prompt;
  - goal only, without the per-step notes;
  - no reasoning text in the answers, with the prompt set to "Reasoning strength: low."

## 2026-11-19: Render test (free, local)

`python main.py render-check` runs rows through Glimmer's real renderer, the same code training uses.

- **500 train rows, 0 problems.**
- **The screenshot is kept:** exactly one image per row, 987 tokens (966–968 for other screen sizes). That matches the "one token per 28×28 square" count.
- **Only the answer is trained:** the trained tokens are ` to=user<|message|>` + the action JSON + `<|eot|>`, about 19 tokens per row. The prompt, goal and image are never trained.
- **Length:** about 1,400 tokens per row, at most 1,557. A 2,048-token limit fits every row. The training script silently drops longer rows, so this matters.
- **Size:** one pass over all train rows is about 124M tokens. That would cost about $720 at the normal $5.86 per million, but it is free before the deadline.

## 2026-11-20: Training plan

Settings for the full run (not launched yet):

| Setting | Value | Note |
|---|---|---|
| LoRA rank / alpha | 16 / 32 | small adapter layers; the 30B base is unchanged |
| Learning rate | 1e-4 | |
| Schedule | warm up over the first 2% of steps, then cosine decay to 10% | |
| Epochs | 1 | more passes mainly memorize the training apps |
| Batch size | 32 rows | about 2,760 updates for one pass |
| Max sequence length | 2,048 | longest row is 1,557 |
| Gradient clipping | 1.0 | |
| Weight decay, Adam beta2 | 0.01, 0.95 | cookbook defaults |
| Checkpoints | every 30 minutes | the training loop runs on this Mac, so a crash or sleep can resume |
| Check-set loss during training | off | Fireworks does not say whether it is free; worst case about $10–30 |

**What is free:** Fireworks' docs and pricing page say "Muse Glimmer 30B SFT is free through November 30" on the Serverless Training API. The cookbook says only LoRA SFT is covered, for jobs started before 2026-12-01 00:00 Pacific.

**What is not free:** running the model to test it (sampling), and rented GPUs (deployments). Each gets its own cost estimate and approval.

## 2026-11-20: Trial runs (free)

Two small runs check the setup before the multi-day full run. `python main.py train overfit|speed`. Both close their training session when they finish and save nothing to the account.

### Trial A: overfitting test

Train on 32 rows (a mix of every action type) over and over until the model memorizes them.
- **What it checks:** the answers reach the trainer and the model can learn them. If the loss doesn't fall close to zero, something in the setup is broken.
- **What it doesn't check:** whether the model looks at the screenshot. With only 32 rows, the goal and past actions are enough to tell rows apart.
- **Pass rule (set before the run):** the average loss over the last pass is below 0.1.

Result: *pending*

### Trial B: speed test

1,000 random train rows, one pass, with the full-run settings.
- **What it checks:** how fast Fireworks trains, which gives the real time estimate for the full run. Also that the loss starts falling and that saving a checkpoint works.

Result: *pending*

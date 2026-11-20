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

**Result: PASSED.** Ran 13:10–13:29 on 2026-11-20: 100 steps in 18.6 minutes, at no cost.

| Step | 1 | 5 | 20 | 40 | 60–90 (typical) | last pass (97–100) average |
|---|---|---|---|---|---|---|
| Loss | 1.44 | 0.15 | 0.056 | 0.0007 | about 0.0000003 | **0.044** |

- **The loss fell from 1.44 to almost zero, so the model memorized all 32 answers.** In plain terms: at the start the model gave each correct answer token about a 24% chance. By step 40 it was above 99.9%.
- **Late spikes.** After the loss reached about zero, a few single steps jumped back up: step 36 (0.16), 91 (0.11), 97 (0.05) and 100 (0.12). Each time the next step was back near zero.
  - **Cause:** the optimizer (Adam) sizes each update by how big recent corrections have been. Once the answers are perfect, corrections become tiny: the gradient size was 0.0005 at step 80. An ordinary small correction then gets blown up into an oversized step, and the step after fixes it.
  - **Effect on the full run:** none expected. It only happens near zero loss on data seen over and over. The full run sees each row once, and its learning rate shrinks toward the end.
- **Gradient clipping was active on almost every early step.** Gradient sizes were 50–85 in the first steps, far above the 1.0 cap. Adam rescales updates anyway, so this changes little, but the cap is acting as a constant rescale, not a rare safety guard.
- **Speed:** 11 seconds per step of 8 rows on average, ranging from 3 to 22 seconds. The training machines are shared with other users, so speed varies.
- **Clean-up:** the training session closed when the run ended; the session check found nothing open afterwards.
- **Fix made:** the first summary marked this run "failed" by mistake. Its log parser skipped 3 steps the log printed as "-0.0000" (rounding noise), which shifted the per-pass averages. It now reads exact losses from the metrics file, and a test covers this case.

### Trial B: speed test

1,000 random train rows, one pass, with the full-run settings.
- **What it checks:** how fast Fireworks trains, which gives the real time estimate for the full run. Also that the loss starts falling and that saving a checkpoint works.

**Result: done, and everything worked.** Ran 13:30–13:36 on 2026-11-20: 32 steps in 5.0 minutes, including start-up and the checkpoint save, at no cost.

- **Speed: 7.9 seconds per step of 32 rows, about 4 rows per second.** Most steps took 2–5 seconds, with a slow one of about 20 seconds every few steps. A step took no longer than in Trial A, even with 4× the rows, so bigger batches use the machines better.
- **Full-run estimate: about 6–9 hours** for all 88,308 rows (about 2,760 steps). That's 6 hours at this pace and 8.5 hours at Trial A's slower pace, and longer if the shared machines are busy.
- **Loss: 1.22 at step 1, 0.25 by step 4, then 0.2–0.3 for the rest** (0.19 at the last step).
  - The quick early drop is the model learning the answer format.
  - What remains is the hard part: which element to tap and exactly where. That needs far more than 1,000 rows.
  - Every row here was seen only once, so this is an honest learning signal, but it is not a test score.
- **Checkpoint saving works:** the resumable copy took 5 seconds and the testable copy 6 seconds (about 33 seconds in all).
- **No rows dropped** for being too long.
- **The learning-rate schedule ran as planned:** 1e-4 at the start, down to 1e-5 at the last step.
- **Gradient sizes were 90–4,320, so the 1.0 cap rescaled every step.** One odd batch at step 6 reached 4,320. Large numbers are expected because the trainer adds the gradient up over all ~600 answer tokens in a batch. The cap stays: with Adam it mainly stops one odd batch like step 6 from swamping the next updates.
- **Clean-up:** the session closed when the run ended; the session check found nothing open.

## 2026-11-20: Full training run (free)

Started at 14:08 on 2026-11-20 with `python main.py train full`, kept awake with `caffeinate`, on AC power.

- **Data:** all 88,308 train rows, one pass, 2,760 steps of 32 rows. The training file (`sft/train.jsonl`, 10.9 GB, screenshots embedded) took 46 seconds to build.
- **Settings:** as in the training plan above.
- **Checkpoints:** every 30 minutes, or every 1/20 of the estimated run time if that is longer, so a crash can resume from the last one. The session stays open when the script exits so a resume is possible, but only right away: the session expires soon after its run stops.
- **Saving the result:** the final adapter is saved to the account as model `glimmer-android-sft-20261120-1408`.
  - **Why:** a training session's checkpoints stop loading soon after its run ends, but a saved model stays.
  - **Consequence:** the paid test can run later on a rented full-precision GPU instead of having to happen inside the session.

Progress so far:

| Time | Step | Loss | Pace |
|---|---|---|---|
| 14:10 | 1 | 1.28 | |
| 14:39 | 41 | 0.31 | about 45 s per step |
| 15:09 | 99 | 0.18 | about 31 s per step |
| 15:37 | 138 | 0.21 | about 43 s per step; first checkpoint saved (60 s resumable copy + 72 s testable copy) |

- **Slower than the speed test.** Steps take 30–45 seconds, against 8 in the speed test an hour earlier.
  - **Cause:** Fireworks' shared training machines. The Mac was 85% idle, the training processes sat at 0% CPU waiting for replies, upload was near zero between steps, and there were no retry or capacity errors.
  - **Effect:** at this pace the run takes about 24–33 hours instead of 6–9. It is still free, because the offer counts the start date.
- **Mac memory:** swap is nearly full (about 13 of 14 GB), mostly from browser and editor windows open for days. The training processes use about 0.2 GB.

Result: *in progress*

## Where things stand (2026-11-20)

- Both free trials passed. The setup is ready for the full run.
- **Training tokens used by the trials:** about 2.5 million, all free. **Money spent:** $0.
- **The full training run started at 14:08.** It's free, and slower than planned because the shared machines are busy (see above).
  - The tuned adapter is saved to the account when training ends. The paid offline test can therefore run afterwards on a rented full-precision GPU, not inside the session. That test needs its own cost estimate and approval first.
  - The Mac has to stay awake, plugged in, with the lid open, for the whole run.

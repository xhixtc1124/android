# Android phone agent (Muse Glimmer 30B)

Fine-tune an open vision model, Meta's Muse Glimmer 30B, into an Android phone-control agent. Measure it properly: base vs tuned, on held-out data including apps the model never saw. Then run it live on an Android emulator. The model runs at full precision (BF16) for every test, so 4-bit compression never affects the results.

**Claim:** built and measured a phone agent end to end. Not a new research result.

## Plan

1. **Data.** Convert AndroidControl human demos into chat rows: screenshot + goal + recent actions → next action as JSON.
2. **Train.** LoRA SFT on the Fireworks Serverless Training API, using the cookbook's `sft_loop`.
3. **Offline test.** Step accuracy of base vs tuned on AndroidControl's held-out splits, including unseen apps, with a paired McNemar test.
4. **Live test.** AndroidWorld task success, base vs tuned. The emulator runs locally; the model runs on a Fireworks dedicated deployment (BF16), with base and tuned served from the same deployment.
5. **Write-up.** Results, cost, and what didn't work.

## Data

[AndroidControl](https://github.com/google-research/google-research/tree/master/android_control) (Google, Apache 2.0): 15,283 human demos, 833 apps. Official splits: 13,603 train / 137 validation / 1,543 test demos. Test groups: same apps (IDD), unseen apps, unseen tasks, unseen app categories.

```bash
bash scripts/download_android_control.sh        # ~50 GB into data/raw (gitignored)
python main.py convert steps                     # screenshots -> 588x1316 JPEGs + step records
python main.py convert sft --split train        # chat rows for training
```

Each step becomes: screenshot + goal + earlier actions → the human's next action, e.g. `{"action": "tap", "x": 951, "y": 87}` (0–1000 scale, so different screen sizes share one label format). The last screen of every demo gets an added `done` label, marked `end_of_demo` and reported separately from human steps.

## Status

Data converter done. Training next.

# On-device Android phone agent (Muse Glimmer 30B)

Fine-tune an open vision model, Meta's Muse Glimmer 30B, into an Android phone-control agent. Measure it properly: base vs tuned, on held-out data including apps the model never saw. Then run it live on an Android emulator.

**Claim:** built and measured a phone agent end to end. Not a new research result.

## Plan

1. **Data.** Convert AndroidControl human demos into chat rows: screenshot + goal + recent actions → next action as JSON.
2. **Train.** LoRA SFT on the Fireworks Serverless Training API, using the cookbook's `sft_loop`.
3. **Offline test.** Step accuracy of base vs tuned on AndroidControl's held-out splits, including unseen apps, with a paired McNemar test.
4. **Live test.** AndroidWorld task success, base vs tuned, on an Android emulator.
5. **Write-up.** Results, cost, and what didn't work.

## Status

Setup only. Results will go here.

# Temporal world models: a small reimplementation

A practice reimplementation of the **time probes** from
["Language Models Represent Space and Time"](https://arxiv.org/abs/2310.02207)
(Gurnee & Tegmark, 2023). A ridge-regression probe is trained on a language
model's residual-stream activations (last entity token, every layer) to predict
when something happened, e.g. the year a historical figure died.

Original code and data: <https://github.com/wesg52/world-models> (MIT license).
The three entity CSVs in `data/entity_datasets/` (`historical_figure`, `art`,
`headline`) are copied from that repository. See `LICENSE`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reimplementation/temporal_world_models.py --dataset historical_figure --n 2000 --model EleutherAI/pythia-410m
```

The first run downloads the model and caches activations to
`reimplementation/outputs/` (the `.pt` cache files are git-ignored). Optional
flags: `--val_select`, `--sweep_alpha`, `--seeds N`, `--compare MODEL ...`.

## Results (pythia-410m, 2,000 samples, historical figures)

- Test R² rises through roughly layer 7, with a small step at layer 12, then plateaus at about 0.5.
- At layer 18: test R² 0.528, Spearman 0.712.
- Figures are in `reimplementation/outputs/`; notes are in `reimplementation/summary.md`.

## Caveats

Single run, one small model (the paper uses Llama-2), 2,000 of ~37.5k rows,
empty prompt. By default the best layer is chosen on the test set, which is
slightly optimistic (use `--val_select` to avoid that). See the status note at
the top of `reimplementation/temporal_world_models.py` for what was scaffolded
and what was written by hand.

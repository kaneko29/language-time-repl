"""
Reimplementation practice: the TIME probes from
"Language Models Represent Space and Time" (Gurnee & Tegmark, 2023).

Goal: reproduce two kinds of figures for a temporal dataset:
  1. "Linear plot"  -> probe test R^2 (and Spearman) vs. layer        (paper Fig. 2)
  2. "Scatter plot" -> true year vs. probe-predicted year, test set   (paper Fig. 1 / 3)

Pipeline (each step is one function below):
  load entities -> build prompt strings -> run model, grab last-token hidden
  state at every layer -> fit a ridge probe per layer -> plot

Run (from the repo root):
  python reimplementation/temporal_world_models.py --dataset historical_figure --n 2000

Status: the core pipeline runs end to end (year conversion, last-token
activations, per-layer ridge probe, R^2-by-layer plot, true-vs-predicted
scatter). 


Known limitations / next steps:
  - By default the "best layer" is chosen by test R^2, which is slightly
    optimistic; use --val_select to choose it on a validation split instead.
  - Results so far are from a single run of pythia-410m on 2,000 of ~37.5k
    rows with the empty prompt: try the "When did " prefix, the full dataset,
    and a larger model (e.g. Llama-2-7B on OSCAR).
  - Only historical_figure so far: extend to art and headline.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "entity_datasets")
OUT_DIR = os.path.join(REPO_ROOT, "reimplementation", "outputs")

# Llama-2 is gated and too big for a laptop; a small Pythia model is enough to
# see the effect (the paper also notes smaller models are noisier).
DEFAULT_MODEL = "EleutherAI/pythia-410m"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

# For each time dataset: which column is the target, and how to turn a row
# into the text the model reads. These mirror the repo's prompts
# (feature_datasets/historical.py, time_art.py, headline.py).
DATASETS = {
    "historical_figure": {
        "target_col": "death_year",
        "prompt": "",                       # repo also tries "When did "
        "entity_text": lambda row: row["name"],
    },
    "art": {
        "target_col": "release_date",
        "prompt": "",                       # repo also tries "When was the release date of "
        "entity_text": lambda row: f"{row['creator']}'s {row['title']}",
    },
    "headline": {
        "target_col": "pub_date",
        "prompt": "",                       # repo also tries "Publication date of: "
        "entity_text": lambda row: row["headline"],
    },
}


def load_entities(dataset, n=None, seed=0):
    """Load the CSV, add a numeric `year` column, optionally subsample n rows."""
    cfg = DATASETS[dataset]
    df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset}.csv"))
    df = df.dropna(subset=[cfg["target_col"]])

    # TODO: make a float column `year` from cfg["target_col"].
    #   - death_year is already a number.
    #   - release_date / pub_date are date strings; pd.to_datetime(...) then
    #     convert to fractional years (e.g. year + dayofyear / 365.25).
    #     Hint: pub_date has a timezone, so pass utc=True.
    if dataset == "historical_figure":
        df["year"] = df[cfg["target_col"]].astype(float)
    else:
        df["year"] = pd.to_datetime(df[cfg["target_col"]], utc=True).dt.year + pd.to_datetime(df[cfg["target_col"]], utc=True).dt.dayofyear / 365.25

    
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def build_texts(df, dataset):
    cfg = DATASETS[dataset]
    return [cfg["prompt"] + cfg["entity_text"](row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# 2. Activations
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()
    return tokenizer, model

@torch.no_grad()
def get_last_token_activations(texts, tokenizer, model, device, batch_size=32):
    """
    Returns a tensor of shape (n_layers, n_texts, d_model): the residual stream
    at the LAST token of each text, after every transformer block.

    Because the entity is at the end of the text and padding is on the right,
    "last entity token" == "last non-padding token".
    """
    per_batch = []
    for i in tqdm(range(0, len(texts), batch_size), desc="activations"):
        batch = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        out = model(**batch, output_hidden_states=True)

        # out.hidden_states is a tuple of (n_layers + 1) tensors, each
        # (batch, seq_len, d_model). Index 0 is the embedding output -- skip it.
        hidden = torch.stack(out.hidden_states[1:])  # (n_layers, batch, seq, d_model)

        # TODO: find the index of the last real token in each row.
        #   Hint: batch["attention_mask"] is 1 for real tokens, 0 for padding.
        last_indices = batch["attention_mask"].sum(dim=1) - 1

        # TODO: gather hidden[:, b, last_ix[b], :] for every row b
        #   -> shape (n_layers, batch, d_model).
        #   Hint: torch.arange(batch_len) together with last_ix as fancy indexes.
        batch_len = hidden.shape[1]
        last_acts = hidden[:, torch.arange(batch_len), last_indices, :]
        
        per_batch.append(last_acts.float().cpu())
    return torch.cat(per_batch, dim=1)


def cached_activations(args, texts, seed=0):
    """Running the model is the slow part, so save activations to disk once.

    The subsample depends on `seed`, so the cache key includes it (seed 0 keeps
    the original filename so existing caches stay valid).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.model.split("/")[-1]
    seed_tag = "" if seed == 0 else f".s{seed}"
    path = os.path.join(OUT_DIR, f"{args.dataset}.{tag}.n{args.n}{seed_tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    tokenizer, model = load_model(args.model, args.device)
    acts = get_last_token_activations(texts, tokenizer, model, args.device, args.batch_size)
    torch.save(acts, path)
    return acts


# ---------------------------------------------------------------------------
# 3. Probes
# ---------------------------------------------------------------------------

def fit_probe(X, y, is_test, alpha):
    """
    Fit one linear probe: activations -> year.

    Returns (test_predictions, metrics_dict).

    TODO:
      - split X, y into train / test with the boolean mask `is_test`
      - standardize y using TRAIN mean/std only (the repo does this; it keeps
        the ridge penalty on a sensible scale)
      - fit sklearn Ridge(alpha=alpha) on train
      - predict on test and un-standardize back to years
      - metrics: R^2 (r2_score) and Spearman (spearmanr(...).correlation),
        maybe also mean absolute error in years
    """
    #split x and y
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    #standarize using train mean
    mean_train = y_train.mean()
    std_train = y_train.std()
    y_train_std = (y_train - mean_train) / std_train

    #fit ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train_std)

    # predict on test and un-standarize
    y_pred_std = ridge.predict(X_test)
    y_pred = y_pred_std * std_train + mean_train

    #metrics
    r2 = r2_score(y_test, y_pred)
    spearman = spearmanr(y_test, y_pred).correlation
    mae = np.mean(np.abs(y_test - y_pred))
    metrics = {"test_r2": r2, "spearman": spearman, "test_mae": mae}

    return y_pred, metrics


def probe_all_layers(acts, df, alpha):
    """Fit a probe at each layer. Returns (results_df, {layer: test_preds}).
    TODO: 
    - results_df has one row per layer with test_r2, spearman, test_mae.
    """
    is_test = df["is_test"].values.astype(bool)
    y = df["year"].values
    rows, preds = [], {}
    for layer in range(acts.shape[0]):
        test_pred, metrics = fit_probe(acts[layer].numpy(), y, is_test, alpha)
        rows.append({"layer": layer, **metrics})
        preds[layer] = test_pred
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plot_metric_by_layer(results, metric, title, path):
    """
    "Linear plot": x = layer, y = metric (e.g. test R^2).

    TODO: ax.plot(results.layer, results[metric], marker="o"); label axes;
    set a y-limit like (0, 1). Bonus: plot several prompts or models on the same
    axes, with layer normalized to [0, 1] so different-depth models line up
    (see performance_by_prompt in analysis/probe_plots.py).
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # plot metric vs later
    ax.plot(results.layer, results[metric], marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_true_vs_pred(true_years, pred_years, title, path):
    """
    "Scatter plot": x = true year, y = predicted year, one dot per test entity.

    TODO:
      - ax.scatter(true, pred, s=small, alpha=low)   # thousands of points overlap
      - draw the y = x line so you can see over/under-prediction
      - optional: color points by something interesting (e.g. page_views or
        occupation for historical figures) to see who the model knows well
      - the paper's Fig. 1 plots the projection against the true date in
        exactly this way
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # scatter plot for true vs predicted
    ax.scatter(true_years, pred_years, s=10, alpha=0.5)
    ax.plot([true_years.min(), true_years.max()], [true_years.min(), true_years.max()], color='red', linestyle='--')
    ax.set_xlabel("True Year")
    ax.set_ylabel("Predicted Year")
    ax.set_title(title)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Some extensions
# ---------------------------------------------------------------------------

def _train_val_split(df, val_frac=0.2, seed=0):
    """Split the TRAIN rows (is_test == False) into train/val. Test rows are never touched.

    Returns (is_test, y_train_rows, val_mask), where val_mask is a boolean mask
    over the train rows (in their original order).
    """
    is_test = df["is_test"].values.astype(bool)
    y_tr = df["year"].values[~is_test]
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(len(y_tr), dtype=bool)
    val_mask[rng.choice(len(y_tr), size=int(len(y_tr) * val_frac), replace=False)] = True
    return is_test, y_tr, val_mask


def choose_layer_on_validation(acts, df, alpha, val_frac=0.2, seed=0):
    """
    Pick the best layer using a validation split taken from the TRAIN rows
    only, so the test set isn't used for model selection.

    Returns (best_layer, val_df) where val_df has one row per layer with
    val_r2 / val_mae.
    """
    is_test, y_tr, val_mask = _train_val_split(df, val_frac, seed)
    rows = []
    for layer in range(acts.shape[0]):
        X_tr = acts[layer].numpy()[~is_test]
        # fit_probe treats `val_mask` rows as its "test" rows, so these are validation scores
        _, m = fit_probe(X_tr, y_tr, val_mask, alpha)
        rows.append({"layer": layer, "val_r2": m["test_r2"], "val_mae": m["test_mae"]})
    val_df = pd.DataFrame(rows)
    return int(val_df["val_r2"].idxmax()), val_df


def sweep_alpha(acts, df, layer, alphas, val_frac=0.2, seed=0):
    """
    Fit the probe at one layer for several ridge strengths (e.g. 0.1x, 1x,
    10x d_model) and report validation R^2 for each.
    """
    is_test, y_tr, val_mask = _train_val_split(df, val_frac, seed)
    X_tr = acts[layer].numpy()[~is_test]
    rows = []
    for a in alphas:
        _, m = fit_probe(X_tr, y_tr, val_mask, a)
        rows.append({"alpha": a, "val_r2": m["test_r2"], "val_mae": m["test_mae"]})
    return pd.DataFrame(rows)


def run_multiple_seeds(args, seeds=(0, 1, 2)):
    """
    Repeat the subsample + probe for several seeds and return mean and spread
    of test R^2 per layer, to get error bars on the line plot.
    """
    per_seed = []
    for seed in seeds:
        df = load_entities(args.dataset, n=args.n, seed=seed)
        texts = build_texts(df, args.dataset)
        acts = cached_activations(args, texts, seed=seed)
        results, _ = probe_all_layers(acts, df, args.alpha or acts.shape[-1])
        results["seed"] = seed
        per_seed.append(results)
    all_results = pd.concat(per_seed)
    return (all_results.groupby("layer")["test_r2"]
            .agg(mean="mean", std="std").reset_index())


def plot_models_compared(results_by_model, metric, path):
    """
    Overlay R^2-by-layer curves for several models on one axis, with layer
    normalized to [0, 1] so models of different depth line up.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, res in results_by_model.items():
        ax.plot(res["layer"] / res["layer"].max(), res[metric], marker="o", label=name)
    ax.set_xlabel("Relative layer (0 = first, 1 = last)")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="historical_figure", choices=DATASETS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n", type=int, default=2000, help="subsample size (full sets are ~30k)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=None,
                   help="ridge strength; default = d_model, like the repo")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--val_select", action="store_true",
                   help="choose the layer on a validation split instead of the test set")
    p.add_argument("--sweep_alpha", action="store_true",
                   help="print validation R^2 for several alphas at the chosen layer")
    p.add_argument("--seeds", type=int, default=1,
                   help="if > 1, repeat over this many seeds and print mean/std test R^2")
    p.add_argument("--compare", nargs="+", default=None,
                   help="extra models to overlay on one R^2-by-layer plot")
    args = p.parse_args()

    df = load_entities(args.dataset, n=args.n)
    texts = build_texts(df, args.dataset)
    print(f"{len(texts)} texts, e.g. {texts[:3]}")

    acts = cached_activations(args, texts)          # (n_layers, n, d_model)
    alpha = args.alpha or acts.shape[-1]

    results, preds = probe_all_layers(acts, df, alpha)
    print(results.to_string(index=False))

    tag = f"{args.dataset}.{args.model.split('/')[-1]}"
    plot_metric_by_layer(results, "test_r2", f"{tag}: test R² by layer",
                         os.path.join(OUT_DIR, f"{tag}.r2_by_layer.png"))

    if args.val_select:
        best, val_df = choose_layer_on_validation(acts, df, alpha)
        print(f"layer chosen on validation: {best} "
              f"(test R² there: {results.loc[best, 'test_r2']:.3f})")
    else:
        best = int(results["test_r2"].idxmax())   # NOTE: picked on test set (optimistic)
    true_test = df.loc[df["is_test"].astype(bool), "year"].values
    plot_true_vs_pred(true_test, preds[best], f"{tag}: layer {best}",
                      os.path.join(OUT_DIR, f"{tag}.scatter_layer{best}.png"))

    if args.sweep_alpha:
        print(sweep_alpha(acts, df, best, [alpha * f for f in (0.1, 0.3, 1, 3, 10)]).to_string(index=False))

    if args.seeds > 1:
        print(run_multiple_seeds(args, seeds=range(args.seeds)).to_string(index=False))

    if args.compare:
        by_model = {args.model.split("/")[-1]: results}
        for m in args.compare:
            m_args = argparse.Namespace(**{**vars(args), "model": m})
            m_acts = cached_activations(m_args, texts)
            m_results, _ = probe_all_layers(m_acts, df, args.alpha or m_acts.shape[-1])
            by_model[m.split("/")[-1]] = m_results
        plot_models_compared(by_model, "test_r2",
                             os.path.join(OUT_DIR, f"{args.dataset}.models_compared.png"))


if __name__ == "__main__":
    main()

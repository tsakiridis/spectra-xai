# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare grid, random and Bayesian hyperparameter search for SVR.

The default grid of SVR has 420 combinations of C, epsilon and gamma. Random and
Bayesian search explore the same ranges with a fixed budget of candidates;
Bayesian search (Optuna's TPE sampler) proposes each candidate from the results
of the previous ones and stops candidates that are clearly behind after a few
folds. Bayesian search needs the optional extra: ``pip install spectraxai[optuna]``.
"""

import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.models import Model, StandardModel
from spectraxai.plotting import style
from spectraxai.spectra import SpectralPreprocessing as SP
from spectraxai.utils import metrics

dataset = load_GR_SSL().preprocess(SP.ABS)
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
train, test = dataset.subset(idx_trn), dataset.subset(idx_tst)

rows = []
searches = {}
for search in ("grid", "random", "bayes"):
    # random and Bayesian search evaluate n_iter candidates (ignored by grid search)
    model = StandardModel(Model.SVR, random_state=0, search=search, n_iter=40)
    t0 = time.time()
    model.fit(train)
    elapsed = time.time() - t0
    results = model.search_results_
    assert results is not None
    # Bayesian search records how many folds each trial ran before any pruning
    folds = results["n_folds"].to_numpy() if "n_folds" in results else 5
    fits = int(np.sum(np.broadcast_to(folds, (len(results),))))
    test_metrics = cast(
        dict[str, float], metrics(test.Y.ravel(), model.predict(test.X).ravel())
    )
    rows.append(
        {
            "search": search,
            "candidates": len(results),
            "fits": fits,
            "time (s)": round(elapsed, 1),
            "CV R2": round(float(model.best_score or np.nan), 3),
            "test RMSE": round(float(test_metrics["RMSE"]), 3),
            "test R2": round(float(test_metrics["R2"]), 3),
        }
    )
    searches[search] = model
print(pd.DataFrame(rows).set_index("search").to_string())

# How the best cross-validated score improves with the number of candidates
with style():
    fig, ax = plt.subplots(figsize=(7, 4))
    for search in ("random", "bayes"):
        results = searches[search].search_results_
        assert results is not None
        scores = results["mean_test_score"].to_numpy(dtype=float)
        if "state" in results:  # pruned trials have a partial score only
            scores = np.where(results["state"] == "complete", scores, -np.inf)
        ax.plot(
            np.arange(1, len(scores) + 1),
            np.maximum.accumulate(scores),
            label=f"{search} search",
        )
    ax.axhline(
        searches["grid"].best_score or np.nan,
        color="grey",
        ls="--",
        label="grid search (420 candidates)",
    )
    ax.set_xlabel("Candidates evaluated")
    ax.set_ylabel("Best cross-validated R²")
    ax.legend()
    fig.tight_layout()
plt.show()

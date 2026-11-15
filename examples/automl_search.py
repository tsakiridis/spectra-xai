# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Search the pre-processing, the model and its hyperparameters together.

Instead of comparing a hand-picked list of pre-treatments for one model,
`PreprocessingSearch` explores pre-processings (including continuous options such
as the fractional derivative order or the Savitzky–Golay window), models and their
hyperparameters in one Bayesian (Optuna) study. Every candidate is cross-validated
as a whole pipeline, so stateful steps such as MSC never see the validation folds.
The best pipeline is then tested on an independent Kennard–Stone test set.

Needs the optional extra: ``pip install spectraxai[optuna]``.
"""

from typing import cast

import pandas as pd

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.models import Model
from spectraxai.search import PreprocessingSearch
from spectraxai.spectra import SpectralPreprocessing as SP
from spectraxai.utils import metrics

dataset = load_GR_SSL()
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
train, test = dataset.subset(idx_trn), dataset.subset(idx_tst)

# Options given as ranges are tuned too (the window, the derivative order)
smooth = (SP.SG0, {"window_length": ("int", 5, 21), "polyorder": 2})
candidates = {
    "ABS": SP.ABS,
    "ABS + SNV": [SP.ABS, SP.SNV],
    "ABS + MSC": [SP.ABS, SP.MSC],
    "ABS + SG1 (window)": [
        SP.ABS,
        (SP.SG1, {"window_length": ("int", 5, 31), "polyorder": 2}),
    ],
    "ABS + SG0 + FD (order)": [
        SP.ABS,
        smooth,
        (SP.FD, {"order": ("float", 0.0, 2.0)}),
    ],
    "ABS + CR": [SP.ABS, SP.CR],
}
names = list(candidates)
search = PreprocessingSearch(
    models=[Model.PLS, Model.SVR, Model.RIDGE],
    preprocessings=list(candidates.values()),
    n_iter=80,
    cv=5,
    random_state=0,
).fit(train)

# The best candidate of every pre-processing and model. Bayesian search spends
# its trials where results look promising, so some combinations stay untried.
results = search.results_[search.results_["state"] == "complete"]
best = results.loc[
    results.groupby(["preprocessing_index", "model"])["mean_test_score"].idxmax()
]
pd.set_option("display.max_colwidth", 60)
print("Best cross-validated R² per pre-processing and model (training set):")
print(
    best.pivot(index="preprocessing_index", columns="model", values="mean_test_score")
    .rename(index=lambda i: names[i])
    .round(3)
    .to_string()
)
print("\nBest:", search.best_params_)
print("Trials by state:", search.results_["state"].value_counts().to_dict())

test_metrics = metrics(test.Y.ravel(), search.predict(test).ravel())
print(
    "Independent test set: "
    + f"N={test_metrics['N']}, "
    + ", ".join(
        f"{k}={float(cast(float, test_metrics[k])):.3f}" for k in ("RMSE", "R2", "RPIQ")
    )
)

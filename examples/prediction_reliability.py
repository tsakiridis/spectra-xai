# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Report predictions with prediction intervals, and flag unusual spectra.

A prediction is only as good as the spectrum it comes from. This script:

1. checks the test spectra against the applicability domain of the training
   spectra (Hotelling's T² and Q residuals), including a few spectra altered on
   purpose to stand in for unusual samples (a contaminant band, a baseline
   shift), which are flagged;
2. gives every prediction a 90% conformal prediction interval (CV+), and checks
   how often the intervals contain the measured value.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.diagnostics import ApplicabilityDomain, DomainStatistic
from spectraxai.models import Model, StandardModel
from spectraxai.plotting import style
from spectraxai.spectra import SpectralPreprocessing as SP
from spectraxai.uncertainty import ConformalMethod, ConformalRegressor
from spectraxai.uncertainty import interval_coverage

dataset = load_GR_SSL().preprocess(SP.ABS)
idx_trn, idx_tst = dataset.train_test_split(DatasetSplit.KENNARD_STONE, 0.8)
train, test = dataset.subset(idx_trn), dataset.subset(idx_tst)

# Unusual spectra: an absorption band never seen in training, and an offset
wavelengths = np.asarray(dataset.wavelengths, dtype=float)
band = 0.15 * np.exp(-(((wavelengths - 1700) / 40) ** 2))
unusual = np.vstack([test.X[:5] + band, test.X[5:10] + 0.3])

# 1. Applicability domain of the training spectra
domain = ApplicabilityDomain(confidence=0.99).fit(train)
print(f"{domain.n_components_} principal components")
# Fewer than 99% of the training spectra fall inside: Kennard-Stone puts the most
# extreme spectra into the training set on purpose.
print(f"training spectra inside: {domain.inside(train).mean():.1%}")
print(f"test spectra inside:     {domain.inside(test).mean():.1%}")
t2, q = domain.statistics(unusual).T
print(
    pd.DataFrame(
        {
            "alteration": ["new band"] * 5 + ["offset"] * 5,
            "T2 / limit": (t2 / domain.t2_limit_).round(2),
            "Q / limit": (q / domain.q_limit_).round(2),
            "inside": domain.inside(unusual),
        }
    ).to_string()
)

# 2. Prediction intervals around PLS
pls = StandardModel(Model.PLS, init_hyperparameters={"n_components": 12})
intervals = {}
for method in ConformalMethod:
    conformal = ConformalRegressor(pls, method=method, alpha=0.1, random_state=0)
    conformal.fit(train)
    pred, lower, upper = conformal.predict(test, return_interval=True)
    intervals[method] = (pred, lower, upper)
    coverage = interval_coverage(test.Y, lower, upper)[0]
    print(
        f"{method}: mean width {np.mean(upper - lower):.3f}, "
        f"coverage of the test set {coverage:.1%} (target 90%)"
    )

with style():
    fig, (ax_domain, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
    domain.set_params(statistic=DomainStatistic.T2_OR_Q)
    domain.plot(np.vstack([test.X, unusual]), ax=ax_domain)

    pred, lower, upper = intervals[ConformalMethod.CV_PLUS]
    order = np.argsort(pred)
    y = test.Y.ravel()[order]
    inside = (y >= lower[order]) & (y <= upper[order])
    x = np.arange(len(order))
    ax_pred.fill_between(
        x, lower[order], upper[order], color="C0", alpha=0.25, label="90% interval"
    )
    ax_pred.plot(x, pred[order], color="C0", label="prediction")
    ax_pred.scatter(x[inside], y[inside], s=8, color="0.2", label="measured")
    ax_pred.scatter(
        x[~inside], y[~inside], s=14, color="C3", label="measured, outside interval"
    )
    ax_pred.set_xlabel("Test samples, sorted by prediction")
    ax_pred.set_ylabel(dataset.Y_names[0])
    ax_pred.set_title("CV+ prediction intervals")
    ax_pred.legend()
    fig.tight_layout()
plt.show()

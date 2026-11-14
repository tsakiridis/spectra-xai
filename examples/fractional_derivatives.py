# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose a fractional derivative order for PLS by cross-validation.

Fractional-order derivatives (Grünwald–Letnikov) fill the gap between the spectra
(order 0), the first derivative (order 1) and the second (order 2). Soil
spectroscopy studies often find the best order in between. Here every order from 0
to 2 in steps of 0.25 is applied to lightly smoothed absorbance spectra and
evaluated with 5-fold cross-validated PLS; the mean spectrum at a few orders is
shown next to the error curve.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectraxai.data import load_GR_SSL
from spectraxai.dataset import DatasetSplit
from spectraxai.models import Model, StandardModel
from spectraxai.plotting import style
from spectraxai.spectra import SpectralPreprocessing as SP

dataset = load_GR_SSL()
wavelengths = np.asarray(dataset.wavelengths)
idx_trn, _ = dataset.train_test_split(DatasetSplit.CROSS_VALIDATION, 5, random_state=0)

smooth = (SP.SG0, {"window_length": 11, "polyorder": 2})
orders = np.round(np.arange(0, 2.01, 0.25), 2)
pretreatments = [[SP.ABS, smooth, (SP.FD, {"order": float(v)})] for v in orders]

pls = StandardModel(Model.PLS, random_state=0)
results = pls.fit_and_predict_multiple(dataset, pretreatments, idx_trn=idx_trn)
# Read the order back from each row's pre-processing (stored as a string)
results["order"] = [SP.init_class(p)[-1][1]["order"] for p in results["pre_process"]]
summary = pd.DataFrame(
    results.groupby("order")[["RMSE", "R2"]].agg(["mean", "std"]).round(3)
)
summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
print(summary.to_string())
best = float(summary.index[np.argmin(summary["RMSE mean"].to_numpy())])
print(f"lowest cross-validated RMSE at order {best}")

with style():
    fig, (ax_err, ax_spec) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_err.errorbar(
        summary.index, summary["RMSE mean"], yerr=summary["RMSE std"], marker="o"
    )
    ax_err.axvline(best, color="grey", ls="--", lw=1)
    ax_err.set_xlabel("Derivative order")
    ax_err.set_ylabel("Cross-validated RMSE")

    # The first bands have few preceding bands to difference against, so their
    # derivative is dominated by the spectrum itself: leave them out of the plot
    skip = 5
    for v in sorted({0.0, 0.5, 1.0, best}):
        spectra = dataset.preprocess([SP.ABS, smooth, (SP.FD, {"order": v})])
        mean = spectra.X[:, skip:].mean(axis=0)
        ax_spec.plot(
            wavelengths[skip:],
            mean / np.abs(mean).max(),
            label=f"order {v:g}",
        )
    ax_spec.set_xlabel(dataset.spectral_label)
    ax_spec.set_ylabel("Mean spectrum (scaled to max 1)")
    ax_spec.legend()
    fig.tight_layout()
plt.show()

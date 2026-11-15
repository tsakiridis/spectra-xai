# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conformal prediction intervals."""

from typing import Any

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

from spectraxai.dataset import Dataset
from spectraxai.dissimilarity import Dissimilarity, pairwise_dissimilarity
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.models import Model, StandardModel
from spectraxai.spectra import SpectralPreprocessing as SP
from spectraxai.uncertainty import (
    ConformalMethod,
    ConformalRegressor,
    interval_coverage,
)


def _data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    # heavy-tailed noise: no Gaussian assumption is needed
    return X, X @ np.arange(1, 7) + rng.standard_t(3, size=n)


@pytest.mark.parametrize("method", list(ConformalMethod), ids=str)
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_coverage_is_close_to_one_minus_alpha(
    method: ConformalMethod, alpha: float
) -> None:
    coverage = []
    for seed in range(40):
        X, y = _data(240, seed)
        conformal = ConformalRegressor(
            Ridge(), method=method, alpha=alpha, random_state=seed
        ).fit(X[:120], y[:120])
        _, lower, upper = conformal.predict(X[120:], return_interval=True)
        coverage.append(interval_coverage(y[120:], lower, upper)[0])
    assert np.mean(coverage) == pytest.approx(1 - alpha, abs=0.025)


def test_split_intervals(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    conformal = ConformalRegressor(PLSRegression(3), random_state=0).fit(X, y)
    assert len(conformal.calibration_indices_) == round(0.2 * len(X))
    pred, lower, upper = conformal.predict(X, return_interval=True)
    np.testing.assert_allclose(upper - lower, 2 * conformal.quantile_[0])
    np.testing.assert_allclose(pred, conformal.estimator_.predict(X).ravel())
    np.testing.assert_allclose(conformal.predict(X), pred)


def test_cv_plus_intervals(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    conformal = ConformalRegressor(
        PLSRegression(3), method=ConformalMethod.CV_PLUS, cv=4, random_state=0
    ).fit(X[:60], y[:60])
    assert len(conformal.fold_estimators_) == 4
    assert set(conformal.fold_of_sample_) == {0, 1, 2, 3}
    pred, lower, upper = conformal.predict(X[60:], return_interval=True)
    assert np.all(lower <= upper)
    assert np.ptp(upper - lower) > 0  # widths vary between samples


def test_cv_plus_by_hand() -> None:
    """Lower and upper bounds as defined by Barber et al. (2021)."""
    X, y = _data(39, 0)
    alpha = 0.2
    conformal = ConformalRegressor(
        Ridge(), method=ConformalMethod.CV_PLUS, alpha=alpha, cv=3, random_state=1
    ).fit(X, y)
    x = _data(1, 5)[0]
    mu = np.array(
        [conformal.fold_estimators_[k].predict(x)[0] for k in conformal.fold_of_sample_]
    )
    R = conformal.residuals_.ravel()
    n = len(y)
    lo = np.sort(mu - R)[int(np.floor(alpha * (n + 1))) - 1]
    hi = np.sort(mu + R)[int(np.ceil((1 - alpha) * (n + 1))) - 1]
    _, lower, upper = conformal.predict(x, return_interval=True)
    assert lower[0] == pytest.approx(lo) and upper[0] == pytest.approx(hi)


@pytest.mark.parametrize("method", list(ConformalMethod), ids=str)
def test_reproducible(method: ConformalMethod) -> None:
    X, y = _data(80, 0)

    def run() -> Any:
        conformal = ConformalRegressor(Ridge(), method=method, random_state=3)
        return conformal.fit(X, y).predict(X, return_interval=True)

    for a, b in zip(run(), run()):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", list(ConformalMethod), ids=str)
def test_multiple_outputs(synthetic2d: Dataset, method: ConformalMethod) -> None:
    conformal = ConformalRegressor(PLSRegression(3), method=method, random_state=0).fit(
        synthetic2d
    )
    pred, lower, upper = conformal.predict(synthetic2d, return_interval=True)
    assert pred.shape == lower.shape == upper.shape == (synthetic2d.n_samples, 2)
    assert interval_coverage(synthetic2d.Y, lower, upper).shape == (2,)


@pytest.mark.parametrize(
    "model",
    [
        StandardModel(Model.PLS, init_hyperparameters={"n_components": 3}),
        StandardModel(Model.LOCAL_PLS, init_hyperparameters={"k": 20}),
        StandardModel(Model.RIDGE, cv=3),  # tuned in every fit
    ],
    ids=["PLS", "LOCAL_PLS", "RIDGE-tuned"],
)
def test_wraps_standard_models(synthetic: Dataset, model: StandardModel) -> None:
    conformal = ConformalRegressor(
        model, method=ConformalMethod.CV_PLUS, cv=3, random_state=0
    ).fit(synthetic)
    pred, lower, upper = conformal.predict(synthetic.X[:5], return_interval=True)
    assert pred.shape == (5,) and np.all(lower <= upper)
    assert model.pipeline_ is None  # the given model is not fitted


def test_too_few_samples_give_infinite_intervals() -> None:
    X, y = _data(10, 0)
    for method in ConformalMethod:
        conformal = ConformalRegressor(Ridge(), method=method, alpha=0.05, cv=2)
        with pytest.warns(UserWarning, match="too few"):
            conformal.fit(X, y)
            _, lower, upper = conformal.predict(X, return_interval=True)
        assert np.all(np.isinf(upper))


def test_interval_coverage() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert interval_coverage(y, y - 1, y + 1)[0] == 1.0
    assert interval_coverage(y, [1, 1, 1, 1], [2, 2, 2, 2])[0] == 0.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "split"}, "ConformalMethod member"),
        ({"alpha": 0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"calibration_size": 1}, "calibration_size"),
    ],
)
def test_invalid_parameters(kwargs: dict, match: str) -> None:
    X, y = _data(30, 0)
    with pytest.raises(SpectraXAIValueError, match=match):
        ConformalRegressor(Ridge(), **kwargs).fit(X, y)


def test_cv_plus_needs_a_partition() -> None:
    X, y = _data(30, 0)
    cv = ShuffleSplit(5, test_size=0.3, random_state=0)
    with pytest.raises(SpectraXAIValueError, match="overlap|cover"):
        ConformalRegressor(Ridge(), method=ConformalMethod.CV_PLUS, cv=cv).fit(X, y)


def test_inputs(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="y should not"):
        ConformalRegressor(Ridge()).fit(synthetic, synthetic.Y)
    with pytest.raises(SpectraXAIValueError, match="y is required"):
        ConformalRegressor(Ridge()).fit(synthetic.X)


# -- fit_and_predict -------------------------------------------------------------


def test_fit_and_predict_with_intervals(synthetic: Dataset) -> None:
    model = StandardModel(Model.RIDGE, random_state=0, cv=3)
    res = model.fit_and_predict(
        synthetic, SP.ABS, idx_trn=np.arange(60), return_intervals=True
    )
    assert res.loc[0, "interval_alpha"] == 0.1
    assert res.loc[0, "interval_width"] > 0
    assert 0 <= res.loc[0, "interval_coverage"] <= 1
    plain = model.fit_and_predict(synthetic, SP.ABS, idx_trn=np.arange(60))
    assert "interval_width" not in plain.columns
    assert res.loc[0, "RMSE"] == pytest.approx(plain.loc[0, "RMSE"])


def test_wider_intervals_for_smaller_alpha(synthetic2d: Dataset) -> None:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    widths = [
        model.fit_and_predict(
            synthetic2d,
            idx_trn=np.arange(60),
            return_intervals=True,
            interval_alpha=alpha,
        )["interval_width"].to_numpy()
        for alpha in (0.3, 0.05)
    ]
    assert np.all(widths[1] > widths[0])  # per output


def test_intervals_by_fold_and_preprocessing(synthetic: Dataset) -> None:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    folds = np.array([np.arange(40), np.arange(40, 80)])
    res = model.fit_and_predict(synthetic, idx_tst=folds, return_intervals=True)
    assert list(res["fold"]) == [1, 2]
    assert bool(res["interval_width"].notna().all())
    res = model.fit_and_predict_multiple(
        synthetic, [SP.NONE, SP.ABS], idx_trn=np.arange(60), return_intervals=True
    )
    assert bool(res["interval_coverage"].notna().all())


def test_no_intervals_with_a_precomputed_matrix(synthetic: Dataset) -> None:
    D = pairwise_dissimilarity(synthetic.X, method=Dissimilarity.EUCLIDEAN)
    model = StandardModel(
        Model.LOCAL_PLS,
        init_hyperparameters={"k": 20, "dissimilarity": Dissimilarity.PRECOMPUTED},
    )
    with pytest.raises(SpectraXAIValueError, match="intervals"):
        model.fit_and_predict(
            synthetic, idx_trn=np.arange(60), dissimilarity=D, return_intervals=True
        )

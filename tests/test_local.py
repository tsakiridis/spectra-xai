# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local (memory-based learning) regressors."""

import importlib.util
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from spectraxai.dataset import Dataset
from spectraxai.dissimilarity import Dissimilarity, SpectralDissimilarity
from spectraxai.local import LocalGPRRegressor, LocalPLSRegressor, pls_vip
from spectraxai.models import Model, StandardModel
from spectraxai.search import PreprocessingSearch
from spectraxai.spectra import SpectralPreprocessing as SP

needs_optuna = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Bayesian search needs the optuna extra",
)
METRICS = [m for m in Dissimilarity if m is not Dissimilarity.PRECOMPUTED]


def _two_populations(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two kinds of spectra in which the same band relates oppositely to y."""
    rng = np.random.default_rng(seed)
    bands = np.arange(60)

    def peak(center: float, width: float) -> np.ndarray:
        return np.exp(-(((bands - center) / width) ** 2))

    group = np.arange(n) % 2
    depth = rng.uniform(0, 1, size=n)
    y = np.where(group == 0, depth, 1 - depth)
    X = (
        0.5 * np.outer(group, peak(45, 8))  # what tells the populations apart
        + np.outer(depth, peak(20, 4))  # what y depends on, with opposite signs
        + rng.uniform(0, 0.2, size=(n, 1)) * peak(5, 6)
        + rng.normal(0, 0.005, size=(n, bands.size))
    )
    return X, y


# -- the local regressors ---------------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
def test_all_neighbours_is_global_pls(
    synthetic: Dataset, metric: Dissimilarity
) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    local = LocalPLSRegressor(k=len(X), n_components=4, dissimilarity=metric).fit(X, y)
    pls = PLSRegression(n_components=4).fit(X, y)
    np.testing.assert_allclose(local.predict(X), pls.predict(X), rtol=1e-10)
    np.testing.assert_allclose(local.vip(), pls_vip(pls), rtol=1e-10)


def test_all_neighbours_with_multiple_outputs(synthetic2d: Dataset) -> None:
    X, Y = synthetic2d.X, synthetic2d.Y
    local = LocalPLSRegressor(k=1000, n_components=3).fit(X, Y)
    assert local.k_ == len(X)  # k is capped at the training size
    pls = PLSRegression(n_components=3).fit(X, Y)
    np.testing.assert_allclose(local.predict(X), pls.predict(X), rtol=1e-10)
    assert local.predict(X).shape == (len(X), 2)


@pytest.mark.parametrize(
    "estimator",
    [
        LocalPLSRegressor(k=20, n_components=4),
        LocalPLSRegressor(
            k=20, n_components=4, dissimilarity=Dissimilarity.CORRELATION
        ),
        LocalGPRRegressor(k=20, noise_variance=0.01),
    ],
    ids=["local-pls", "local-pls-correlation", "local-gpr"],
)
def test_local_models_beat_a_global_model_on_two_populations(estimator: Any) -> None:
    X, y = _two_populations()
    cv = KFold(5, shuffle=True, random_state=0)
    global_r2 = cross_val_score(PLSRegression(n_components=4), X, y, cv=cv).mean()
    local_r2 = cross_val_score(estimator, X, y, cv=cv).mean()
    assert global_r2 < 0.5
    assert local_r2 > 0.9


@pytest.mark.parametrize("cls", [LocalPLSRegressor, LocalGPRRegressor])
def test_deterministic_and_independent_of_batching(
    synthetic: Dataset, cls: type
) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    first = cls(k=25).fit(X[:60], y[:60]).predict(X[60:])
    again = cls(k=25).fit(X[:60], y[:60])
    np.testing.assert_array_equal(first, again.predict(X[60:]))
    one_by_one = [again.predict(X[i : i + 1])[0] for i in range(60, len(X))]
    np.testing.assert_allclose(first, one_by_one, rtol=1e-10)


def test_neighbours_by_correlation_ignore_intensity(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    local = LocalPLSRegressor(k=5, dissimilarity=Dissimilarity.CORRELATION).fit(X, y)
    (train, _), *_ = local._neighbourhoods(3 * X[7:8] + 1)
    assert 7 in train
    euclidean = LocalPLSRegressor(k=5, dissimilarity=Dissimilarity.EUCLIDEAN).fit(X, y)
    (train, _), *_ = euclidean._neighbourhoods(3 * X[7:8] + 1)
    assert 7 not in train


def test_components_are_capped_by_the_neighbourhood(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    local = LocalPLSRegressor(k=4, n_components=50).fit(X, y)
    assert np.all(np.isfinite(local.predict(X[:5])))
    (train, _), *_ = local._neighbourhoods(X[:1])
    assert local._fit_local(train).n_components == 3


def test_vip_of_given_spectra(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    local = LocalPLSRegressor(k=20, n_components=3).fit(X, y)
    vip = local.vip(X[:10])
    assert vip.shape == (X.shape[1],) and np.all(vip >= 0)
    singles = np.mean([local.vip(X[i : i + 1]) for i in range(10)], axis=0)
    np.testing.assert_allclose(vip, singles)


def test_dissimilarity_options_are_cloned(synthetic: Dataset) -> None:
    measure = SpectralDissimilarity(Dissimilarity.PCA_EUCLIDEAN, n_components=3)
    local = LocalPLSRegressor(k=10, dissimilarity=measure)
    local.fit(synthetic.X, synthetic.Y.ravel())
    assert local.dissimilarity_ is not None
    assert local.dissimilarity_.projection_.n_components_ == 3
    assert not hasattr(measure, "reference_")  # the given one is left unfitted


@pytest.mark.parametrize(
    "estimator, match",
    [
        (LocalPLSRegressor(k=1), "k should"),
        (LocalPLSRegressor(k=2.5), "k should"),  # type: ignore[arg-type]
        (LocalPLSRegressor(n_components=0), "n_components"),
        (LocalPLSRegressor(dissimilarity=cast(Any, "SAM")), "Dissimilarity member"),
        (LocalGPRRegressor(noise_variance=0), "noise_variance"),
    ],
)
def test_invalid_parameters(synthetic: Dataset, estimator: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        estimator.fit(synthetic.X, synthetic.Y.ravel())


@parametrize_with_checks(
    [
        LocalPLSRegressor(k=10, n_components=2),
        LocalPLSRegressor(k=10, dissimilarity=Dissimilarity.CORRELATION),
        LocalGPRRegressor(k=10),
    ]
)
def test_sklearn_conventions(estimator: Any, check: Callable[[Any], None]) -> None:
    check(estimator)


# -- through StandardModel and the searches ---------------------------------------


def test_standard_model_with_local_pls(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.LOCAL_PLS,
        grid_search_hyperparameters={"k": [20, 40], "n_components": [2, 4]},
        cv=3,
    )
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert res.loc[0, "R2"] > 0.5
    assert res.loc[0, "k"] in (20, 40) and res.loc[0, "n_components"] in (2, 4)
    assert res.loc[0, "importance_method"] == "VIP"
    assert res.loc[0, "feature_importance"].shape == (synthetic.n_features,)
    assert model.search_results_ is not None and len(model.search_results_) == 4
    assert model.pipeline_ is not None and "scaler" not in model.pipeline_.named_steps


def test_standard_model_with_local_gpr(synthetic: Dataset) -> None:
    model = StandardModel(Model.LOCAL_GPR, init_hyperparameters={"k": 30})
    res = model.fit_and_predict(
        synthetic, idx_trn=np.arange(60), return_permutation_importance=True
    )
    assert res.loc[0, "R2"] > 0.5
    assert res.loc[0, "noise_variance"] == pytest.approx(1e-3)
    assert res.loc[0, "importance_method"] == "permutation"


def test_default_grids(synthetic: Dataset) -> None:
    for model in (Model.LOCAL_PLS, Model.LOCAL_GPR):
        sm = StandardModel(model, cv=2).fit(synthetic)
        assert sm.search_results_ is not None and len(sm.search_results_) >= 9
        assert cast(StandardModel, clone(sm)).get_params()["model"] == model


def test_multiple_outputs_and_preprocessings(synthetic2d: Dataset) -> None:
    model = StandardModel(Model.LOCAL_PLS, init_hyperparameters={"k": 30})
    res = model.fit_and_predict_multiple(
        synthetic2d, [SP.NONE, SP.ABS], idx_trn=np.arange(60)
    )
    assert len(res) == 4 and np.all(res["R2"] > 0.3)


@needs_optuna
def test_bayes_search_of_local_pls(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.LOCAL_PLS, search="bayes", n_iter=6, cv=3, random_state=0
    ).fit(synthetic)
    assert model.best_hyperparameters is not None
    assert 20 <= model.best_hyperparameters["k"] <= 500
    assert 2 <= model.best_hyperparameters["n_components"] <= 20
    measures = [Dissimilarity.SAM, Dissimilarity.CORRELATION]
    model = StandardModel(
        Model.LOCAL_PLS,
        search="bayes",
        n_iter=4,
        cv=3,
        random_state=0,
        search_space={"k": ("int", 10, 40), "dissimilarity": measures},
    ).fit(synthetic)
    assert model.best_hyperparameters is not None
    assert model.best_hyperparameters["dissimilarity"] in measures
    assert isinstance(model.best_hyperparameters["dissimilarity"], Dissimilarity)


def test_in_preprocessing_search(synthetic: Dataset) -> None:
    search = PreprocessingSearch(
        [Model.PLS, StandardModel(Model.LOCAL_PLS, init_hyperparameters={"k": 30})],
        [SP.NONE, SP.ABS],
        search="grid",
        cv=3,
    ).fit(synthetic)
    assert "LOCAL_PLS" in set(search.results_["model"])

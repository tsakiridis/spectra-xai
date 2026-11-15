# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Joint search of pre-processing, model and hyperparameters."""

import importlib.util
from typing import Any, cast

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.models import _DEFAULT_GRIDS, Model, StandardModel
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.search import PreprocessingSearch
from spectraxai.spectra import SpectralPreprocessing as SP

needs_optuna = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Bayesian search needs the optuna extra",
)


@pytest.fixture
def scattered() -> Dataset:
    """Spectra with strong multiplicative scatter that only MSC/SNV remove."""
    rng = np.random.default_rng(0)
    wavelengths = np.arange(400, 2500, 50)
    centers = np.array([700.0, 1400.0, 1900.0, 2200.0])
    peaks = np.exp(-(((wavelengths[None, :] - centers[:, None]) / 80) ** 2))
    depths = rng.uniform(0.05, 0.3, size=(90, centers.size))
    signal = 1 + depths @ peaks
    scale = rng.uniform(0.3, 3.0, size=(90, 1))
    X = scale * signal + rng.normal(0, 1e-4, size=signal.shape)
    y = depths @ np.array([1.0, -2.0, 0.5, 1.5])
    return Dataset(X, y, wavelengths, ["y"])


def _fixed_pls(n: int = 4) -> StandardModel:
    return StandardModel(Model.PLS, init_hyperparameters={"n_components": n})


# -- Grid search --------------------------------------------------------------------


def test_grid_finds_the_planted_preprocessing(scattered: Dataset) -> None:
    search = PreprocessingSearch(
        [_fixed_pls()], [SP.NONE, SP.MSC, SP.SNV], search="grid", cv=3
    ).fit(scattered)
    assert search.best_params_["preprocessing"] in (SP.MSC, SP.SNV)
    scores = dict(
        zip(search.results_["preprocessing"], search.results_["mean_test_score"])
    )
    assert scores[str(SP.MSC)] > scores[str(SP.NONE)] + 0.05


def test_stateful_steps_are_learnt_on_training_folds(scattered: Dataset) -> None:
    cv = KFold(3, shuffle=True, random_state=0)
    search = PreprocessingSearch(
        [_fixed_pls(3)], [[SP.MSC, SP.MEAN_CENTER]], search="grid", cv=cv
    ).fit(scattered)
    pipe = make_pipeline(
        SpectralPreprocessor([SP.MSC, SP.MEAN_CENTER]), PLSRegression(n_components=3)
    )
    y = scattered.Y.ravel()
    expected = cross_val_score(pipe, scattered.X, y, cv=cv).mean()
    assert search.best_score_ == pytest.approx(expected)
    # learning MSC on all samples first would give a different (leaky) score
    leaky = scattered.preprocess([SP.MSC, SP.MEAN_CENTER])
    leaky_score = cross_val_score(
        PLSRegression(n_components=3), leaky.X, y, cv=cv
    ).mean()
    assert search.best_score_ != pytest.approx(leaky_score, abs=1e-12)


def test_grid_covers_every_combination(synthetic: Dataset) -> None:
    ridge_grid = len(_DEFAULT_GRIDS[Model.RIDGE]["alpha"])
    preps = [SP.NONE, SP.ABS, [SP.ABS, SP.SNV]]
    search = PreprocessingSearch(
        [Model.RIDGE, _fixed_pls()], preps, search="grid", cv=3
    ).fit(synthetic)
    results = search.results_
    assert len(results) == len(preps) * (ridge_grid + 1)
    assert set(results["model"]) == {"RIDGE", "PLS"}
    assert set(results["preprocessing"]) == {str(p) for p in preps}
    for index, method in zip(results["preprocessing_index"], results["preprocessing"]):
        assert method == str(preps[index])
    best = results.loc[results["rank_test_score"] == 1].iloc[0]
    assert best["mean_test_score"] == pytest.approx(search.best_score_)
    assert best["model"] == search.best_params_["model"]


def test_scaled_models_get_a_scaler(synthetic: Dataset) -> None:
    search = PreprocessingSearch(
        [StandardModel(Model.RIDGE, init_hyperparameters={"alpha": 1.0})],
        search="grid",
        cv=3,
    ).fit(synthetic)
    pipe = cast(Pipeline, search.best_estimator_)
    assert type(pipe.named_steps["scale"]).__name__ == "StandardScaler"
    search = PreprocessingSearch([_fixed_pls()], search="grid", cv=3).fit(synthetic)
    assert search.best_estimator_.named_steps["scale"] == "passthrough"


def test_best_pipeline_predicts_new_data(synthetic: Dataset) -> None:
    train, test = synthetic.subset(np.arange(60)), synthetic.subset(np.arange(60, 80))
    search = PreprocessingSearch(
        [_fixed_pls()], [SP.ABS, [SP.ABS, SP.SNV]], search="grid", cv=3
    ).fit(train)
    pipe = search.best_estimator_
    assert isinstance(pipe.named_steps["pre"], SpectralPreprocessor)
    np.testing.assert_allclose(search.predict(test), pipe.predict(test.X))
    assert search.score(test.X, test.Y.ravel()) > 0.8
    np.testing.assert_allclose(
        pipe.named_steps["pre"].wavelengths, np.asarray(synthetic.wavelengths)
    )


def test_custom_and_repeated_models(synthetic: Dataset) -> None:
    models = [
        _fixed_pls(2),
        _fixed_pls(6),
        StandardModel(Model.CUSTOM, estimator=Ridge(), init_hyperparameters={}),
    ]
    search = PreprocessingSearch(models, search="grid", cv=3).fit(synthetic)
    assert set(search.results_["model"]) == {"PLS", "PLS_2", "CUSTOM"}


def test_random_search(synthetic: Dataset) -> None:
    kwargs: dict[str, Any] = dict(search="random", n_iter=6, cv=3, random_state=0)
    preps = [SP.NONE, SP.ABS]
    first = PreprocessingSearch([Model.RIDGE, Model.PLS], preps, **kwargs).fit(
        synthetic
    )
    again = PreprocessingSearch([Model.RIDGE, Model.PLS], preps, **kwargs).fit(
        synthetic
    )
    assert len(first.results_) == 6
    assert first.results_.equals(again.results_)


def test_ranges_in_options_need_bayes(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="bayes"):
        PreprocessingSearch(
            [_fixed_pls()], [(SP.FD, {"order": ("float", 0, 2)})], search="grid"
        ).fit(synthetic)


# -- Bayesian search ----------------------------------------------------------------


@needs_optuna
def test_bayes_finds_the_planted_preprocessing(scattered: Dataset) -> None:
    search = PreprocessingSearch(
        [Model.PLS, Model.RIDGE],
        # (absorbance + detrending would also remove multiplicative scatter)
        [SP.NONE, SP.MSC, SP.DETREND],
        n_iter=25,
        cv=3,
        random_state=0,
    ).fit(scattered)
    assert search.best_params_["preprocessing"] == SP.MSC
    assert search.best_score_ > 0.9
    assert {"n_folds", "state", "rank_test_score"} <= set(search.results_)


@needs_optuna
def test_bayes_tunes_options_inside_the_preprocessing(synthetic: Dataset) -> None:
    search = PreprocessingSearch(
        [_fixed_pls()],
        [[SP.ABS, (SP.FD, {"order": ("float", 0.0, 2.0), "n_terms": 20})]],
        n_iter=12,
        cv=3,
        random_state=0,
    ).fit(synthetic)
    step = search.best_params_["preprocessing"][1]
    assert step[0] == SP.FD and 0 <= step[1]["order"] <= 2
    assert step[1]["n_terms"] == 20  # fixed options are kept
    orders = [SP.init_class(p)[1][1]["order"] for p in search.results_["preprocessing"]]
    assert len(set(orders)) > 5
    assert set(search.results_["preprocessing_index"]) == {0}
    assert SP.init_class(str(search.best_params_["preprocessing"])) == (
        search.best_params_["preprocessing"]
    )


@needs_optuna
def test_bayes_is_reproducible(synthetic: Dataset) -> None:
    def run() -> PreprocessingSearch:
        return PreprocessingSearch(
            [Model.PLS, Model.RIDGE],
            [SP.ABS, [SP.ABS, SP.SNV]],
            n_iter=10,
            cv=3,
            random_state=3,
        ).fit(synthetic)

    first, again = run(), run()
    cols = ["preprocessing", "model", "mean_test_score", "state"]
    assert first.results_[cols].equals(again.results_[cols])
    assert first.best_params_ == again.best_params_


@needs_optuna
def test_invalid_combinations_fail_single_trials(synthetic: Dataset) -> None:
    search = PreprocessingSearch(
        [_fixed_pls()],
        [(SP.SG1, {"window_length": ("int", 3, 9), "polyorder": ("int", 1, 6)})],
        n_iter=20,
        cv=3,
        random_state=0,
    ).fit(synthetic)
    states = set(search.results_["state"])
    assert "failed" in states and "complete" in states
    window = search.best_params_["preprocessing"][1]
    assert window["polyorder"] < window["window_length"]


@needs_optuna
def test_bayes_with_multiple_outputs(synthetic2d: Dataset) -> None:
    search = PreprocessingSearch(
        [Model.PLS], [SP.ABS], n_iter=5, cv=3, random_state=0
    ).fit(synthetic2d)
    assert search.predict(synthetic2d.X).shape == (synthetic2d.n_samples, 2)


# -- scikit-learn conventions and errors ---------------------------------------------


def test_clone_and_nested_cross_validation(synthetic: Dataset) -> None:
    search = PreprocessingSearch([_fixed_pls()], [SP.NONE, SP.ABS], search="grid", cv=3)
    params = cast(PreprocessingSearch, clone(search)).get_params()
    assert params["search"] == "grid" and params["cv"] == 3
    scores = cross_val_score(search, synthetic.X, synthetic.Y.ravel(), cv=3)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"models": []}, "models"),
        ({"models": ["PLS"]}, "models"),
        ({"preprocessings": []}, "preprocessings"),
        ({"search": "hyperband"}, "search"),
        ({"n_iter": 0}, "n_iter"),
    ],
)
def test_invalid_arguments(synthetic: Dataset, kwargs: dict, match: str) -> None:
    with pytest.raises(SpectraXAIValueError, match=match):
        PreprocessingSearch(**kwargs).fit(synthetic)


def test_inputs(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="y should not"):
        PreprocessingSearch().fit(synthetic, synthetic.Y)
    with pytest.raises(SpectraXAIValueError, match="y is required"):
        PreprocessingSearch().fit(synthetic.X)

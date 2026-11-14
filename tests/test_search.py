# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Random and Bayesian hyperparameter search (`spectraxai.search`)."""

import builtins
import importlib.util
import warnings
from typing import Any, cast

import numpy as np
import pytest
from scipy.stats import uniform
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.models import (
    _DEFAULT_GRIDS,
    _DEFAULT_SPACES,
    Model,
    StandardModel,
)
from spectraxai.search import BayesianSearchCV, random_distributions

needs_optuna = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None,
    reason="Bayesian search needs the optuna extra",
)


def _params(model: StandardModel) -> dict[str, Any]:
    assert model.best_hyperparameters is not None
    return {k: np.asarray(v).item() for k, v in model.best_hyperparameters.items()}


# -- Grid search is unchanged -------------------------------------------------------


def test_grid_search_is_the_default_and_unchanged(synthetic: Dataset) -> None:
    default = StandardModel(Model.RIDGE, random_state=0, cv=3).fit(synthetic)
    explicit = StandardModel(Model.RIDGE, random_state=0, cv=3, search="grid")
    explicit.fit(synthetic)
    assert default.search == "grid"
    assert _params(default) == _params(explicit)
    assert default.search_space_ is default.param_grid_
    assert default.search_results_ is not None
    assert len(default.search_results_) == len(_DEFAULT_GRIDS[Model.RIDGE]["alpha"])
    assert {"param_alpha", "mean_test_score", "rank_test_score"} <= set(
        default.search_results_.columns
    )


def test_no_search_with_init_hyperparameters(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.PLS, init_hyperparameters={"n_components": 3}, search="bayes"
    ).fit(synthetic)
    assert model.search_results_ is None and model.search_space_ is None
    assert model.best_hyperparameters == {"n_components": 3}


# -- Default spaces -----------------------------------------------------------------


def test_every_tunable_model_has_a_default_space() -> None:
    assert set(_DEFAULT_SPACES) == set(_DEFAULT_GRIDS)
    for model, grid in _DEFAULT_GRIDS.items():
        assert set(_DEFAULT_SPACES[model]) == set(grid), model


@pytest.mark.parametrize("model", sorted(_DEFAULT_GRIDS, key=str))
def test_default_spaces_cover_the_default_grids(model: Model) -> None:
    for name, values in _DEFAULT_GRIDS[model].items():
        entry = _DEFAULT_SPACES[model][name]
        if isinstance(entry, list):
            assert list(values) == entry
        else:
            _, low, high = entry
            assert low <= np.min(values) and np.max(values) <= high, (model, name)


def test_dataset_dependent_ranges(synthetic: Dataset) -> None:
    pls = StandardModel(Model.PLS, search="random", n_iter=5, random_state=0)
    pls.fit(synthetic)
    assert pls.search_space_ is not None
    assert pls.search_space_["n_components"] == ("int", 1, synthetic.n_features - 1)
    svr = StandardModel(Model.SVR, search="random", n_iter=5, random_state=0)
    svr.fit(synthetic)
    assert svr.search_space_ is not None
    kind, low, high = svr.search_space_["gamma"]
    assert kind == "log" and 0 < low < high
    # an explicit range is kept
    custom = StandardModel(
        Model.PLS,
        search="random",
        n_iter=5,
        search_space={"n_components": ("int", 2, 4)},
        random_state=0,
    ).fit(synthetic)
    assert 2 <= _params(custom)["n_components"] <= 4


# -- Random search ------------------------------------------------------------------


def test_random_search_is_reproducible(synthetic: Dataset) -> None:
    def run(seed: int) -> StandardModel:
        return StandardModel(
            Model.SVR, search="random", n_iter=8, random_state=seed, cv=3
        ).fit(synthetic)

    first, again = run(0), run(0)
    assert _params(first) == _params(again)
    assert first.search_results_ is not None and len(first.search_results_) == 8
    np.testing.assert_allclose(first.predict(synthetic.X), again.predict(synthetic.X))


def test_custom_grid_lists_are_choices(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.RIDGE,
        grid_search_hyperparameters={"alpha": [0.1, 1.0, 10.0]},
        search="random",
        n_iter=3,
        random_state=0,
    ).fit(synthetic)
    assert _params(model)["alpha"] in (0.1, 1.0, 10.0)


def test_random_distributions() -> None:
    rng = np.random.default_rng(0)
    dists = random_distributions(
        {
            "f": ("float", 2, 3),
            "l": ("log", 1e-3, 1e3),
            "i": ("int", 1, 4),
            "li": ("logint", 1, 1000),
            "c": ["a", "b"],
            "t": [(50,), (50, 50)],
            "a": np.array([1, 2]),
            "s": uniform(0, 1),
        }
    )
    for _ in range(50):
        assert 2 <= dists["f"].rvs(random_state=rng) <= 3
        assert 1e-3 <= dists["l"].rvs(random_state=rng) <= 1e3
        assert dists["i"].rvs(random_state=rng) in (1, 2, 3, 4)
        assert 1 <= dists["li"].rvs(random_state=rng) <= 1000
    assert dists["c"] == ["a", "b"] and dists["t"] == [(50,), (50, 50)]
    assert dists["a"] == [1, 2]


@pytest.mark.parametrize(
    "entry",
    [
        ("log", 0, 1),
        ("float", 2, 1),
        ("int", 1.5, 3),
        ("float", "a", 1),
        [],
        3.0,
    ],
)
def test_invalid_space_entries(entry: Any) -> None:
    with pytest.raises(SpectraXAIValueError):
        random_distributions({"x": entry})


# -- Bayesian search ----------------------------------------------------------------


@needs_optuna
def test_bayesian_search_is_reproducible(synthetic: Dataset) -> None:
    def run() -> StandardModel:
        return StandardModel(
            Model.SVR, search="bayes", n_iter=12, random_state=0, cv=3
        ).fit(synthetic)

    first, again = run(), run()
    assert _params(first) == _params(again)
    assert first.best_score == again.best_score
    results = first.search_results_
    assert results is not None and len(results) == 12
    assert {"param_C", "param_epsilon", "param_gamma", "state"} <= set(results)
    assert first.best_score == pytest.approx(
        results.loc[results.state == "complete", "mean_test_score"].max()
    )


@needs_optuna
def test_bayesian_search_finds_a_good_ridge(synthetic: Dataset) -> None:
    grid = StandardModel(Model.RIDGE, random_state=0, cv=3).fit(synthetic)
    bayes = StandardModel(
        Model.RIDGE, search="bayes", n_iter=25, random_state=0, cv=3
    ).fit(synthetic)
    assert grid.best_score is not None and bayes.best_score is not None
    assert bayes.best_score >= grid.best_score - 0.01


@needs_optuna
def test_pruning_stops_unpromising_trials(synthetic: Dataset) -> None:
    search = BayesianSearchCV(
        Ridge(),
        {"alpha": ("log", 1e-4, 1e6)},
        n_iter=40,
        cv=5,
        random_state=0,
    ).fit(synthetic.X, synthetic.Y.ravel())
    pruned = search.results_[search.results_.state == "pruned"]
    assert len(pruned) > 0
    assert (pruned.n_folds < 5).all()
    no_pruning = BayesianSearchCV(
        Ridge(), {"alpha": ("log", 1e-4, 1e6)}, n_iter=10, pruning=False
    ).fit(synthetic.X, synthetic.Y.ravel())
    assert (no_pruning.results_.state == "complete").all()


@needs_optuna
def test_bayesian_search_on_a_pipeline_with_any_entry(synthetic: Dataset) -> None:
    import optuna

    pipe = make_pipeline(StandardScaler(), Ridge())
    search = BayesianSearchCV(
        pipe,
        {
            "ridge__alpha": optuna.distributions.FloatDistribution(1e-3, 10, log=True),
            "ridge__fit_intercept": [True, False],
            "standardscaler__with_std": optuna.distributions.CategoricalDistribution(
                [True, False]
            ),
            "ridge__solver": ("auto", "svd", "cholesky", "lsqr"),
        },
        n_iter=10,
        cv=3,
        random_state=1,
    ).fit(synthetic.X, synthetic.Y.ravel())
    assert set(search.best_params_) == {
        "ridge__alpha",
        "ridge__fit_intercept",
        "standardscaler__with_std",
        "ridge__solver",
    }
    assert search.predict(synthetic.X).shape == (synthetic.n_samples,)
    assert search.score(synthetic.X, synthetic.Y.ravel()) > 0.5


@needs_optuna
def test_non_primitive_choices_are_returned_as_given(synthetic: Dataset) -> None:
    model = StandardModel(
        Model.MLP,
        search="bayes",
        n_iter=3,
        search_space={"hidden_layer_sizes": [(5,), (5, 5)], "max_iter": [200]},
        random_state=0,
        cv=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # tiny MLPs, few epochs
        model.fit(synthetic)
    assert model.best_hyperparameters is not None
    assert model.best_hyperparameters["hidden_layer_sizes"] in [(5,), (5, 5)]


@needs_optuna
def test_custom_sampler_is_copied_for_every_fit(synthetic: Dataset) -> None:
    import optuna

    sampler = optuna.samplers.RandomSampler(seed=5)
    model = StandardModel(Model.RIDGE, search="bayes", n_iter=6, sampler=sampler, cv=3)
    first = _params(cast(StandardModel, clone(model)).fit(synthetic))
    second = _params(model.fit(synthetic))
    third = _params(model.fit(synthetic))
    assert first == second == third


@needs_optuna
def test_fit_and_predict_with_bayesian_search(synthetic: Dataset) -> None:
    model = StandardModel(Model.PLS, search="bayes", n_iter=8, random_state=0, cv=3)
    res = model.fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert np.isfinite(res.loc[0, "RMSE"])
    assert 1 <= res.loc[0, "n_components"] < synthetic.n_features


def test_new_parameters_round_trip_through_clone(synthetic: Dataset) -> None:
    space = {"alpha": ("log", 0.1, 10.0)}
    model = StandardModel(
        Model.RIDGE, search="random", n_iter=4, search_space=space, random_state=0
    )
    params = cast(StandardModel, clone(model)).get_params()
    assert params["search"] == "random" and params["n_iter"] == 4
    assert params["search_space"] == space and params["sampler"] is None
    scores = cross_val_score(model, synthetic.X, synthetic.Y.ravel(), cv=3)
    assert np.all(np.isfinite(scores))


# -- Errors -------------------------------------------------------------------------


def test_unknown_search(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="search should be one of"):
        StandardModel(Model.RIDGE, search="hyperband").fit(synthetic)


def test_search_space_needs_random_or_bayes(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="search_space"):
        StandardModel(Model.RIDGE, search_space={"alpha": ("log", 1, 2)}).fit(synthetic)


@pytest.mark.parametrize("n_iter", [0, -1, 2.5, True])
@needs_optuna
def test_invalid_n_iter(synthetic: Dataset, n_iter: Any) -> None:
    with pytest.raises(SpectraXAIValueError, match="n_iter"):
        BayesianSearchCV(Ridge(), {"alpha": ("log", 1, 2)}, n_iter=n_iter).fit(
            synthetic.X, synthetic.Y.ravel()
        )


def test_missing_optuna_explains_how_to_install(
    synthetic: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "optuna" or name.startswith("optuna."):
            raise ImportError("No module named 'optuna'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"spectraxai\[optuna\]"):
        StandardModel(Model.RIDGE, search="bayes", n_iter=2).fit(synthetic)
    # grid and random search do not need Optuna
    StandardModel(Model.RIDGE, search="random", n_iter=2, random_state=0).fit(synthetic)

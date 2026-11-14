# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hyperparameter search beyond grids: random and Bayesian (Optuna) search.

`spectraxai.models.StandardModel` uses this module through its ``search=`` option
(``"grid"``, ``"random"`` or ``"bayes"``); `BayesianSearchCV` can also be used on
its own with any scikit-learn estimator or pipeline.

Search spaces
-------------
A search space maps each hyperparameter to one of:

- ``("float", low, high)``: a real number, sampled uniformly;
- ``("log", low, high)``: a positive real number, sampled on a log scale;
- ``("int", low, high)``: an integer between ``low`` and ``high`` (inclusive);
- ``("logint", low, high)``: an integer, sampled on a log scale;
- a list, tuple or array of choices, e.g. ``["uniform", "distance"]`` or
  ``[(50,), (100,), (50, 50)]`` (only a ``(kind, low, high)`` tuple whose kind is
  one of the above is read as a range);
- an Optuna distribution (Bayesian search) or a SciPy distribution (random
  search), passed through as is.

The same space works for random and Bayesian search, for example::

    {"C": ("log", 0.25, 512), "epsilon": ("log", 0.01, 0.5)}

Bayesian search needs the optional ``optuna`` extra:
``pip install spectraxai[optuna]``.

"""

import copy
import numbers
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import pandas
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted

from spectraxai.exceptions import SpectraXAIValueError

_KINDS = ("float", "log", "int", "logint")


def _is_spec(value: Any) -> bool:
    """Whether ``value`` is a ``(kind, low, high)`` range spec."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and value[0] in _KINDS
    )


def _check_spec(name: str, spec: tuple) -> tuple[str, float, float]:
    kind, low, high = spec
    if not (isinstance(low, numbers.Real) and isinstance(high, numbers.Real)):
        raise SpectraXAIValueError(f"{name}: bounds should be numbers, got {spec!r}")
    low, high = float(low), float(high)
    if not low < high and not (kind in ("int", "logint") and low == high):
        raise SpectraXAIValueError(f"{name}: low should be below high, got {spec!r}")
    if kind in ("log", "logint") and low <= 0:
        raise SpectraXAIValueError(f"{name}: log ranges need low > 0, got {spec!r}")
    if kind in ("int", "logint") and (int(low) != low or int(high) != high):
        raise SpectraXAIValueError(f"{name}: int ranges need integer bounds")
    return kind, low, high


def _choices(name: str, value: Any) -> list[Any] | None:
    """The list of choices of a categorical entry, or None if it is not one."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list) or (isinstance(value, tuple) and not _is_spec(value)):
        choices = list(value)
        if not choices:
            raise SpectraXAIValueError(f"{name}: the list of choices is empty")
        return choices
    return None


class _LogUniformInt:
    """An integer distribution sampled on a log scale (for random search)."""

    def __init__(self, low: int, high: int) -> None:
        self.low, self.high = low, high

    def rvs(self, random_state: Any = None) -> int:
        value = loguniform(self.low, self.high + 1).rvs(random_state=random_state)
        return int(min(self.high, np.floor(value)))


def random_distributions(space: dict[str, Any]) -> dict[str, Any]:
    """Convert a search space to the distributions of `RandomizedSearchCV`."""
    distributions: dict[str, Any] = {}
    for name, value in space.items():
        if _is_spec(value):
            kind, low, high = _check_spec(name, value)
            if kind == "float":
                distributions[name] = uniform(low, high - low)
            elif kind == "log":
                distributions[name] = loguniform(low, high)
            elif kind == "int":
                distributions[name] = randint(int(low), int(high) + 1)
            else:
                distributions[name] = _LogUniformInt(int(low), int(high))
        elif (choices := _choices(name, value)) is not None:
            distributions[name] = choices
        elif hasattr(value, "rvs"):
            distributions[name] = value
        else:
            raise SpectraXAIValueError(
                f"{name}: cannot search {value!r}; use a list of choices or a "
                "('float' | 'log' | 'int' | 'logint', low, high) range"
            )
    return distributions


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Bayesian search needs Optuna: pip install spectraxai[optuna]"
        ) from error
    return optuna


def _primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _suggest(trial: Any, name: str, value: Any) -> Any:
    """Ask an Optuna trial for a value of one search-space entry."""
    optuna = _import_optuna()
    distributions = optuna.distributions
    if _is_spec(value):
        kind, low, high = _check_spec(name, value)
        if kind in ("float", "log"):
            return trial.suggest_float(name, low, high, log=kind == "log")
        return trial.suggest_int(name, int(low), int(high), log=kind == "logint")
    if (choices := _choices(name, value)) is not None:
        if all(_primitive(c) for c in choices):
            return trial.suggest_categorical(name, choices)
        # Optuna stores only primitive choices: search over their positions
        return choices[trial.suggest_categorical(name, list(range(len(choices))))]
    if isinstance(value, distributions.FloatDistribution):
        return trial.suggest_float(
            name, value.low, value.high, log=value.log, step=value.step
        )
    if isinstance(value, distributions.IntDistribution):
        return trial.suggest_int(
            name, value.low, value.high, log=value.log, step=value.step
        )
    if isinstance(value, distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, list(value.choices))
    raise SpectraXAIValueError(
        f"{name}: cannot search {value!r}; use a list of choices, a "
        "('float' | 'log' | 'int' | 'logint', low, high) range or an Optuna "
        "distribution"
    )


class BayesianSearchCV(BaseEstimator):
    """Bayesian hyperparameter search with cross-validation, using Optuna.

    Each trial proposes hyperparameters (the TPE sampler by default), fits a clone
    of ``estimator`` on every training fold and scores it on the validation fold.
    After each fold the running mean score is reported, so that trials clearly
    behind the median of earlier trials are stopped early (``pruning``). The best
    hyperparameters are then refitted on all the data.

    Folds of a trial run one after another (so that pruning can stop them);
    the estimator itself may still use several cores (e.g. XGBoost, RF with
    ``n_jobs``).

    Requires the optional ``optuna`` extra: ``pip install spectraxai[optuna]``.

    Parameters
    ----------
    estimator: scikit-learn estimator
        The estimator (or pipeline) to tune; it is cloned, never modified.

    search_space: `dict`
        Hyperparameter name to search entry (see the module documentation).

    n_iter: `int`, optional
        Number of trials. Defaults to 50.

    cv: int, cross-validation generator or iterable, optional
        The cross-validation splitting strategy, as in scikit-learn. Defaults
        to 5 folds.

    scoring: str or callable, optional
        The score to maximise. Defaults to the estimator's ``score`` (R² for
        regressors).

    sampler: `optuna.samplers.BaseSampler`, optional
        The Optuna sampler, e.g. ``optuna.samplers.GPSampler()``. It is copied
        at every ``fit``, so repeated fits start from the same state. Defaults
        to ``TPESampler(seed=random_state)``.

    pruning: `bool`, optional
        Stop unpromising trials early (median pruning). Defaults to True.

    random_state: `int`, optional
        Seed of the default sampler. Defaults to None.

    Attributes
    ----------
    best_estimator_:
        The estimator refitted on all data with the best hyperparameters.

    best_params_: `dict`
        The best hyperparameters.

    best_score_: `float`
        The mean cross-validated score of the best trial.

    results_: `pandas.DataFrame`
        One row per trial: its hyperparameters (``param_<name>``),
        ``mean_test_score``, ``std_test_score``, ``n_folds`` (fewer if pruned)
        and ``state`` (``"complete"`` or ``"pruned"``).

    study_: `optuna.Study`
        The Optuna study, for Optuna's own analysis and plots.

    """

    def __init__(
        self,
        estimator: Any,
        search_space: dict[str, Any],
        *,
        n_iter: int = 50,
        cv: int | Iterable[tuple[np.ndarray, np.ndarray]] | Any = 5,
        scoring: Any = None,
        sampler: Any = None,
        pruning: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.search_space = search_space
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.sampler = sampler
        self.pruning = pruning
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "BayesianSearchCV":
        """Search the hyperparameters, then refit the best ones on all of X, y."""
        optuna = _import_optuna()
        if (
            isinstance(self.n_iter, bool)
            or not isinstance(self.n_iter, numbers.Integral)
            or self.n_iter < 1
        ):
            raise SpectraXAIValueError("n_iter should be a positive integer")
        if not isinstance(self.search_space, dict):
            raise SpectraXAIValueError("search_space should be a dict")
        X, y = np.asarray(X), np.asarray(y)
        folds = list(cast(Any, check_cv)(self.cv).split(X, y))
        scorer = cast(Any, check_scoring(self.estimator, scoring=self.scoring))
        sampler = (
            optuna.samplers.TPESampler(seed=self.random_state)
            if self.sampler is None
            else copy.deepcopy(self.sampler)
        )
        pruner = (
            optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
            if self.pruning
            else optuna.pruners.NopPruner()
        )
        rows: list[dict[str, Any]] = []
        chosen: dict[int, dict[str, Any]] = {}

        def objective(trial: Any) -> float:
            params = {
                name: _suggest(trial, name, value)
                for name, value in self.search_space.items()
            }
            chosen[trial.number] = params
            scores: list[float] = []
            row = {f"param_{k}": v for k, v in params.items()}
            rows.append(row)
            for step, (train, test) in enumerate(folds):
                model = cast(Any, clone(self.estimator)).set_params(**params)
                model.fit(X[train], y[train])
                scores.append(float(scorer(model, X[test], y[test])))
                row.update(
                    mean_test_score=float(np.mean(scores)),
                    std_test_score=float(np.std(scores)),
                    n_folds=len(scores),
                    state="complete",
                )
                if step < len(folds) - 1:
                    trial.report(float(np.mean(scores)), step)
                    if trial.should_prune():
                        row["state"] = "pruned"
                        raise optuna.TrialPruned()
            return float(np.mean(scores))

        verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner
            )
            study.optimize(objective, n_trials=int(self.n_iter))
        finally:
            optuna.logging.set_verbosity(verbosity)
        complete = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete:
            raise RuntimeError("No trial completed; check the estimator and space")
        best = max(complete, key=lambda t: t.value)
        self.study_ = study
        self.best_params_ = chosen[best.number]
        self.best_score_ = float(best.value)
        self.best_estimator_ = cast(Any, clone(self.estimator)).set_params(
            **self.best_params_
        )
        self.best_estimator_.fit(X, y)
        self.results_ = pandas.DataFrame(rows)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the refitted best estimator."""
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X: Any, y: Any) -> float:
        """Score the refitted best estimator (its own ``score``, R² for regressors)."""
        check_is_fitted(self, "best_estimator_")
        return float(self.best_estimator_.score(X, y))

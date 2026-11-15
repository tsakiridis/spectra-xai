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
from sklearn.base import BaseEstimator, RegressorMixin, clone
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


def _check_n_iter(n_iter: Any) -> None:
    if (
        isinstance(n_iter, bool)
        or not isinstance(n_iter, numbers.Integral)
        or n_iter < 1
    ):
        raise SpectraXAIValueError("n_iter should be a positive integer")


def _cross_validate(
    trial: Any,
    make_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    scorer: Any,
    row: dict[str, Any],
) -> float:
    """Score a trial fold by fold, recording progress in ``row``.

    After each fold but the last the running mean is reported to the trial, which
    may then be pruned (``row["state"]`` becomes ``"pruned"``).

    """
    optuna = _import_optuna()
    scores: list[float] = []
    row.update(mean_test_score=np.nan, std_test_score=np.nan, n_folds=0)
    row["state"] = "failed"  # until a fold succeeds
    for step, (train, test) in enumerate(folds):
        model = make_model()
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


_TRIAL_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)
"""Errors that fail a single trial (e.g. an invalid combination of options)
instead of the whole search."""


def _optimize(
    optuna: Any,
    objective: Any,
    n_iter: int,
    sampler: Any,
    pruning: bool,
    random_state: int | None,
    catch: tuple[type[BaseException], ...] = (),
) -> Any:
    """Run an Optuna study that maximises ``objective`` (quietly)."""
    sampler = (
        optuna.samplers.TPESampler(seed=random_state)
        if sampler is None
        else copy.deepcopy(sampler)
    )
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        if pruning
        else optuna.pruners.NopPruner()
    )
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    try:
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        study.optimize(objective, n_trials=int(n_iter), catch=catch)
    finally:
        optuna.logging.set_verbosity(verbosity)
    return study


def _ranked(results: pandas.DataFrame) -> pandas.DataFrame:
    """Add ``rank_test_score`` (1 = best), ranking complete trials only.

    Pruned and failed trials have partial or no scores, so they get no rank.

    """
    complete = results["state"] == "complete"
    results["rank_test_score"] = (
        results["mean_test_score"]
        .where(complete)
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    return results


def _best_trial(optuna: Any, study: Any) -> Any:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No trial completed; check the estimator and space")
    return max(complete, key=lambda t: t.value)


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
        ``mean_test_score``, ``std_test_score``, ``n_folds`` (fewer if pruned),
        ``state`` (``"complete"`` or ``"pruned"``) and ``rank_test_score``
        (complete trials only; pruned trials have partial scores).

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
        _check_n_iter(self.n_iter)
        if not isinstance(self.search_space, dict):
            raise SpectraXAIValueError("search_space should be a dict")
        X, y = np.asarray(X), np.asarray(y)
        folds = list(cast(Any, check_cv)(self.cv).split(X, y))
        scorer = cast(Any, check_scoring(self.estimator, scoring=self.scoring))
        rows: list[dict[str, Any]] = []
        chosen: dict[int, dict[str, Any]] = {}

        def objective(trial: Any) -> float:
            params = {
                name: _suggest(trial, name, value)
                for name, value in self.search_space.items()
            }
            chosen[trial.number] = params
            row: dict[str, Any] = {f"param_{k}": v for k, v in params.items()}
            rows.append(row)
            return _cross_validate(
                trial,
                lambda: cast(Any, clone(self.estimator)).set_params(**params),
                X,
                y,
                folds,
                scorer,
                row,
            )

        study = _optimize(
            optuna,
            objective,
            self.n_iter,
            self.sampler,
            self.pruning,
            self.random_state,
        )
        best = _best_trial(optuna, study)
        self.study_ = study
        self.best_params_ = chosen[best.number]
        self.best_score_ = float(best.value)
        self.best_estimator_ = cast(Any, clone(self.estimator)).set_params(
            **self.best_params_
        )
        self.best_estimator_.fit(X, y)
        self.results_ = _ranked(pandas.DataFrame(rows))
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


def _is_tunable(value: Any) -> bool:
    """Whether a pre-processing option is a range to search over."""
    return _is_spec(value) or type(value).__module__.startswith("optuna.")


def _resolve_options(method: Any, choose: Any, prefix: str) -> Any:
    """Replace the tunable options of a pre-processing by ``choose(name, spec)``."""
    if isinstance(method, list):
        return [
            _resolve_options(step, choose, f"{prefix}.{i}")
            for i, step in enumerate(method)
        ]
    if isinstance(method, tuple) and len(method) == 2 and isinstance(method[1], dict):
        step, options = method
        return (
            step,
            {
                key: (
                    choose(f"{prefix}:{step.name}.{key}", value)
                    if _is_tunable(value)
                    else value
                )
                for key, value in options.items()
            },
        )
    return method


def _midpoint(name: str, value: Any) -> Any:
    """A representative value of a range (used to size dataset-dependent ranges)."""
    if _is_spec(value):
        kind, low, high = _check_spec(name, value)
        mid = np.sqrt(low * high) if kind in ("log", "logint") else (low + high) / 2
        return int(round(mid)) if kind in ("int", "logint") else float(mid)
    low = getattr(value, "low", None)
    high = getattr(value, "high", None)
    if low is not None and high is not None:
        return type(low)((low + high) / 2)
    return list(getattr(value, "choices", [None]))[0]


def _has_tunable_options(method: Any) -> bool:
    found: list[bool] = []
    _resolve_options(method, lambda name, value: found.append(True), "p")
    return bool(found)


class PreprocessingSearch(RegressorMixin, BaseEstimator):
    """Search the pre-processing, the model and its hyperparameters together.

    Every candidate is a scikit-learn pipeline of a
    `spectraxai.preprocessing.SpectralPreprocessor`, an input scaler (for the
    models `spectraxai.models.StandardModel` scales) and the model. Each candidate
    is cross-validated as a whole, so stateful pre-processing such as MSC is
    learnt on the training folds only.

    Three strategies are available:

    - ``"bayes"`` (default): one Optuna study picks a pre-processing, the values
      of its tunable options (e.g. ``(SP.FD, {"order": ("float", 0, 2)})``), a
      model and its hyperparameters in every trial, and stops unpromising trials
      after each fold. Needs ``pip install spectraxai[optuna]``.
    - ``"grid"``: every pre-processing with every model and every combination of
      the model's grid (`sklearn.model_selection.GridSearchCV`).
    - ``"random"``: ``n_iter`` random candidates from the same combinations, with
      the models' search spaces (`sklearn.model_selection.RandomizedSearchCV`).

    Parameters
    ----------
    models: sequence of `spectraxai.models.Model` or `StandardModel`, optional
        The candidate models. A `StandardModel` brings its own grid, search space
        or fixed ``init_hyperparameters`` (and, with `Model.CUSTOM`, its
        estimator). Defaults to ``[Model.PLS]``.

    preprocessings: sequence of `SpectralPreprocessingSequence`, optional
        The candidate pre-processings, e.g. ``[SP.ABS, [SP.ABS, SP.SNV]]``. With
        Bayesian search, options may be ranges, e.g.
        ``[SP.ABS, (SP.FD, {"order": ("float", 0, 2)})]`` or
        ``(SP.SG1, {"window_length": ("int", 5, 31), "polyorder": 2})``.
        Defaults to ``[SpectralPreprocessing.NONE]``.

    search: `str`, optional
        ``"bayes"`` (default), ``"grid"`` or ``"random"``.

    n_iter: `int`, optional
        Number of trials (Bayesian) or candidates (random). Defaults to 100.

    cv: int, cross-validation generator or iterable, optional
        The cross-validation strategy. Defaults to 5 folds.

    scoring: str or callable, optional
        The score to maximise. Defaults to R².

    sampler: `optuna.samplers.BaseSampler`, optional
        The Optuna sampler of Bayesian search. Defaults to TPE seeded by
        ``random_state``.

    pruning: `bool`, optional
        Stop unpromising Bayesian trials early. Defaults to True.

    random_state: `int`, optional
        Seed of the search and of the stochastic models. Defaults to None.

    n_jobs: `int`, optional
        Parallel jobs of grid and random search. Defaults to -1 (all processors).

    Attributes
    ----------
    best_estimator_: `sklearn.pipeline.Pipeline`
        The best pipeline (``pre`` → ``scale`` → ``model``), refitted on all data.

    best_params_: `dict`
        ``{"preprocessing": ..., "model": ..., "hyperparameters": {...}}``.

    best_score_: `float`
        The mean cross-validated score of the best candidate.

    results_: `pandas.DataFrame`
        One row per candidate: ``preprocessing`` (as a string, see
        `spectraxai.spectra.SpectralPreprocessing.init_class`),
        ``preprocessing_index`` (its position in ``preprocessings``, useful to
        group candidates whose options were tuned), ``model``,
        ``hyperparameters``, ``mean_test_score``, ``std_test_score`` and
        ``rank_test_score``; Bayesian search adds ``n_folds`` and ``state``
        (``"complete"``, ``"pruned"`` or ``"failed"``).

    Notes
    -----
    The ranges that depend on the data (PLS ``n_components``, SVR ``gamma``) are
    derived from the pre-processed training inputs, as in `StandardModel`: for
    each pre-processing in grid and random search, and once over all
    pre-processings (tunable options at their midpoint) in Bayesian search.

    """

    def __init__(
        self,
        models: Any = None,
        preprocessings: Any = None,
        *,
        search: str = "bayes",
        n_iter: int = 100,
        cv: Any = 5,
        scoring: Any = None,
        sampler: Any = None,
        pruning: bool = True,
        random_state: int | None = None,
        n_jobs: int | None = -1,
    ) -> None:
        self.models = models
        self.preprocessings = preprocessings
        self.search = search
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.sampler = sampler
        self.pruning = pruning
        self.random_state = random_state
        self.n_jobs = n_jobs

    # -- set-up ----------------------------------------------------------------

    def _candidate_models(self) -> list[tuple[str, Any]]:
        """``(label, StandardModel)`` for every candidate model."""
        from spectraxai.models import Model, StandardModel

        models = [Model.PLS] if self.models is None else list(self.models)
        if not models:
            raise SpectraXAIValueError("models cannot be empty")
        out: list[tuple[str, Any]] = []
        for model in models:
            if isinstance(model, StandardModel):
                sm = model
            elif isinstance(model, Model):
                sm = StandardModel(model, random_state=self.random_state)
            else:
                raise SpectraXAIValueError(
                    f"models should contain Model members or StandardModel "
                    f"instances, got {model!r}"
                )
            label = sm.model.name
            if any(existing == label for existing, _ in out):
                label = f"{label}_{sum(e.startswith(label) for e, _ in out) + 1}"
            out.append((label, sm))
        return out

    def _candidate_preprocessings(self) -> list[Any]:
        from spectraxai.spectra import SpectralPreprocessing

        preps = (
            [SpectralPreprocessing.NONE]
            if self.preprocessings is None
            else list(self.preprocessings)
        )
        if not preps:
            raise SpectraXAIValueError("preprocessings cannot be empty")
        return preps

    @staticmethod
    def _inputs(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray, Any]:
        from spectraxai.dataset import Dataset

        if isinstance(X, Dataset):
            if y is not None:
                raise SpectraXAIValueError("y should not be given with a Dataset")
            Y = X.Y
            return (
                np.asarray(X.X),
                np.squeeze(Y, axis=1) if Y.shape[1] == 1 else Y,
                X.wavelengths,
            )
        if y is None:
            raise SpectraXAIValueError("y is required when X is not a Dataset")
        return np.asarray(X), np.asarray(y), None

    def _model_inputs(
        self, sm: Any, X: np.ndarray, method: Any, wavelengths: Any
    ) -> np.ndarray:
        """The pre-processed (and, if needed, scaled) inputs seen by a model."""
        from sklearn.preprocessing import StandardScaler

        from spectraxai.models import _SCALED_MODELS
        from spectraxai.preprocessing import SpectralPreprocessor

        Xp = SpectralPreprocessor(method, wavelengths=wavelengths).fit_transform(X)
        if sm.model in _SCALED_MODELS:
            Xp = StandardScaler().fit_transform(Xp)
        return np.asarray(Xp)

    @staticmethod
    def _scale_step(sm: Any) -> Any:
        from sklearn.preprocessing import StandardScaler

        from spectraxai.models import _SCALED_MODELS

        return StandardScaler() if sm.model in _SCALED_MODELS else "passthrough"

    @staticmethod
    def _model_space(sm: Any, Xp: np.ndarray, grid: bool) -> dict[str, Any]:
        """The model's own grid or search space (fixed hyperparameters as choices)."""
        if sm.init_hyperparameters:
            return {k: [v] for k, v in sm.init_hyperparameters.items()}
        return sm._grid(Xp) if grid else sm._search_space(Xp)

    def _pipeline(self, wavelengths: Any, model: Any) -> Any:
        """The candidate template; ``model`` is replaced by every candidate."""
        from sklearn.pipeline import Pipeline

        from spectraxai.preprocessing import SpectralPreprocessor

        return Pipeline(
            [
                ("pre", SpectralPreprocessor(wavelengths=wavelengths)),
                ("scale", "passthrough"),
                ("model", model),
            ]
        )

    # -- fitting ---------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> "PreprocessingSearch":
        """Search on a `spectraxai.dataset.Dataset` (or on ``X, y``), then refit."""
        if self.search not in ("grid", "random", "bayes"):
            raise SpectraXAIValueError(
                f"search should be 'grid', 'random' or 'bayes', got {self.search!r}"
            )
        _check_n_iter(self.n_iter)
        X, y, wavelengths = self._inputs(X, y)
        models = self._candidate_models()
        preps = self._candidate_preprocessings()
        if self.search == "bayes":
            self._fit_bayes(X, y, wavelengths, models, preps)
        else:
            self._fit_sklearn(X, y, wavelengths, models, preps)
        self.n_features_in_ = X.shape[1]
        return self

    def _fit_sklearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: Any,
        models: list[tuple[str, Any]],
        preps: list[Any],
    ) -> None:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        grid = self.search == "grid"
        labels: dict[int, str] = {}
        candidates: list[dict[str, Any]] = []
        for method in preps:
            if _has_tunable_options(method):
                raise SpectraXAIValueError(
                    "Ranges in pre-processing options need search='bayes'; list "
                    "the values to try as separate pre-processings instead"
                )
            for label, sm in models:
                estimator = sm._new_estimator(grid_search=True)
                labels[id(estimator)] = label
                space = self._model_space(
                    sm, self._model_inputs(sm, X, method, wavelengths), grid
                )
                values = space if grid else random_distributions(space)
                candidates.append(
                    {
                        "pre__method": [method],
                        "scale": [self._scale_step(sm)],
                        "model": [estimator],
                        **{f"model__{k}": v for k, v in values.items()},
                    }
                )
        pipeline = self._pipeline(wavelengths, models[0][1]._new_estimator(True))
        if grid:
            searcher: Any = GridSearchCV(
                pipeline,
                candidates,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
        else:
            searcher = RandomizedSearchCV(
                pipeline,
                candidates,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        searcher.fit(X, y)
        cv_results = searcher.cv_results_
        rows = []
        for i, params in enumerate(cv_results["params"]):
            rows.append(
                {
                    "preprocessing": str(params["pre__method"]),
                    "preprocessing_index": next(
                        k for k, p in enumerate(preps) if p is params["pre__method"]
                    ),
                    "model": labels[id(params["model"])],
                    "hyperparameters": {
                        k.removeprefix("model__"): v
                        for k, v in params.items()
                        if k.startswith("model__")
                    },
                    "mean_test_score": float(cv_results["mean_test_score"][i]),
                    "std_test_score": float(cv_results["std_test_score"][i]),
                    "rank_test_score": int(cv_results["rank_test_score"][i]),
                }
            )
        best = cv_results["params"][searcher.best_index_]
        self.results_ = pandas.DataFrame(rows)
        self.best_params_ = {
            "preprocessing": best["pre__method"],
            "model": labels[id(best["model"])],
            "hyperparameters": rows[searcher.best_index_]["hyperparameters"],
        }
        self.best_score_ = float(searcher.best_score_)
        self.best_estimator_ = searcher.best_estimator_

    def _fit_bayes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelengths: Any,
        models: list[tuple[str, Any]],
        preps: list[Any],
    ) -> None:
        optuna = _import_optuna()
        folds = list(cast(Any, check_cv)(self.cv).split(X, y))
        pipeline = self._pipeline(wavelengths, models[0][1]._new_estimator(True))
        scorer = cast(Any, check_scoring(pipeline, scoring=self.scoring))

        # One space per model, sized on every pre-processing (at mid options)
        spaces: dict[str, dict[str, Any]] = {}
        for label, sm in models:
            merged: dict[str, Any] = {}
            for method in preps:
                mid = _resolve_options(method, _midpoint, "p")
                space = self._model_space(
                    sm, self._model_inputs(sm, X, mid, wavelengths), grid=False
                )
                for name, value in space.items():
                    merged[name] = _widen(merged.get(name), value)
            spaces[label] = merged

        rows: list[dict[str, Any]] = []
        chosen: dict[int, tuple[Any, int, dict[str, Any]]] = {}

        def build(method: Any, index: int, params: dict[str, Any]) -> Any:
            label, sm = models[index]
            estimator = cast(Any, sm._new_estimator(grid_search=True)).set_params(
                **params
            )
            return cast(Any, clone(pipeline)).set_params(
                pre__method=method, scale=self._scale_step(sm), model=estimator
            )

        def objective(trial: Any) -> float:
            p = trial.suggest_categorical("preprocessing", list(range(len(preps))))
            method = _resolve_options(
                preps[p],
                lambda name, value: _suggest(trial, name, value),
                f"p{p}",
            )
            m = (
                trial.suggest_categorical("model", list(range(len(models))))
                if len(models) > 1
                else 0
            )
            label = models[m][0]
            params = {
                name: _suggest(trial, f"{label}.{name}", value)
                for name, value in spaces[label].items()
            }
            chosen[trial.number] = (method, m, params)
            row: dict[str, Any] = {
                "preprocessing": str(method),
                "preprocessing_index": p,
                "model": label,
                "hyperparameters": params,
            }
            rows.append(row)
            return _cross_validate(
                trial, lambda: build(method, m, params), X, y, folds, scorer, row
            )

        study = _optimize(
            optuna,
            objective,
            self.n_iter,
            self.sampler,
            self.pruning,
            self.random_state,
            catch=_TRIAL_ERRORS,
        )
        best = _best_trial(optuna, study)
        method, m, params = chosen[best.number]
        self.study_ = study
        self.results_ = _ranked(pandas.DataFrame(rows))
        self.best_params_ = {
            "preprocessing": method,
            "model": models[m][0],
            "hyperparameters": params,
        }
        self.best_score_ = float(best.value)
        self.best_estimator_ = build(method, m, params).fit(X, y)

    # -- prediction ------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the best pipeline (accepts a `Dataset` or an array)."""
        from spectraxai.dataset import Dataset

        check_is_fitted(self, "best_estimator_")
        if isinstance(X, Dataset):
            X = X.X
        return self.best_estimator_.predict(np.asarray(X))


def _widen(current: Any, new: Any) -> Any:
    """Merge two search entries of the same hyperparameter (union of ranges)."""
    if current is None:
        return new
    if _is_spec(current) and _is_spec(new) and current[0] == new[0]:
        return (current[0], min(current[1], new[1]), max(current[2], new[2]))
    return current

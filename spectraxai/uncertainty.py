# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.uncertainty` gives predictions with intervals.

`ConformalRegressor` wraps any model (a `spectraxai.models.StandardModel` or a
scikit-learn regressor) and returns, with each prediction, an interval that
contains the true value with probability at least 1 − ``alpha`` (e.g. 90%),
whatever the model and the distribution of the data, provided new samples are
exchangeable with the training samples (drawn from the same population; see
`spectraxai.diagnostics.ApplicabilityDomain` to check spectra). The method is
chosen with the `ConformalMethod` enum:

* `ConformalMethod.SPLIT` (split conformal prediction): the model is trained on
  part of the data and its absolute errors on the rest (the calibration set) set
  the half-width of the intervals. One model fit; the intervals have the same
  width for every sample.
* `ConformalMethod.CV_PLUS` (CV+, Barber et al., 2021): the model is trained once
  per cross-validation fold and every sample serves for calibration, using the
  model that did not see it. ``cv`` model fits; narrower and more stable
  intervals on small datasets, whose width varies between samples.

For each output, coverage is guaranteed on average over samples (marginal
coverage), at least 1 − alpha for split conformal and 1 − 2 alpha for CV+
(in practice close to 1 − alpha).

References
----------
Barber, R. F., Candès, E. J., Ramdas, A. and Tibshirani, R. J. (2021). Predictive
inference with the jackknife+. *The Annals of Statistics*, 49(1), 486–507.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018).
Distribution-free predictive inference for regression. *Journal of the American
Statistical Association*, 113(523), 1094–1111.

"""

import math
import numbers
import warnings
from enum import Enum
from typing import Any, Self, cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError

_CHUNK = 2**22
"""Largest number of calibration predictions held in memory at once."""


class ConformalMethod(str, Enum):
    """How `ConformalRegressor` calibrates its intervals (see the module docs)."""

    SPLIT = "Split conformal prediction"
    CV_PLUS = "CV+ (cross-validation conformal prediction)"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"ConformalMethod.{self.name}"


def interval_coverage(y_true: Any, lower: Any, upper: Any) -> np.ndarray:
    """Fraction of true values inside their intervals, per output.

    Parameters
    ----------
    y_true, lower, upper: array-like of shape (n_samples,) or (n_samples, n_outputs)
        The true values and the interval bounds

    Returns
    -------
    `np.ndarray`
        The empirical coverage of each output, of shape (n_outputs,).

    """
    y = np.asarray(y_true, dtype=float).reshape(len(y_true), -1)
    lo = np.asarray(lower, dtype=float).reshape(y.shape)
    hi = np.asarray(upper, dtype=float).reshape(y.shape)
    return np.mean((y >= lo) & (y <= hi), axis=0)


def _rank(level: float, n: int) -> int:
    """The 1-based rank of the conformal quantile of n scores at ``level``."""
    return math.ceil(level * (n + 1))


def _as_xy(X: Any, y: Any) -> tuple[Any, np.ndarray]:
    if isinstance(X, Dataset):
        if y is not None:
            raise SpectraXAIValueError("y should not be given with a Dataset")
        return X.X, X.Y.ravel() if X.n_outputs == 1 else X.Y
    if y is None:
        raise SpectraXAIValueError("y is required when X is not a Dataset")
    return np.asarray(X), np.asarray(y, dtype=float)


def _predict(model: Any, X: Any) -> np.ndarray:
    """Predictions as an (n_samples, n_outputs) array."""
    pred = np.asarray(model.predict(X), dtype=float)
    return pred.reshape(len(pred), -1)


class ConformalRegressor(RegressorMixin, BaseEstimator):
    """Prediction intervals with guaranteed coverage around any regressor.

    Parameters
    ----------
    model: estimator
        A `spectraxai.models.StandardModel` or scikit-learn regressor; it is
        cloned, never modified. A `StandardModel` without
        ``init_hyperparameters`` tunes them in every fit.
    method: `ConformalMethod` or None
        Defaults to None, i.e. `ConformalMethod.SPLIT`.
    alpha: `float`
        Miscoverage rate, in (0, 1): intervals contain the true value with
        probability 1 − alpha. Defaults to 0.1 (90% intervals).
    calibration_size: `float`
        For `ConformalMethod.SPLIT`, the fraction of the samples kept for
        calibration. Defaults to 0.2.
    cv: `int` or cross-validation splitter
        For `ConformalMethod.CV_PLUS`, the number of folds (shuffled with
        ``random_state``) or a splitter whose test folds partition the samples.
        Defaults to 5.
    random_state: `int` or None
        Seeds the calibration split or the folds. Defaults to None.

    Attributes
    ----------
    estimator_: estimator
        The model used for the point predictions: trained on the training part
        (split) or on all samples (CV+).
    quantile_: `np.ndarray`
        For split conformal, the half-width of the intervals per output.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from spectraxai.uncertainty import ConformalRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 5))
    >>> y = X @ np.ones(5) + rng.normal(size=200)
    >>> conformal = ConformalRegressor(Ridge(), random_state=0).fit(X, y)
    >>> pred, lower, upper = conformal.predict(X[:3], return_interval=True)
    >>> bool(np.all(lower < upper))
    True

    """

    def __init__(
        self,
        model: Any,
        *,
        method: ConformalMethod | None = None,
        alpha: float = 0.1,
        calibration_size: float = 0.2,
        cv: Any = 5,
        random_state: int | None = None,
    ) -> None:
        self.model = model
        self.method = method
        self.alpha = alpha
        self.calibration_size = calibration_size
        self.cv = cv
        self.random_state = random_state

    def _clone(self) -> Any:
        return clone(cast(Any, self.model))

    @property
    def _method(self) -> ConformalMethod:
        return ConformalMethod.SPLIT if self.method is None else self.method

    def _check_params(self) -> None:
        if self.method is not None and not isinstance(self.method, ConformalMethod):
            raise SpectraXAIValueError(
                "method should be a ConformalMethod member, e.g. "
                f"ConformalMethod.CV_PLUS, got {self.method!r}"
            )
        for name in ("alpha", "calibration_size"):
            value = getattr(self, name)
            if not (isinstance(value, numbers.Real) and 0 < float(value) < 1):
                raise SpectraXAIValueError(f"{name} should be in (0, 1), got {value!r}")

    def fit(self, X: Any, y: Any = None) -> Self:
        """Train the model and calibrate the intervals.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`
            The training inputs
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The outputs (omitted when ``X`` is a `Dataset`)

        Returns
        -------
        The fitted regressor (``self``).

        """
        self._check_params()
        X, y = _as_xy(X, y)
        if len(X) != len(y):
            raise SpectraXAIValueError("X and y should have the same number of rows")
        self._y_1d = y.ndim == 1
        self.n_features_in_ = X.shape[1]
        Y = y.reshape(len(y), -1)
        if self._method == ConformalMethod.SPLIT:
            self._fit_split(X, y, Y)
        else:
            self._fit_cv_plus(X, y, Y)
        return self

    def _fit_split(self, X: Any, y: np.ndarray, Y: np.ndarray) -> None:
        n = len(X)
        n_cal = min(n - 1, max(1, round(self.calibration_size * n)))
        order = np.random.default_rng(self.random_state).permutation(n)
        cal, train = order[:n_cal], order[n_cal:]
        self.estimator_ = self._clone().fit(X[train], y[train])
        residuals = np.abs(Y[cal] - _predict(self.estimator_, X[cal]))
        rank = _rank(1 - self.alpha, n_cal)
        if rank > n_cal:
            warnings.warn(
                f"{n_cal} calibration samples are too few for alpha={self.alpha}; "
                "the intervals are infinite",
                stacklevel=3,
            )
            self.quantile_ = np.full(Y.shape[1], np.inf)
        else:
            self.quantile_ = np.sort(residuals, axis=0)[rank - 1]
        self.calibration_indices_ = np.sort(cal)

    def _fit_cv_plus(self, X: Any, y: np.ndarray, Y: np.ndarray) -> None:
        cv = self.cv
        if isinstance(cv, numbers.Integral):
            cv = KFold(int(cv), shuffle=True, random_state=self.random_state)
        n = len(X)
        self.fold_of_sample_ = np.full(n, -1)
        self.residuals_ = np.empty_like(Y)
        self.fold_estimators_ = []
        for fold, (train, test) in enumerate(cast(Any, cv).split(X, y)):
            if np.any(self.fold_of_sample_[test] >= 0):
                raise SpectraXAIValueError("CV+ needs test folds that do not overlap")
            model = self._clone().fit(X[train], y[train])
            self.fold_estimators_.append(model)
            self.fold_of_sample_[test] = fold
            self.residuals_[test] = np.abs(Y[test] - _predict(model, X[test]))
        if np.any(self.fold_of_sample_ < 0):
            raise SpectraXAIValueError("CV+ needs test folds that cover every sample")
        self.estimator_ = self._clone().fit(X, y)

    def _intervals(self, X: Any, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._method == ConformalMethod.SPLIT:
            return pred - self.quantile_, pred + self.quantile_
        n = len(self.residuals_)
        k_lo = math.floor(self.alpha * (n + 1))
        k_hi = _rank(1 - self.alpha, n)
        lower = np.full_like(pred, -np.inf)
        upper = np.full_like(pred, np.inf)
        fold_preds = np.stack([_predict(m, X) for m in self.fold_estimators_])
        step = max(1, _CHUNK // (n * pred.shape[1]))
        for start in range(0, len(pred), step):
            rows = slice(start, start + step)
            # prediction of each test sample by the model that left out sample i
            mu = fold_preds[self.fold_of_sample_, rows]  # (n, rows, outputs)
            R = self.residuals_[:, None, :]
            if k_lo >= 1:
                lower[rows] = np.partition(mu - R, k_lo - 1, axis=0)[k_lo - 1]
            if k_hi <= n:
                upper[rows] = np.partition(mu + R, k_hi - 1, axis=0)[k_hi - 1]
        if k_lo < 1 or k_hi > n:
            warnings.warn(
                f"{n} samples are too few for alpha={self.alpha}; the intervals are "
                "infinite",
                stacklevel=3,
            )
        return lower, upper

    def predict(self, X: Any, return_interval: bool = False) -> Any:
        """Predict, optionally with the prediction intervals.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`
            The inputs
        return_interval: `bool`
            Also return the lower and upper bounds. Defaults to False.

        Returns
        -------
        `np.ndarray` or `tuple`
            The predictions, of shape (n_samples,) or (n_samples, n_outputs) as
            the training outputs; with ``return_interval=True`` the tuple
            ``(predictions, lower, upper)`` of three such arrays.

        """
        check_is_fitted(self, "estimator_")
        X = X.X if isinstance(X, Dataset) else np.asarray(X)
        pred = _predict(self.estimator_, X)
        if not return_interval:
            return pred.ravel() if self._y_1d else pred
        lower, upper = self._intervals(X, pred)
        if self._y_1d:
            return pred.ravel(), lower.ravel(), upper.ravel()
        return pred, lower, upper

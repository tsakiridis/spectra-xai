# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.diagnostics` checks whether spectra suit a model.

A model is only reliable for spectra like those it was trained on.
`ApplicabilityDomain` describes the training spectra by a principal component
analysis (PCA) and measures two things for any new spectrum:

* **Hotelling's T²**: how far its scores lie from the centre of the training
  scores, relative to their spread. A large T² means an extreme spectrum of a
  kind the model knows (e.g. a soil richer in some constituent than any in the
  training set).
* **Q residual** (squared prediction error): how much of the spectrum the
  principal components do not describe. A large Q means a spectrum unlike the
  training spectra (e.g. a new material, an instrument fault, or contamination).

Each statistic has a critical limit at the chosen ``confidence``; a spectrum
whose statistic exceeds it is outside the domain. Which statistics count is set
with the `DomainStatistic` enum.

**Critical limits.** T² uses the F-distribution limit for a new observation,
A (n − 1)(n + 1) / (n (n − A)) F(A, n − A), with A components and n training
spectra. Q is a weighted sum of squared residual scores, whose distribution is
approximated from the eigenvalues of the components left out (`QLimit`): by a
scaled chi-squared distribution (Box, 1954; the default, valid for any
eigenvalues) or by the normal approximation of Jackson and Mudholkar (1979),
which breaks down when the left-out eigenvalues are very unequal, as for many
mid-infrared spectra. Both limits assume normally distributed scores; real
spectra often exceed them somewhat more often than 1 − ``confidence``, so check
the training spectra (``training_statistics_``) or raise ``confidence``.

References
----------
Box, G. E. P. (1954). Some theorems on quadratic forms applied in the study of
analysis of variance problems, I. *The Annals of Mathematical Statistics*, 25(2),
290–302.

Jackson, J. E. and Mudholkar, G. S. (1979). Control procedures for residuals
associated with principal component analysis. *Technometrics*, 21(3), 341–349.

"""

from enum import Enum
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
from scipy import stats
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.plotting import _styled_unless


class DomainStatistic(str, Enum):
    """Which statistics decide whether a spectrum is inside the domain."""

    T2 = "Hotelling's T²"
    """Extreme scores within the space of the training spectra."""
    Q = "Q residual"
    """Spectra the principal components of the training spectra do not describe."""
    T2_OR_Q = "Hotelling's T² or Q residual"
    """Outside if either statistic exceeds its limit (the usual choice)."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"DomainStatistic.{self.name}"


def _spectra(X: Any) -> Any:
    return X.X if isinstance(X, Dataset) else X


class QLimit(str, Enum):
    """How the critical limit of the Q residual is approximated."""

    BOX = "Scaled chi-squared (Box, 1954)"
    """g χ²(h) with g = θ₂/θ₁ and h = θ₁²/θ₂, matching the mean and variance of Q.
    Valid for any eigenvalues."""
    JACKSON_MUDHOLKAR = "Jackson and Mudholkar (1979)"
    """A normal approximation of a power of Q; undefined when the left-out
    eigenvalues are very unequal (h₀ ≤ 0)."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"QLimit.{self.name}"


def q_limit(
    eigenvalues: np.ndarray, confidence: float, method: QLimit = QLimit.BOX
) -> float:
    """Critical limit of the Q residual.

    Parameters
    ----------
    eigenvalues: `np.ndarray`
        The covariance eigenvalues of the components left out of the model,
        θₖ being the sum of their k-th powers
    confidence: `float`
        The confidence level, e.g. 0.99
    method: `QLimit`
        The approximation. Defaults to `QLimit.BOX`.

    Returns
    -------
    `float`
        The critical value of Q (0 if no variance is left out).

    """
    lam = np.asarray(eigenvalues, dtype=float)
    theta1, theta2, theta3 = (np.sum(lam**i) for i in (1, 2, 3))
    if theta1 <= 0 or theta2 <= 0:
        return 0.0
    if method == QLimit.BOX:
        g, h = theta2 / theta1, theta1**2 / theta2
        return float(g * stats.chi2.ppf(confidence, h))
    h0 = 1 - 2 * theta1 * theta3 / (3 * theta2**2)
    if h0 <= 0:
        raise SpectraXAIValueError(
            "The Jackson-Mudholkar Q limit is undefined for these eigenvalues "
            f"(h0 = {h0:.3g} <= 0); use QLimit.BOX"
        )
    z = stats.norm.ppf(confidence)
    base = (
        z * np.sqrt(2 * theta2 * h0**2) / theta1
        + 1
        + theta2 * h0 * (h0 - 1) / theta1**2
    )
    return float(theta1 * base ** (1 / h0))


def t2_limit(n_components: int, n_samples: int, confidence: float) -> float:
    """F-distribution limit of Hotelling's T² for a new observation.

    Parameters
    ----------
    n_components: `int`
        The number of principal components, A
    n_samples: `int`
        The number of training spectra, n (> A)
    confidence: `float`
        The confidence level, e.g. 0.99

    Returns
    -------
    `float`
        A (n − 1)(n + 1) / (n (n − A)) F(A, n − A) at ``confidence``.

    """
    a, n = n_components, n_samples
    f = stats.f.ppf(confidence, a, n - a)
    return float(a * (n - 1) * (n + 1) / (n * (n - a)) * f)


class ApplicabilityDomain(OutlierMixin, BaseEstimator):
    """Flag spectra outside the domain of the training spectra (T² and Q).

    Follows scikit-learn's outlier detectors: `predict` returns 1 inside the
    domain and -1 outside, `decision_function` is negative outside, and
    `score_samples` is higher for more typical spectra. `statistics` returns T²
    and Q themselves.

    Parameters
    ----------
    n_components: `int`, `float` or None
        The number of principal components, or the fraction of variance they
        should explain. Defaults to None (99% of the variance).
    confidence: `float`
        Confidence level of the critical limits, in (0, 1). Defaults to 0.99:
        about 1% of spectra like the training ones exceed each limit.
    statistic: `DomainStatistic` or None
        Which statistics count. Defaults to None, i.e. `DomainStatistic.T2_OR_Q`.
    q_limit_method: `QLimit` or None
        How the Q limit is approximated. Defaults to None, i.e. `QLimit.BOX`.

    Attributes
    ----------
    t2_limit_, q_limit_: `float`
        The critical limits.
    n_components_: `int`
        The number of principal components kept.
    training_statistics_: `np.ndarray`
        T² and Q of the training spectra (one row per spectrum).
    offset_: `float`
        -1: `decision_function` is `score_samples` minus ``offset_``.

    Examples
    --------
    >>> import numpy as np
    >>> from spectraxai.diagnostics import ApplicabilityDomain
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 3)) @ rng.normal(size=(3, 40))
    >>> X += rng.normal(scale=0.01, size=X.shape)
    >>> domain = ApplicabilityDomain(n_components=3).fit(X)
    >>> domain.predict(np.vstack([X[:2], X[:1] + 1])).tolist()
    [1, 1, -1]

    """

    def __init__(
        self,
        n_components: int | float | None = None,
        *,
        confidence: float = 0.99,
        statistic: DomainStatistic | None = None,
        q_limit_method: QLimit | None = None,
    ) -> None:
        self.n_components = n_components
        self.confidence = confidence
        self.statistic = statistic
        self.q_limit_method = q_limit_method

    def _check_params(self, n_samples: int, n_features: int) -> int | float:
        if not (isinstance(self.confidence, float) and 0 < self.confidence < 1):
            raise SpectraXAIValueError(
                f"confidence should be a float in (0, 1), got {self.confidence!r}"
            )
        if self.statistic is not None and not isinstance(
            self.statistic, DomainStatistic
        ):
            raise SpectraXAIValueError(
                "statistic should be a DomainStatistic member, e.g. "
                f"DomainStatistic.T2_OR_Q, got {self.statistic!r}"
            )
        if self.q_limit_method is not None and not isinstance(
            self.q_limit_method, QLimit
        ):
            raise SpectraXAIValueError(
                "q_limit_method should be a QLimit member, e.g. QLimit.BOX, got "
                f"{self.q_limit_method!r}"
            )
        n = 0.99 if self.n_components is None else self.n_components
        is_int = isinstance(n, (int, np.integer)) and not isinstance(n, bool)
        if is_int:
            if not 1 <= n < n_samples:
                raise SpectraXAIValueError(
                    f"n_components should be between 1 and n_samples - 1 "
                    f"({n_samples - 1}), got {n}"
                )
            return int(min(n, n_features))
        if not (isinstance(n, float) and 0 < n < 1):
            raise SpectraXAIValueError(
                "n_components should be None, a positive integer or a fraction of "
                f"variance in (0, 1), got {n!r}"
            )
        return n

    def fit(self, X: Any, y: Any = None) -> Self:
        """Learn the principal components and limits of the training spectra.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`
            The training spectra
        y: ignored

        Returns
        -------
        The fitted domain (``self``).

        """
        X = np.asarray(
            validate_data(self, _spectra(X), ensure_min_samples=3), dtype=float
        )
        n_components = self._check_params(*X.shape)
        self.mean_ = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        eigenvalues = s**2 / (len(X) - 1)
        if isinstance(n_components, float):
            explained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_components = int(np.searchsorted(explained, n_components) + 1)
        n_components = max(1, min(n_components, len(X) - 1, int(np.sum(s > 0))))
        self.n_components_ = n_components
        self.components_ = Vt[:n_components]
        self.eigenvalues_ = eigenvalues
        self.t2_limit_ = t2_limit(n_components, len(X), self.confidence)
        self.q_limit_ = q_limit(
            eigenvalues[n_components:],
            self.confidence,
            self.q_limit_method or QLimit.BOX,
        )
        self.offset_ = -1.0
        self.training_statistics_ = self.statistics(X)
        return self

    def statistics(self, X: Any) -> np.ndarray:
        """Hotelling's T² and the Q residual of each spectrum.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`
            The spectra

        Returns
        -------
        `np.ndarray`
            An array of shape (n_samples, 2): T² in the first column, Q in the
            second.

        """
        check_is_fitted(self, "components_")
        X = np.asarray(validate_data(self, _spectra(X), reset=False), dtype=float)
        centred = X - self.mean_
        scores = centred @ self.components_.T
        t2 = np.sum(scores**2 / self.eigenvalues_[: self.n_components_], axis=1)
        q = np.sum((centred - scores @ self.components_) ** 2, axis=1)
        return np.column_stack([t2, q])

    def score_samples(self, X: Any) -> np.ndarray:
        """How typical each spectrum is: minus its largest statistic-to-limit ratio.

        The ratios are those of the statistics that count (see ``statistic``);
        spectra scoring below -1 are outside the domain.

        """
        t2, q = self.statistics(X).T
        statistic = self.statistic or DomainStatistic.T2_OR_Q
        ratios = []
        if statistic in (DomainStatistic.T2, DomainStatistic.T2_OR_Q):
            ratios.append(t2 / self.t2_limit_)
        if statistic in (DomainStatistic.Q, DomainStatistic.T2_OR_Q):
            limit = self.q_limit_ if self.q_limit_ > 0 else np.finfo(float).tiny
            ratios.append(q / limit)
        return -np.max(np.asarray(ratios), axis=0)

    def decision_function(self, X: Any) -> np.ndarray:
        """How far inside the domain each spectrum is: negative outside."""
        return self.score_samples(X) - self.offset_

    def predict(self, X: Any) -> np.ndarray:
        """1 for spectra inside the domain, -1 for spectra outside.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`
            The spectra

        Returns
        -------
        `np.ndarray`
            An integer array of shape (n_samples,).

        """
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def inside(self, X: Any) -> np.ndarray:
        """Boolean mask of the spectra inside the domain."""
        return self.predict(X) == 1

    @_styled_unless("ax")
    def plot(
        self,
        X: Any = None,
        ax: Axes | None = None,
        *,
        log: bool = True,
        **kwargs: Any,
    ) -> Axes:
        """Plot T² against Q with the critical limits (the influence plot).

        The training spectra are drawn in grey; the spectra of ``X``, if given,
        are drawn in colour, those outside the domain marked with crosses.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features) or `Dataset`, optional
            New spectra to place in the plot
        ax: `matplotlib.axes.Axes`, optional
            The axes to draw into. Defaults to None, which creates a figure.
        log: `bool`
            Logarithmic axes, which spread the many small values. Defaults to True.
        **kwargs
            Passed on to `matplotlib.axes.Axes.scatter` for the spectra of ``X``.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes with the plot.

        """
        check_is_fitted(self, "components_")
        if ax is None:
            _, ax = plt.subplots()
        train = self.training_statistics_
        ax.scatter(*train.T, s=12, color="0.7", label="training spectra")
        if X is not None:
            scores = self.statistics(X)
            inside = self.predict(X) == 1
            points: dict[str, Any] = {"s": 20, "color": "C0"}
            ax.scatter(*scores[inside].T, **{**points, **kwargs}, label="inside")
            ax.scatter(
                *scores[~inside].T,
                **{**points, "color": "C3", "marker": "x", **kwargs},
                label="outside",
            )
        statistic = self.statistic or DomainStatistic.T2_OR_Q
        if statistic != DomainStatistic.Q:
            ax.axvline(self.t2_limit_, color="0.3", ls="--", lw=1)
        if statistic != DomainStatistic.T2 and self.q_limit_ > 0:
            ax.axhline(self.q_limit_, color="0.3", ls="--", lw=1)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
                axis.set_major_formatter(FormatStrFormatter("%g"))
                axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"Hotelling's T² ({self.n_components_} components)")
        ax.set_ylabel("Q residual")
        ax.set_title(f"Applicability domain ({self.confidence:.0%} limits)")
        ax.legend()
        return ax

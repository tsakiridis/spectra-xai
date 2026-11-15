# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.dissimilarity` measures how different spectra are.

Dissimilarities are chosen with the `Dissimilarity` enum and computed between a
reference set of spectra (e.g. a training set or spectral library) and other
spectra by `SpectralDissimilarity`, or in one call by `pairwise_dissimilarity`.
They drive the neighbour search of the local models (`spectraxai.local`), and
can be used on their own, e.g. to find the library spectra closest to a new one.

Larger values always mean more different spectra. Every method is computed with
matrix products, without loops over spectra or windows: Euclidean distances use
the expansion ‖x − r‖² = ‖x‖² + ‖r‖² − 2 x·r (scikit-learn's
``euclidean_distances``, which trades a little precision, about 1e-8, for
speed), correlations and angles are products of normalized spectra, the
moving-window correlation one product of all normalized windows side by side,
and SID ∑(p − q)(log p − log q) is expanded into products too. The methods are:

* **Euclidean distance** of the spectra (`Dissimilarity.EUCLIDEAN`).
* **Projections** that first compress the spectra, as the R package resemble does:
  Euclidean distance of the principal-component scores (`PCA_EUCLIDEAN`), the
  Mahalanobis distance in principal-component space, i.e. the Euclidean distance
  of the standardized scores (`PCA_MAHALANOBIS`), and the Mahalanobis distance in
  the space of partial least squares scores (`PLS_MAHALANOBIS`), which needs the
  outputs and so favours the directions that predict them. (The Mahalanobis
  distance of whole spectra is not offered: with more wavelengths than spectra
  their covariance matrix is singular.)
* **Correlation**: one minus the Pearson correlation of two spectra
  (`CORRELATION`), or of the mean correlation over a moving window of
  wavelengths (`MOVING_CORRELATION`), which weighs local shape more.
* **Spectral similarity measures** from remote sensing (Du et al., 2004):
  the spectral angle (`SAM`, in radians; insensitive to the overall intensity),
  the spectral information divergence (`SID`, the symmetric Kullback–Leibler
  divergence of the spectra normalized to sum to one; needs non-negative
  spectra), and their combinations SID × tan(SAM) (`SID_SAM_TAN`) and
  SID × sin(SAM) (`SID_SAM_SIN`), which separate spectra better than either
  alone.

Dissimilarities computed elsewhere, e.g. from other data than the spectra such
as geographic distance, soil classes or other measurements, can be passed to the
local models as `Dissimilarity.PRECOMPUTED` matrices; a weighted sum of such a
matrix and a spectral one combines both.

References
----------
Du, Y., Chang, C.-I., Ren, H., Chang, C.-C., Jensen, J. O. and D'Amico, F. M.
(2004). New hyperspectral discrimination measure for spectral characterization.
*Optical Engineering*, 43(8), 1777–1786.

Ramirez-Lopez, L., Behrens, T., Schmidt, K., Stevens, A., Demattê, J. A. M. and
Scholten, T. (2013). The spectrum-based learner: A new local approach for
modeling soil vis–NIR spectra of complex datasets. *Geoderma*, 195–196, 268–279.

"""

import numbers
from enum import Enum
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array, check_is_fitted

from spectraxai.exceptions import SpectraXAIValueError


class Dissimilarity(str, Enum):
    """How the dissimilarity of two spectra is measured (see the module docs)."""

    EUCLIDEAN = "Euclidean distance"
    PCA_EUCLIDEAN = "Euclidean distance of principal-component scores"
    PCA_MAHALANOBIS = "Mahalanobis distance in principal-component space"
    PLS_MAHALANOBIS = "Mahalanobis distance in partial least squares space"
    CORRELATION = "One minus the correlation"
    MOVING_CORRELATION = "One minus the mean moving-window correlation"
    SAM = "Spectral angle"
    SID = "Spectral information divergence"
    SID_SAM_TAN = "SID x tan(SAM)"
    SID_SAM_SIN = "SID x sin(SAM)"
    PRECOMPUTED = "Precomputed dissimilarity matrix"
    """A matrix computed elsewhere, passed when predicting (local models only)."""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Dissimilarity.{self.name}"


_PROJECTIONS = frozenset(
    {
        Dissimilarity.PCA_EUCLIDEAN,
        Dissimilarity.PCA_MAHALANOBIS,
        Dissimilarity.PLS_MAHALANOBIS,
    }
)
_DIVERGENCES = frozenset(
    {Dissimilarity.SID, Dissimilarity.SID_SAM_TAN, Dissimilarity.SID_SAM_SIN}
)


def _unit_rows(X: np.ndarray, center: bool) -> np.ndarray:
    Z = X - X.mean(axis=1, keepdims=True) if center else X
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(norm > 0, norm, 1.0)


def _angle(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Angles (radians) between rows of unit length."""
    return np.arccos(np.clip(U @ V.T, -1.0, 1.0))


def _probabilities(X: np.ndarray) -> np.ndarray:
    if np.any(X < 0):
        raise SpectraXAIValueError(
            "SID needs non-negative spectra (e.g. reflectance or absorbance, not "
            "derivatives)"
        )
    P = X + np.finfo(float).tiny
    return P / P.sum(axis=1, keepdims=True)


def _check_n_components(value: Any) -> None:
    if value is None:
        return
    integral = isinstance(value, numbers.Integral) and not isinstance(value, bool)
    fraction = isinstance(value, float) and 0 < value < 1
    if not ((integral and int(value) >= 1) or fraction):
        raise SpectraXAIValueError(
            "n_components should be None, a positive integer or (for PCA) a "
            f"fraction of explained variance in (0, 1), got {value!r}"
        )


class SpectralDissimilarity(BaseEstimator):
    """Dissimilarity of spectra to a reference set of spectra.

    Parameters
    ----------
    method: `Dissimilarity`
        How dissimilarity is measured. Defaults to
        `Dissimilarity.PCA_MAHALANOBIS`.
    n_components: `int`, `float` or None
        For the projections (``PCA_*``, ``PLS_*``): the number of components, or
        for PCA the fraction of variance to explain. Defaults to None: 99% of the
        variance for PCA, 10 components for PLS (capped by the data).
    window: `int`
        For `Dissimilarity.MOVING_CORRELATION`, the number of consecutive
        wavelengths in each window. Defaults to 11.

    Examples
    --------
    >>> import numpy as np
    >>> from spectraxai.dissimilarity import Dissimilarity, SpectralDissimilarity
    >>> rng = np.random.default_rng(0)
    >>> library, new = rng.uniform(size=(50, 30)), rng.uniform(size=(3, 30))
    >>> d = SpectralDissimilarity(Dissimilarity.SID_SAM_TAN).fit(library)
    >>> d.pairwise(new).shape
    (3, 50)

    """

    def __init__(
        self,
        method: Dissimilarity = Dissimilarity.PCA_MAHALANOBIS,
        *,
        n_components: int | float | None = None,
        window: int = 11,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.window = window

    def _check_params(self) -> None:
        if not isinstance(self.method, Dissimilarity):
            raise SpectraXAIValueError(
                "method should be a Dissimilarity member, e.g. "
                f"Dissimilarity.PCA_MAHALANOBIS, got {self.method!r}"
            )
        if self.method == Dissimilarity.PRECOMPUTED:
            raise SpectraXAIValueError(
                "Dissimilarity.PRECOMPUTED is not computed from spectra; pass the "
                "matrix itself where it is used"
            )
        _check_n_components(self.n_components)
        if self.n_components is not None and self.method not in _PROJECTIONS:
            raise SpectraXAIValueError(
                f"n_components applies to PCA_* and PLS_* methods, not {self.method}"
            )
        if (
            isinstance(self.n_components, float)
            and self.method == Dissimilarity.PLS_MAHALANOBIS
        ):
            raise SpectraXAIValueError("PLS needs an integer n_components")
        w = self.window
        if isinstance(w, bool) or not isinstance(w, numbers.Integral) or w < 2:
            raise SpectraXAIValueError(f"window should be an integer >= 2, got {w!r}")

    def fit(self, X: Any, y: Any = None) -> Self:
        """Set the reference spectra (and learn the projection, if any).

        Parameters
        ----------
        X: array-like of shape (n_reference, n_features)
            The reference spectra
        y: array-like of shape (n_reference,) or (n_reference, n_outputs)
            The outputs, needed by `Dissimilarity.PLS_MAHALANOBIS` only

        Returns
        -------
        The fitted dissimilarity (``self``).

        """
        self._check_params()
        X = np.asarray(check_array(X, ensure_min_samples=2), dtype=float)
        method = self.method
        if method == Dissimilarity.MOVING_CORRELATION and self.window > X.shape[1]:
            raise SpectraXAIValueError(
                f"window ({self.window}) exceeds the number of features ({X.shape[1]})"
            )
        self.reference_ = X
        self.projection_: Any = None
        if method in (Dissimilarity.PCA_EUCLIDEAN, Dissimilarity.PCA_MAHALANOBIS):
            n = 0.99 if self.n_components is None else self.n_components
            if isinstance(n, numbers.Integral):
                n = min(int(n), *X.shape)
            self.projection_ = PCA(
                n_components=n,
                whiten=method == Dissimilarity.PCA_MAHALANOBIS,
                svd_solver="full",
            ).fit(X)
        elif method == Dissimilarity.PLS_MAHALANOBIS:
            if y is None:
                raise SpectraXAIValueError("PLS_MAHALANOBIS needs the outputs y")
            n = 10 if self.n_components is None else int(self.n_components)
            n = max(1, min(n, X.shape[1], len(X) - 1))
            self.projection_ = PLSRegression(n_components=n).fit(X, y)
            scores = self.projection_.transform(X)
            std = scores.std(axis=0, ddof=1)
            self.score_scale_ = np.where(std > 0, std, 1.0)
        elif method in _DIVERGENCES:
            _probabilities(X)  # fail early on negative spectra
        self.reference_embedding_ = self._embed(X)
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """The representation of the spectra that the method compares."""
        method = self.method
        if method in (Dissimilarity.PCA_EUCLIDEAN, Dissimilarity.PCA_MAHALANOBIS):
            return self.projection_.transform(X)
        if method == Dissimilarity.PLS_MAHALANOBIS:
            return self.projection_.transform(X) / self.score_scale_
        if method == Dissimilarity.CORRELATION:
            return _unit_rows(X, center=True)
        if method == Dissimilarity.MOVING_CORRELATION:
            # every window centred and of unit norm, all windows side by side and
            # scaled so that one matrix product gives the mean correlation
            windows = np.lib.stride_tricks.sliding_window_view(
                X, int(self.window), axis=1
            )
            Z = windows - windows.mean(axis=2, keepdims=True)
            norm = np.linalg.norm(Z, axis=2, keepdims=True)
            Z = Z / np.where(norm > 0, norm, 1.0) / np.sqrt(windows.shape[1])
            return Z.reshape(len(X), -1)
        if method == Dissimilarity.SAM:
            return _unit_rows(X, center=False)
        if method in _DIVERGENCES:
            return _probabilities(X)
        return X

    def pairwise(self, X: Any = None) -> np.ndarray:
        """Dissimilarity of each spectrum of ``X`` to each reference spectrum.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features), optional
            The spectra to compare. Defaults to the reference spectra.

        Returns
        -------
        `np.ndarray`
            A non-negative matrix of shape (n_samples, n_reference).

        """
        check_is_fitted(self, "reference_")
        if X is None:
            E = self.reference_embedding_
            X = self.reference_
        else:
            X = np.asarray(check_array(X), dtype=float)
            if X.shape[1] != self.reference_.shape[1]:
                raise SpectraXAIValueError(
                    f"X has {X.shape[1]} features, but the reference spectra have "
                    f"{self.reference_.shape[1]}"
                )
            E = self._embed(X)
        R = self.reference_embedding_
        method = self.method
        if method in (Dissimilarity.CORRELATION, Dissimilarity.MOVING_CORRELATION):
            return np.clip(1.0 - E @ R.T, 0.0, 2.0)
        if method == Dissimilarity.SAM:
            return _angle(E, R)
        if method in _DIVERGENCES:
            # sum (p - q)(log p - log q), expanded into matrix products
            logE, logR = np.log(E), np.log(R)
            sid = (
                np.sum(E * logE, axis=1)[:, None]
                + np.sum(R * logR, axis=1)[None, :]
                - E @ logR.T
                - logE @ R.T
            )
            sid = np.maximum(sid, 0.0)
            if method == Dissimilarity.SID:
                return sid
            # the angle is scale-invariant, so that of the normalized spectra
            angle = _angle(_unit_rows(E, False), _unit_rows(R, False))
            trig = np.tan if method == Dissimilarity.SID_SAM_TAN else np.sin
            return sid * trig(angle)
        return euclidean_distances(E, R)


def pairwise_dissimilarity(
    reference: Any,
    X: Any = None,
    method: Dissimilarity = Dissimilarity.PCA_MAHALANOBIS,
    *,
    y: Any = None,
    n_components: int | float | None = None,
    window: int = 11,
) -> np.ndarray:
    """Dissimilarity of spectra ``X`` to ``reference`` spectra in one call.

    Shorthand for ``SpectralDissimilarity(method, ...).fit(reference, y)
    .pairwise(X)``; see `SpectralDissimilarity` for the parameters.

    Returns
    -------
    `np.ndarray`
        A matrix of shape (n_samples, n_reference), or (n_reference, n_reference)
        when ``X`` is None.

    """
    return (
        SpectralDissimilarity(method, n_components=n_components, window=window)
        .fit(reference, y)
        .pairwise(X)
    )

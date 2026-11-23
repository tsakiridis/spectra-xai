# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The module `spectraxai.local` defines local (memory-based learning) regressors.

A local regressor keeps the training spectra in memory. To predict a new spectrum it
finds its `k` nearest training spectra and fits a model on these neighbours only,
as the R package resemble does. This suits large, diverse spectral libraries in
which the relation between spectrum and property changes across the library (e.g.
across soil types), where a single global model compromises.

Two local models are provided, both usable as scikit-learn estimators and through
`spectraxai.models.StandardModel` (``Model.LOCAL_PLS`` and ``Model.LOCAL_GPR``):

* `LocalPLSRegressor` fits a partial least squares model per neighbourhood;
* `LocalGPRRegressor` fits a Gaussian process with a linear (dot-product) kernel
  per neighbourhood, as resemble's ``local_fit_gpr``.

**Choosing the neighbours.** ``dissimilarity`` sets how the nearest training
spectra are found: a `spectraxai.dissimilarity.Dissimilarity` member (by default,
None, the Mahalanobis distance in principal-component space), a
`spectraxai.dissimilarity.SpectralDissimilarity` to set its options, or
`Dissimilarity.PRECOMPUTED` to pass a matrix computed elsewhere, e.g. from other
data than the spectra:

>>> import numpy as np
>>> from spectraxai.dissimilarity import Dissimilarity, SpectralDissimilarity
>>> from spectraxai.local import LocalPLSRegressor
>>> rng = np.random.default_rng(0)
>>> X, y = rng.uniform(size=(60, 20)), rng.normal(size=60)
>>> sam = LocalPLSRegressor(k=30, n_components=3, dissimilarity=Dissimilarity.SAM)
>>> pca = SpectralDissimilarity(Dissimilarity.PCA_EUCLIDEAN, n_components=5)
>>> euclid = LocalPLSRegressor(k=30, n_components=3, dissimilarity=pca)
>>> # a precomputed matrix: one row per spectrum to predict, one column per
>>> # training spectrum (here the distance between sampling locations)
>>> xy = rng.uniform(size=(62, 2))
>>> geo = np.linalg.norm(xy[:, None] - xy[None, :60], axis=-1)
>>> local = LocalPLSRegressor(30, 3, dissimilarity=Dissimilarity.PRECOMPUTED)
>>> local.fit(X, y).predict(X[:2], dissimilarity=geo[:2]).shape
(2,)

Test spectra that share the same neighbourhood share a single local model.

"""

import numbers
from collections.abc import Iterator
from typing import Any, Self, cast

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from spectraxai.dissimilarity import Dissimilarity, SpectralDissimilarity
from spectraxai.exceptions import SpectraXAIValueError

_CHUNK = 2**22
"""Largest number of dissimilarities held in memory at once."""


def pls_vip(model: PLSRegression) -> np.ndarray:
    """Variable importance in projection (VIP) of a fitted PLS model, per feature."""
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p = w.shape[0]
    # variance of the outputs explained by each component
    s = np.diag(t.T @ t @ q.T @ q)
    w_norm = (w / np.linalg.norm(w, axis=0)) ** 2
    return np.sqrt(p * (w_norm @ s) / np.sum(s))


def _check_int(name: str, value: Any, minimum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < minimum
    ):
        raise SpectraXAIValueError(
            f"{name} should be an integer >= {minimum}, got {value!r}"
        )


class _LocalRegressor(RegressorMixin, BaseEstimator):
    """Neighbour search and per-neighbourhood fitting shared by local regressors."""

    k: int
    dissimilarity: Dissimilarity | SpectralDissimilarity | None

    def _local_estimator(self, n_samples: int, n_features: int) -> Any:
        """An unfitted model for a neighbourhood of ``n_samples`` spectra."""
        raise NotImplementedError

    def _check_params(self) -> None:
        _check_int("k", self.k, 2)
        if self.dissimilarity is not None and not isinstance(
            self.dissimilarity, (Dissimilarity, SpectralDissimilarity)
        ):
            raise SpectraXAIValueError(
                "dissimilarity should be a Dissimilarity member (e.g. "
                "Dissimilarity.PCA_MAHALANOBIS) or a SpectralDissimilarity, got "
                f"{self.dissimilarity!r}"
            )

    @property
    def _precomputed(self) -> bool:
        return self.dissimilarity is Dissimilarity.PRECOMPUTED

    def fit(self, X: Any, y: Any) -> Self:
        """Store the training spectra and prepare the neighbour search.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training spectra
        y: array-like of shape (n_samples,) or (n_samples, n_outputs)
            The training outputs

        Returns
        -------
        The fitted regressor (``self``).

        """
        self._check_params()
        X, y = validate_data(
            self, X, y, multi_output=True, y_numeric=True, ensure_min_samples=2
        )
        y = np.asarray(y, dtype=float)
        self._y_1d = y.ndim == 1
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = y.reshape(len(y), -1)
        self.k_ = min(int(self.k), len(X))
        self.dissimilarity_: SpectralDissimilarity | None = None
        if not self._precomputed:
            measure = self.dissimilarity
            if measure is None:
                measure = Dissimilarity.PCA_MAHALANOBIS
            if isinstance(measure, Dissimilarity):
                measure = SpectralDissimilarity(measure)
            self.dissimilarity_ = cast(SpectralDissimilarity, clone(measure))
            self.dissimilarity_.fit(self.X_train_, y)
        return self

    def _check_matrix(self, X: np.ndarray, dissimilarity: Any) -> np.ndarray | None:
        """Validate a precomputed matrix, or that none is passed otherwise."""
        if not self._precomputed:
            if dissimilarity is not None:
                raise SpectraXAIValueError(
                    "a dissimilarity matrix is passed only with "
                    "dissimilarity=Dissimilarity.PRECOMPUTED"
                )
            return None
        if dissimilarity is None:
            raise SpectraXAIValueError(
                "With Dissimilarity.PRECOMPUTED pass the matrix of dissimilarities "
                "between the spectra and the training spectra: "
                "dissimilarity=D, of shape (n_samples, n_training_samples)"
            )
        D = np.asarray(dissimilarity, dtype=float)
        expected = (len(X), len(self.X_train_))
        if D.shape != expected:
            raise SpectraXAIValueError(
                f"the dissimilarity matrix should have shape {expected} (one row per "
                f"spectrum, one column per training spectrum), got {D.shape}"
            )
        if not np.all(np.isfinite(D)) or np.any(D < 0):
            raise SpectraXAIValueError(
                "the dissimilarity matrix should be finite and non-negative"
            )
        return D

    def _neighbourhoods(
        self, X: np.ndarray, dissimilarity: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(training indices, rows of X)`` per distinct neighbourhood."""
        D = self._check_matrix(X, dissimilarity)
        k, n_train = self.k_, len(self.X_train_)
        idx = np.empty((len(X), k), dtype=np.intp)
        step = max(1, _CHUNK // n_train)
        for start in range(0, len(X), step):
            rows = slice(start, start + step)
            if k == n_train:
                idx[rows] = np.arange(n_train)
                continue
            if D is not None:
                dist = D[rows]
            else:
                dist = cast(SpectralDissimilarity, self.dissimilarity_).pairwise(
                    X[rows]
                )
            idx[rows] = np.argpartition(dist, k - 1, axis=1)[:, :k]
        idx = np.sort(idx, axis=1)
        _, first, groups = np.unique(
            idx, axis=0, return_index=True, return_inverse=True
        )
        # rows of each neighbourhood, by one sort instead of a scan per group
        order = np.argsort(np.ravel(groups), kind="stable")
        bounds = np.cumsum(np.bincount(np.ravel(groups), minlength=len(first)))
        for row, rows in zip(first, np.split(order, bounds[:-1])):
            yield idx[row], rows

    def _fit_local(self, train: np.ndarray) -> Any:
        model = self._local_estimator(len(train), self.X_train_.shape[1])
        y = self.y_train_[train]
        return model.fit(self.X_train_[train], y.ravel() if self._y_1d else y)

    def _validated(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "X_train_")
        return np.asarray(validate_data(self, X, reset=False), dtype=float)

    def predict(self, X: Any, dissimilarity: Any = None) -> np.ndarray:
        """Predict each spectrum with a model fitted on its `k` nearest neighbours.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The spectra to predict
        dissimilarity: array-like of shape (n_samples, n_training_samples)
            With ``dissimilarity=Dissimilarity.PRECOMPUTED`` only: the
            dissimilarity of each spectrum to each training spectrum.

        Returns
        -------
        `np.ndarray`
            The predictions, of shape (n_samples,) or (n_samples, n_outputs) as the
            training outputs.

        """
        X = self._validated(X)
        y_hat = np.empty((len(X), self.y_train_.shape[1]))
        for train, rows in self._neighbourhoods(X, dissimilarity):
            pred = self._fit_local(train).predict(X[rows])
            y_hat[rows] = np.asarray(pred).reshape(len(rows), -1)
        return y_hat.ravel() if self._y_1d else y_hat

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        return tags


class LocalPLSRegressor(_LocalRegressor):
    """Local partial least squares regression (memory-based learning).

    Each spectrum is predicted by a `sklearn.cross_decomposition.PLSRegression`
    fitted on its `k` nearest training spectra. With `k` equal to the number of
    training spectra this is global PLS.

    Parameters
    ----------
    k: `int`
        Number of nearest training spectra in each local model (capped at the
        number of training spectra). Defaults to 100.
    n_components: `int`
        Number of PLS components of each local model, capped at ``k - 1`` and at
        the number of features. Defaults to 10.
    dissimilarity: `Dissimilarity`, `SpectralDissimilarity` or None
        How the nearest training spectra are found (see `spectraxai.local`).
        Defaults to None, i.e. `Dissimilarity.PCA_MAHALANOBIS`.

    Examples
    --------
    >>> import numpy as np
    >>> from spectraxai.local import LocalPLSRegressor
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(60, 20)), rng.normal(size=60)
    >>> LocalPLSRegressor(k=30, n_components=3).fit(X, y).predict(X[:2]).shape
    (2,)

    """

    def __init__(
        self,
        k: int = 100,
        n_components: int = 10,
        *,
        dissimilarity: Dissimilarity | SpectralDissimilarity | None = None,
    ) -> None:
        self.k = k
        self.n_components = n_components
        self.dissimilarity = dissimilarity

    def _check_params(self) -> None:
        super()._check_params()
        _check_int("n_components", self.n_components, 1)

    def _local_estimator(self, n_samples: int, n_features: int) -> Any:
        n = max(1, min(int(self.n_components), n_samples - 1, n_features))
        return PLSRegression(n_components=n)

    def vip(self, X: Any = None, dissimilarity: Any = None) -> np.ndarray:
        """Mean VIP of the local models that predict ``X``, per feature.

        Each spectrum contributes the VIP scores of its own local model, so this
        is the importance of each wavelength for predicting ``X``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features), optional
            The spectra whose local models are averaged. Defaults to the training
            spectra.
        dissimilarity: array-like of shape (n_samples, n_training_samples)
            With ``Dissimilarity.PRECOMPUTED`` only, as in `predict` (a square
            matrix for the training spectra).

        Returns
        -------
        `np.ndarray`
            The mean VIP score of each feature.

        """
        X = self.X_train_ if X is None else self._validated(X)
        total = np.zeros(self.X_train_.shape[1])
        for train, rows in self._neighbourhoods(X, dissimilarity):
            total += len(rows) * pls_vip(self._fit_local(train))
        return total / len(X)


class LocalGPRRegressor(_LocalRegressor):
    """Local Gaussian process regression (memory-based learning).

    Each spectrum is predicted by a Gaussian process with a linear (dot-product)
    kernel and fixed noise variance, fitted on its `k` nearest training spectra
    after standardizing them, as ``local_fit_gpr`` of the R package resemble. The
    kernel has no hyperparameters to optimize, so each local fit is a single
    linear solve.

    Parameters
    ----------
    k: `int`
        Number of nearest training spectra in each local model (capped at the
        number of training spectra). Defaults to 100.
    noise_variance: `float`
        Variance of the noise added to the kernel diagonal (on standardized
        outputs); larger values smooth the local models. Defaults to 0.001.
    dissimilarity: `Dissimilarity`, `SpectralDissimilarity` or None
        How the nearest training spectra are found (see `spectraxai.local`).
        Defaults to None, i.e. `Dissimilarity.PCA_MAHALANOBIS`.

    """

    def __init__(
        self,
        k: int = 100,
        noise_variance: float = 0.001,
        *,
        dissimilarity: Dissimilarity | SpectralDissimilarity | None = None,
    ) -> None:
        self.k = k
        self.noise_variance = noise_variance
        self.dissimilarity = dissimilarity

    def _check_params(self) -> None:
        super()._check_params()
        v = self.noise_variance
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not v > 0:
            raise SpectraXAIValueError(f"noise_variance should be positive, got {v!r}")

    def _local_estimator(self, n_samples: int, n_features: int) -> Any:
        gpr = GaussianProcessRegressor(
            kernel=DotProduct(sigma_0=0.0, sigma_0_bounds="fixed"),
            alpha=float(self.noise_variance),
            optimizer=cast(Any, None),  # fixed kernel, nothing to tune
            normalize_y=True,
        )
        return make_pipeline(StandardScaler(), clone(gpr))

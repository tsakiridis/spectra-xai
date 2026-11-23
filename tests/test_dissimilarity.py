# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral dissimilarities and precomputed dissimilarity matrices."""

from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from spectraxai.dataset import Dataset
from spectraxai.dissimilarity import (
    Dissimilarity,
    SpectralDissimilarity,
    pairwise_dissimilarity,
)
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.local import LocalGPRRegressor, LocalPLSRegressor
from spectraxai.models import Model, StandardModel

COMPUTED = [m for m in Dissimilarity if m is not Dissimilarity.PRECOMPUTED]


@pytest.fixture
def spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive spectra with more samples than wavelengths, and outputs."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.1, 1.0, size=(40, 12))
    y = X @ rng.normal(size=12)
    return X[:30], X[30:], y[:30]


def _sid(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


# -- every method -------------------------------------------------------------------


@pytest.mark.parametrize("method", COMPUTED, ids=str)
def test_shape_and_self_dissimilarity(spectra: Any, method: Dissimilarity) -> None:
    ref, new, y = spectra
    d = SpectralDissimilarity(method).fit(ref, y)
    assert d.pairwise(new).shape == (len(new), len(ref))
    D = d.pairwise()
    assert D.shape == (len(ref), len(ref)) and np.all(D >= 0)
    np.testing.assert_allclose(np.diag(D), 0, atol=1e-6)
    np.testing.assert_allclose(D, D.T, atol=1e-8)
    np.testing.assert_allclose(D, d.pairwise(ref), atol=1e-7)
    np.testing.assert_array_equal(
        pairwise_dissimilarity(ref, new, method, y=y), d.pairwise(new)
    )


def test_enum_prints_its_name() -> None:
    assert str(Dissimilarity.SID_SAM_TAN) == "SID_SAM_TAN"
    assert repr(Dissimilarity.SAM) == "Dissimilarity.SAM"


# -- against reference implementations ----------------------------------------------


def test_euclidean(spectra: Any) -> None:
    ref, new, _ = spectra
    np.testing.assert_allclose(
        pairwise_dissimilarity(ref, new, Dissimilarity.EUCLIDEAN), cdist(new, ref)
    )


def test_pca_with_all_components(spectra: Any) -> None:
    ref, new, _ = spectra
    n = ref.shape[1]
    euclid = pairwise_dissimilarity(
        ref, new, Dissimilarity.PCA_EUCLIDEAN, n_components=n
    )
    np.testing.assert_allclose(euclid, cdist(new, ref))
    # whitened scores: the Mahalanobis distance of the spectra themselves
    mahalanobis = pairwise_dissimilarity(
        ref, new, Dissimilarity.PCA_MAHALANOBIS, n_components=n
    )
    VI = np.linalg.inv(np.cov(ref, rowvar=False))
    np.testing.assert_allclose(mahalanobis, cdist(new, ref, "mahalanobis", VI=VI))


def test_pca_components_and_variance(spectra: Any) -> None:
    ref, _, _ = spectra
    d = SpectralDissimilarity(Dissimilarity.PCA_MAHALANOBIS, n_components=3).fit(ref)
    assert d.projection_.n_components_ == 3
    d = SpectralDissimilarity(Dissimilarity.PCA_EUCLIDEAN, n_components=0.5).fit(ref)
    assert 1 <= d.projection_.n_components_ < ref.shape[1]


def test_pls_mahalanobis_uses_the_outputs(spectra: Any) -> None:
    ref, new, y = spectra
    d = SpectralDissimilarity(Dissimilarity.PLS_MAHALANOBIS, n_components=2)
    D = d.fit(ref, y).pairwise(new)
    scores = d.projection_.transform(ref)
    assert np.allclose(np.std(scores / d.score_scale_, axis=0, ddof=1), 1)
    other = SpectralDissimilarity(Dissimilarity.PLS_MAHALANOBIS, n_components=2)
    assert not np.allclose(D, other.fit(ref, -(y**2)).pairwise(new))
    with pytest.raises(SpectraXAIValueError, match="needs the outputs"):
        SpectralDissimilarity(Dissimilarity.PLS_MAHALANOBIS).fit(ref)


def test_correlation(spectra: Any) -> None:
    ref, new, _ = spectra
    D = pairwise_dissimilarity(ref, new, Dissimilarity.CORRELATION)
    np.testing.assert_allclose(D, cdist(new, ref, "correlation"), atol=1e-12)


def test_moving_correlation(spectra: Any) -> None:
    ref, new, _ = spectra
    whole = pairwise_dissimilarity(
        ref, new, Dissimilarity.MOVING_CORRELATION, window=ref.shape[1]
    )
    np.testing.assert_allclose(
        whole, pairwise_dissimilarity(ref, new, Dissimilarity.CORRELATION)
    )
    D = pairwise_dissimilarity(ref, new, Dissimilarity.MOVING_CORRELATION, window=4)
    windows = [
        1 - np.diag(np.corrcoef(new[0, s : s + 4], ref[1, s : s + 4]), 1)[0]
        for s in range(ref.shape[1] - 3)
    ]
    assert D[0, 1] == pytest.approx(np.mean(windows))


def test_spectral_angle(spectra: Any) -> None:
    ref, new, _ = spectra
    D = pairwise_dissimilarity(ref, new, Dissimilarity.SAM)
    np.testing.assert_allclose(D, np.arccos(1 - cdist(new, ref, "cosine")))
    # insensitive to the intensity of the spectra
    np.testing.assert_allclose(
        pairwise_dissimilarity(3 * ref, 0.5 * new, Dissimilarity.SAM), D, atol=1e-7
    )


def test_spectral_information_divergence(spectra: Any) -> None:
    ref, new, _ = spectra
    D = pairwise_dissimilarity(ref, new, Dissimilarity.SID)
    assert D[2, 5] == pytest.approx(_sid(new[2], ref[5]))
    np.testing.assert_allclose(
        pairwise_dissimilarity(2 * ref, new, Dissimilarity.SID), D, atol=1e-12
    )


def test_sid_sam_combinations(spectra: Any) -> None:
    ref, new, _ = spectra
    sid = pairwise_dissimilarity(ref, new, Dissimilarity.SID)
    sam = pairwise_dissimilarity(ref, new, Dissimilarity.SAM)
    np.testing.assert_allclose(
        pairwise_dissimilarity(ref, new, Dissimilarity.SID_SAM_TAN),
        sid * np.tan(sam),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        pairwise_dissimilarity(ref, new, Dissimilarity.SID_SAM_SIN),
        sid * np.sin(sam),
        rtol=1e-6,
    )


def test_divergences_need_non_negative_spectra(spectra: Any) -> None:
    ref, new, _ = spectra
    for method in (Dissimilarity.SID, Dissimilarity.SID_SAM_TAN):
        with pytest.raises(SpectraXAIValueError, match="non-negative"):
            SpectralDissimilarity(method).fit(ref - 0.5)
        with pytest.raises(SpectraXAIValueError, match="non-negative"):
            SpectralDissimilarity(method).fit(ref).pairwise(new - 0.5)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "SAM"}, "Dissimilarity member"),
        ({"method": Dissimilarity.PRECOMPUTED}, "not computed"),
        ({"method": Dissimilarity.SAM, "n_components": 3}, "n_components applies"),
        ({"method": Dissimilarity.PCA_EUCLIDEAN, "n_components": 0}, "n_components"),
        ({"method": Dissimilarity.PCA_EUCLIDEAN, "n_components": 1.5}, "n_components"),
        ({"method": Dissimilarity.PLS_MAHALANOBIS, "n_components": 0.5}, "integer"),
        ({"method": Dissimilarity.MOVING_CORRELATION, "window": 1}, "window"),
        ({"method": Dissimilarity.MOVING_CORRELATION, "window": 50}, "exceeds"),
    ],
)
def test_invalid_parameters(spectra: Any, kwargs: dict, match: str) -> None:
    ref, _, y = spectra
    with pytest.raises(SpectraXAIValueError, match=match):
        SpectralDissimilarity(**kwargs).fit(ref, y)


def test_feature_mismatch(spectra: Any) -> None:
    ref, new, _ = spectra
    d = SpectralDissimilarity(Dissimilarity.SAM).fit(ref)
    with pytest.raises(SpectraXAIValueError, match="features"):
        d.pairwise(new[:, :5])


# -- precomputed matrices in local models -------------------------------------------


@pytest.mark.parametrize("cls", [LocalPLSRegressor, LocalGPRRegressor])
def test_precomputed_equals_built_in(synthetic: Dataset, cls: type) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    trn, tst = X[:60], X[60:]
    built_in = cls(k=20, dissimilarity=Dissimilarity.SAM).fit(trn, y[:60])
    pre = cls(k=20, dissimilarity=Dissimilarity.PRECOMPUTED).fit(trn, y[:60])
    D = pairwise_dissimilarity(trn, tst, Dissimilarity.SAM)
    np.testing.assert_allclose(
        pre.predict(tst, dissimilarity=D), built_in.predict(tst), rtol=1e-10
    )


def test_neighbours_from_other_data(synthetic: Dataset) -> None:
    """Neighbours chosen by an external grouping (e.g. soil class) of the samples."""
    X, y = synthetic.X, synthetic.Y.ravel().copy()
    group = np.arange(len(X)) % 2
    y[group == 1] += 10  # an offset that the spectra do not show
    D = (group[:, None] != group[None, :]).astype(float)
    trn, tst = np.arange(60), np.arange(60, len(X))
    by_group = LocalPLSRegressor(
        k=25, n_components=3, dissimilarity=Dissimilarity.PRECOMPUTED
    )
    by_group.fit(X[trn], y[trn])
    by_spectra = LocalPLSRegressor(k=25, n_components=3).fit(X[trn], y[trn])
    err_group = np.abs(by_group.predict(X[tst], D[np.ix_(tst, trn)]) - y[tst]).mean()
    err_spectra = np.abs(by_spectra.predict(X[tst]) - y[tst]).mean()
    assert err_group < err_spectra / 3


def test_combined_spectral_and_external_dissimilarity(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    spectral = pairwise_dissimilarity(X[:60], X[60:], Dissimilarity.SID_SAM_TAN)
    external = np.random.default_rng(0).uniform(size=spectral.shape)
    combined = spectral / spectral.max() + 0.5 * external
    local = LocalPLSRegressor(
        k=20, n_components=3, dissimilarity=Dissimilarity.PRECOMPUTED
    )
    pred = local.fit(X[:60], y[:60]).predict(X[60:], dissimilarity=combined)
    assert np.all(np.isfinite(pred))
    vip = local.vip(X[60:], dissimilarity=combined)
    assert vip.shape == (X.shape[1],)


def test_precomputed_errors(synthetic: Dataset) -> None:
    X, y = synthetic.X, synthetic.Y.ravel()
    pre = LocalPLSRegressor(k=10, dissimilarity=Dissimilarity.PRECOMPUTED).fit(X, y)
    with pytest.raises(SpectraXAIValueError, match="pass the matrix"):
        pre.predict(X[:3])
    with pytest.raises(SpectraXAIValueError, match="shape"):
        pre.predict(X[:3], dissimilarity=np.ones((3, 5)))
    with pytest.raises(SpectraXAIValueError, match="non-negative"):
        pre.predict(X[:3], dissimilarity=-np.ones((3, len(X))))
    with pytest.raises(SpectraXAIValueError, match="pass the matrix"):
        pre.vip()  # the training spectra need a square matrix
    spectral = LocalPLSRegressor(k=10).fit(X, y)
    with pytest.raises(SpectraXAIValueError, match="only with"):
        spectral.predict(X[:3], dissimilarity=np.ones((3, len(X))))


# -- through StandardModel -----------------------------------------------------------


def test_fit_and_predict_with_a_precomputed_matrix(synthetic: Dataset) -> None:
    D = pairwise_dissimilarity(synthetic.X, method=Dissimilarity.EUCLIDEAN)
    hyper = {"k": 20, "n_components": 3}
    pre = StandardModel(
        Model.LOCAL_PLS,
        init_hyperparameters={**hyper, "dissimilarity": Dissimilarity.PRECOMPUTED},
    )
    res = pre.fit_and_predict(synthetic, idx_trn=np.arange(60), dissimilarity=D)
    built_in = StandardModel(
        Model.LOCAL_PLS,
        init_hyperparameters={**hyper, "dissimilarity": Dissimilarity.EUCLIDEAN},
    ).fit_and_predict(synthetic, idx_trn=np.arange(60))
    assert res.loc[0, "RMSE"] == pytest.approx(built_in.loc[0, "RMSE"])
    assert res.loc[0, "importance_method"] == "VIP"
    assert pre.feature_importance() == (None, None)
    folds = np.array([np.arange(40), np.arange(40, 80)])
    by_fold = pre.fit_and_predict(synthetic, idx_tst=folds, dissimilarity=D)
    assert list(by_fold["fold"]) == [1, 2]
    with pytest.raises(SpectraXAIValueError, match="square"):
        pre.fit_and_predict(synthetic, idx_trn=np.arange(60), dissimilarity=D[:5])


def test_local_gpr_permutation_importance_with_a_matrix(synthetic: Dataset) -> None:
    D = pairwise_dissimilarity(synthetic.X, method=Dissimilarity.SAM)
    model = StandardModel(
        Model.LOCAL_GPR,
        init_hyperparameters={"k": 20, "dissimilarity": Dissimilarity.PRECOMPUTED},
    )
    res = model.fit_and_predict(
        synthetic,
        idx_trn=np.arange(60),
        dissimilarity=D,
        return_permutation_importance=True,
    )
    assert res.loc[0, "importance_method"] == "permutation"


def test_searching_dissimilarities(synthetic: Dataset) -> None:
    grid = {
        "k": [20],
        "n_components": [3],
        "dissimilarity": [Dissimilarity.SAM, Dissimilarity.SID_SAM_TAN],
    }
    model = StandardModel(Model.LOCAL_PLS, grid_search_hyperparameters=grid, cv=3)
    model.fit(synthetic)
    assert model.best_hyperparameters is not None
    assert model.best_hyperparameters["dissimilarity"] in grid["dissimilarity"]
    grid["dissimilarity"] = [Dissimilarity.PRECOMPUTED]
    with pytest.raises(SpectraXAIValueError, match="PRECOMPUTED"):
        StandardModel(Model.LOCAL_PLS, grid_search_hyperparameters=grid).fit(synthetic)

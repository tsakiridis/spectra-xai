# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Applicability domain: Hotelling's T² and Q residuals."""

from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats
from sklearn.utils.estimator_checks import parametrize_with_checks

from spectraxai.dataset import Dataset
from spectraxai.diagnostics import (
    ApplicabilityDomain,
    DomainStatistic,
    QLimit,
    q_limit,
    t2_limit,
)
from spectraxai.exceptions import SpectraXAIValueError

EIGENVALUES = np.array([9.0, 4.0, 1.0, 0.5, 0.3, 0.2, 0.1])


def _gaussian(n: int, seed: int) -> np.ndarray:
    """Samples whose covariance has eigenvalues `EIGENVALUES` (in a random basis)."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(np.random.default_rng(99).normal(size=(7, 7)))
    return (rng.normal(size=(n, 7)) * np.sqrt(EIGENVALUES)) @ basis.T


@pytest.fixture
def spectra() -> np.ndarray:
    """Low-rank spectra: three constituents plus a little noise."""
    rng = np.random.default_rng(0)
    bands = np.arange(60)
    peaks = np.exp(-(((bands[None, :] - np.array([[10], [30], [45]])) / 5) ** 2))
    X = rng.uniform(0.2, 1.0, size=(150, 3)) @ peaks
    return X + rng.normal(0, 0.005, size=X.shape)


# -- the limits --------------------------------------------------------------------


def test_t2_limit_formula() -> None:
    expected = 3 * 49 * 51 / (50 * 47) * stats.f.ppf(0.95, 3, 47)
    assert t2_limit(3, 50, 0.95) == pytest.approx(expected)


@pytest.mark.parametrize("method", list(QLimit), ids=str)
def test_q_limit_matches_the_simulated_quantile(method: QLimit) -> None:
    left_out = EIGENVALUES[2:]
    z = np.random.default_rng(0).normal(size=(400_000, left_out.size))
    q = (z**2) @ left_out  # Q of samples whose left-out scores are Gaussian
    for confidence in (0.9, 0.95, 0.99):
        simulated = np.quantile(q, confidence)
        # approximations: within a few percent (JM +6%, Box -5% at 0.99 here)
        limit = q_limit(left_out, confidence, method)
        assert limit == pytest.approx(simulated, rel=0.07)
    assert q_limit(np.zeros(3), 0.99, method) == 0.0


def test_box_q_limit_formula() -> None:
    lam = np.array([2.0, 1.0, 0.5])
    t1, t2 = lam.sum(), np.sum(lam**2)
    expected = t2 / t1 * stats.chi2.ppf(0.95, t1**2 / t2)
    assert q_limit(lam, 0.95) == pytest.approx(expected)


def test_jackson_mudholkar_breaks_on_unequal_eigenvalues() -> None:
    """One dominant left-out eigenvalue and a long tail of tiny ones (as in MIR)."""
    lam = np.concatenate([[1.0], np.full(100, 1e-2)])
    with pytest.raises(SpectraXAIValueError, match="QLimit.BOX"):
        q_limit(lam, 0.99, QLimit.JACKSON_MUDHOLKAR)
    z = np.random.default_rng(0).normal(size=(200_000, lam.size))
    simulated = np.quantile((z**2) @ lam, 0.99)
    # Box's limit stays defined, if rough for so extreme a case (14% low here)
    assert q_limit(lam, 0.99) == pytest.approx(simulated, rel=0.2)


def test_limits_hold_for_new_samples() -> None:
    """About 1 - confidence of new samples like the training ones exceed each limit."""
    exceed_t2, exceed_q = [], []
    for seed in range(20):
        domain = ApplicabilityDomain(n_components=2, confidence=0.95)
        domain.fit(_gaussian(100, seed))
        t2, q = domain.statistics(_gaussian(500, 1000 + seed)).T
        exceed_t2.append(np.mean(t2 > domain.t2_limit_))
        exceed_q.append(np.mean(q > domain.q_limit_))
    assert np.mean(exceed_t2) == pytest.approx(0.05, abs=0.015)
    assert np.mean(exceed_q) == pytest.approx(0.05, abs=0.02)


# -- flagging ------------------------------------------------------------------------


def test_training_spectra_are_mostly_inside(spectra: np.ndarray) -> None:
    domain = ApplicabilityDomain(n_components=3, confidence=0.99).fit(spectra)
    assert domain.inside(spectra).mean() >= 0.95
    assert domain.training_statistics_.shape == (len(spectra), 2)


def test_unusual_spectra_are_flagged_by_the_right_statistic(
    spectra: np.ndarray,
) -> None:
    domain = ApplicabilityDomain(n_components=3).fit(spectra)
    typical = spectra[:5]
    # far more of the same constituents: extreme scores, small residuals
    extreme = 4 * spectra[:5]
    # a constituent never seen: large residuals
    new_band = spectra[:5] + 0.3 * np.exp(-(((np.arange(60) - 20) / 3) ** 2))
    assert np.all(domain.predict(typical) == 1)
    t2, q = domain.statistics(extreme).T
    assert np.all(t2 > domain.t2_limit_)
    t2, q = domain.statistics(new_band).T
    assert np.all(q > domain.q_limit_)
    for statistic, flagged in [
        (DomainStatistic.T2, extreme),
        (DomainStatistic.Q, new_band),
    ]:
        domain.set_params(statistic=statistic)
        assert np.all(domain.predict(flagged) == -1)
    domain.set_params(statistic=DomainStatistic.Q)
    assert np.all(domain.predict(new_band) == -1)
    domain.set_params(statistic=None)  # T2_OR_Q
    assert np.all(domain.predict(np.vstack([extreme, new_band])) == -1)


def test_scores_follow_scikit_learn(spectra: np.ndarray) -> None:
    domain = ApplicabilityDomain(n_components=3).fit(spectra)
    X = np.vstack([spectra[:5], 4 * spectra[:5]])
    score = domain.score_samples(X)
    np.testing.assert_allclose(domain.decision_function(X), score + 1)
    np.testing.assert_array_equal(
        domain.predict(X), np.where(domain.decision_function(X) >= 0, 1, -1)
    )
    t2, q = domain.statistics(X).T
    expected = -np.maximum(t2 / domain.t2_limit_, q / domain.q_limit_)
    np.testing.assert_allclose(score, expected)


def test_statistics_match_a_direct_computation(spectra: np.ndarray) -> None:
    domain = ApplicabilityDomain(n_components=3).fit(spectra)
    centred = spectra - spectra.mean(axis=0)
    U, s, Vt = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ Vt[:3].T
    t2 = np.sum(scores**2 / (s[:3] ** 2 / (len(spectra) - 1)), axis=1)
    q = np.sum((centred - scores @ Vt[:3]) ** 2, axis=1)
    np.testing.assert_allclose(domain.statistics(spectra), np.column_stack([t2, q]))


def test_components_by_variance(spectra: np.ndarray) -> None:
    assert ApplicabilityDomain(0.9).fit(spectra).n_components_ <= 3
    assert ApplicabilityDomain().fit(spectra).n_components_ >= 1


def test_dataset_input(spectra: np.ndarray) -> None:
    dataset = Dataset(spectra, np.zeros(len(spectra)))
    domain = ApplicabilityDomain(3).fit(dataset)
    np.testing.assert_array_equal(domain.predict(dataset), domain.predict(spectra))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"confidence": 1.0}, "confidence"),
        ({"confidence": 95}, "confidence"),
        ({"statistic": "T2"}, "DomainStatistic member"),
        ({"q_limit_method": "BOX"}, "QLimit member"),
        ({"n_components": 500}, "n_components"),
        ({"n_components": 0}, "n_components"),
        ({"n_components": 1.5}, "n_components"),
    ],
)
def test_invalid_parameters(spectra: np.ndarray, kwargs: dict, match: str) -> None:
    with pytest.raises(SpectraXAIValueError, match=match):
        ApplicabilityDomain(**kwargs).fit(spectra)


def test_feature_mismatch(spectra: np.ndarray) -> None:
    domain = ApplicabilityDomain(3).fit(spectra)
    with pytest.raises(ValueError, match="features"):
        domain.predict(spectra[:, :10])


_NO_TRAINING_OUTLIERS = (
    "assumes a detector flags part of its training data (contamination); a "
    "domain at 99% confidence may flag none"
)


@parametrize_with_checks(
    [ApplicabilityDomain(n_components=1)],
    expected_failed_checks=lambda _: {
        "check_outliers_fit_predict": _NO_TRAINING_OUTLIERS,
        "check_outliers_train": _NO_TRAINING_OUTLIERS,
    },
)
def test_sklearn_conventions(estimator: Any, check: Callable[[Any], None]) -> None:
    check(estimator)


# -- plot -------------------------------------------------------------------------


def test_plot(spectra: np.ndarray) -> None:
    domain = ApplicabilityDomain(3).fit(spectra)
    fig, (other, ax) = plt.subplots(1, 2)
    new = np.vstack([spectra[:5], 4 * spectra[:5]])
    assert domain.plot(new, ax=ax, s=5) is ax
    assert not other.has_data()
    labels = [t.get_text() for t in cast(Any, ax.get_legend()).get_texts()]
    assert labels == ["training spectra", "inside", "outside"]
    assert ax.get_xscale() == "log"
    assert len(ax.lines) == 2  # both limits
    ax = domain.set_params(statistic=DomainStatistic.T2).plot(log=False)
    assert len(ax.lines) == 1 and ax.get_xscale() == "linear"

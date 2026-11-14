# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Grünwald–Letnikov fractional-order derivatives (`SpectralPreprocessing.FD`)."""

import numpy as np
import pytest
from scipy.special import gamma
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.models import Model, StandardModel
from spectraxai.preprocessing import SpectralPreprocessor
from spectraxai.spectra import Spectra
from spectraxai.spectra import SpectralPreprocessing as SP

RNG = np.random.default_rng(0)
X = RNG.random((4, 30))


def test_order_zero_returns_the_spectra() -> None:
    np.testing.assert_allclose(Spectra(X).fd(0).X, X)


@pytest.mark.parametrize("order", [1, 2])
def test_integer_orders_are_backward_differences(order: int) -> None:
    out = Spectra(X).fd(order).X
    np.testing.assert_allclose(out[:, order:], np.diff(X, n=order))


def test_coefficients_follow_the_binomial_series() -> None:
    from math import comb

    from spectraxai.spectra import _gl_weights

    np.testing.assert_allclose(
        _gl_weights(3, 6), [(-1) ** m * comb(3, m) for m in range(6)]
    )
    w = _gl_weights(0.5, 4)
    np.testing.assert_allclose(w, [1, -0.5, -0.125, -0.0625])


@pytest.mark.parametrize("order", [0.3, 0.5, 1.2, 1.7])
def test_matches_closed_form_for_power_functions(order: float) -> None:
    # D^v x^p = Γ(p+1) / Γ(p+1-v) x^(p-v) for the lower terminal at 0
    n = 4001
    x = np.linspace(0, 1, n)
    h = x[1] - x[0]
    for p in (1, 2):
        out = Spectra(x**p).fd(order, h=h).X.ravel()[n // 2 :]
        half = x[n // 2 :]
        exact = gamma(p + 1) / gamma(p + 1 - order) * half ** (p - order)
        np.testing.assert_allclose(out, exact, rtol=2e-3)


def test_orders_compose_like_derivatives() -> None:
    # With full memory the GL operators compose exactly: D^a D^b = D^(a+b)
    twice = Spectra(X).fd(0.5).fd(0.7).X
    np.testing.assert_allclose(twice, Spectra(X).fd(1.2).X, atol=1e-12)


def test_step_scales_the_derivative() -> None:
    np.testing.assert_allclose(
        Spectra(X).fd(0.5, h=0.25).X, Spectra(X).fd(0.5).X / 0.25**0.5
    )


def test_truncated_memory() -> None:
    out = Spectra(X).fd(0.5, n_terms=3).X
    w = [1, -0.5, -0.125]
    expected = w[0] * X[:, 5] + w[1] * X[:, 4] + w[2] * X[:, 3]
    np.testing.assert_allclose(out[:, 5], expected)
    # more terms than bands is the same as full memory
    np.testing.assert_allclose(Spectra(X).fd(0.5, n_terms=1000).X, Spectra(X).fd(0.5).X)


def test_fractional_orders_lie_between_integer_ones() -> None:
    # D^v x^2 at x = 1 is 2 / Γ(3 - v): it grows smoothly from 1 (v = 0) to 2 (v = 1)
    x = np.linspace(0, 1, 2001) ** 2
    ends = [Spectra(x).fd(v, h=1 / 2000).X[0, -1] for v in np.arange(0, 1.01, 0.25)]
    assert np.all(np.diff(ends) > 0)
    assert ends[0] == pytest.approx(1) and ends[-1] == pytest.approx(2, rel=1e-3)


def test_keeps_wavelengths_and_shape() -> None:
    wvs = np.linspace(400, 2500, X.shape[1])
    out = Spectra(X, wavelengths=wvs).fd(0.5)
    assert out.X.shape == X.shape
    np.testing.assert_array_equal(out.wavelengths, wvs)


@pytest.mark.parametrize("order", [-0.1, 2.5, "half", None, True, [0.5]])
def test_invalid_orders_raise(order: object) -> None:
    with pytest.raises(SpectraXAIValueError):
        Spectra(X).fd(order)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs", [{"h": 0}, {"h": -1}, {"n_terms": 0}, {"n_terms": 1.5}]
)
def test_invalid_options_raise(kwargs: dict) -> None:
    with pytest.raises(SpectraXAIValueError):
        Spectra(X).fd(0.5, **kwargs)


def test_preprocess_requires_an_order() -> None:
    with pytest.raises(SpectraXAIValueError, match="order"):
        Spectra(X).preprocess(SP.FD)


def test_preprocess_sequence() -> None:
    seq = [SP.ABS, (SP.FD, {"order": 0.5})]
    out = Spectra(X + 0.1).preprocess(seq).X
    np.testing.assert_allclose(out, Spectra(-np.log10(X + 0.1)).fd(0.5).X)


def test_round_trips_through_strings() -> None:
    seq = [
        SP.ABS,
        (SP.SG0, {"window_length": 5, "polyorder": 2}),
        (SP.FD, {"order": 1.5, "n_terms": 20}),
    ]
    assert SP.init_class(str(seq)) == seq
    assert SP.init_class("(FD, {'order': 0.5})") == (SP.FD, {"order": 0.5})
    assert not SP.FD.is_stateful


def test_dataset_preprocess(synthetic: Dataset) -> None:
    out = synthetic.preprocess((SP.FD, {"order": 0.5}))
    np.testing.assert_allclose(out.X, Spectra(synthetic.X).fd(0.5).X)


def test_sklearn_transformer_and_pipeline(synthetic: Dataset) -> None:
    pre = SpectralPreprocessor(method=[SP.ABS, (SP.FD, {"order": 0.75})])
    out = pre.fit_transform(synthetic.X)
    np.testing.assert_allclose(
        out, Spectra(synthetic.X).preprocess([SP.ABS, (SP.FD, {"order": 0.75})]).X
    )
    pipe = make_pipeline(clone(pre), PLSRegression(n_components=3))
    pipe.fit(synthetic.X, synthetic.Y)
    assert pipe.predict(synthetic.X).shape == (synthetic.n_samples, 1)


def test_fit_and_predict(synthetic: Dataset) -> None:
    pls = StandardModel(Model.PLS, init_hyperparameters={"n_components": 5})
    res = pls.fit_and_predict_multiple(
        synthetic,
        [[SP.ABS, (SP.FD, {"order": v})] for v in (0.5, 1.0)],
        idx_trn=np.arange(60),
    )
    assert len(res) == 2
    assert np.all(np.isfinite(res["RMSE"]))
    assert SP.init_class(res.loc[0, "pre_process"]) == [SP.ABS, (SP.FD, {"order": 0.5})]

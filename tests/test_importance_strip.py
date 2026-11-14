# SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""The importance strip plot (`plot_importance_strip`)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh

from spectraxai.dataset import Dataset
from spectraxai.exceptions import SpectraXAIValueError
from spectraxai.explain import FeatureRanking, PostHocAnalysis, PreHocAnalysis
from spectraxai.models import Model, StandardModel
from spectraxai.selection import SelectKBest


def _mesh(ax: Axes) -> np.ma.MaskedArray:
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    return np.ma.asarray(meshes[0].get_array()).reshape(
        len(ax.get_yticklabels()) or 1, -1
    )


def _line_positions(ax: Axes) -> list[np.ndarray]:
    """The x position of the lines drawn in each row, top row first."""
    rows: dict[int, list[float]] = {}
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            for seg in coll.get_segments():
                rows.setdefault(int(np.floor(seg[0, 1])), []).append(seg[0, 0])
    return [np.sort(rows[r]) for r in sorted(rows)]


@pytest.fixture
def importance(synthetic: Dataset) -> np.ndarray:
    return np.random.default_rng(0).normal(size=synthetic.n_features)


def test_continuous_strip_shades_by_absolute_importance(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    _, ax = plt.subplots()
    out = PostHocAnalysis(synthetic).plot_importance_strip(importance, ax=ax)
    assert out is ax
    shade = _mesh(ax)[0]
    magnitude = np.abs(importance)
    assert shade.min() == 0 and shade.max() == 1
    np.testing.assert_allclose(
        shade, (magnitude - magnitude.min()) / np.ptp(magnitude), atol=1e-12
    )
    assert np.argmax(shade) == np.argmax(magnitude)
    assert ax.get_yticks().size == 0  # a single unlabelled row
    ticks = ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks()
    assert not any(tick.gridline.get_visible() for tick in ticks), "no grid"
    assert ax.get_xlabel() == synthetic.spectral_label


def test_band_edges_cover_the_spectrum(synthetic: Dataset) -> None:
    ax = PreHocAnalysis(synthetic).plot_importance_strip(FeatureRanking.CORR)
    wavelengths = synthetic.wavelengths
    assert wavelengths is not None
    low, high = sorted(ax.get_xlim())
    assert low < wavelengths.min() and high > wavelengths.max()


def test_dict_stacks_labelled_rows_top_to_bottom(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    ax = PreHocAnalysis(synthetic).plot_importance_strip(
        {"random": importance, "corr": FeatureRanking.CORR}
    )
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["random", "corr"]
    assert ax.get_ylim() == (2, 0)
    shade = _mesh(ax)
    assert shade.shape == (2, synthetic.n_features)
    assert np.argmax(shade[0]) == np.argmax(np.abs(importance))


def test_per_row_and_shared_scales(synthetic: Dataset, importance: np.ndarray) -> None:
    rows = {"full": importance, "half": importance / 2}
    post = PostHocAnalysis(synthetic)
    per_row = _mesh(post.plot_importance_strip(rows))
    np.testing.assert_allclose(per_row[0], per_row[1])
    shared = _mesh(post.plot_importance_strip(rows, shared_scale=True))
    assert shared[0].max() == 1
    assert shared[1].max() < 0.6


def test_multi_output_rankings_get_one_row_per_output(synthetic2d: Dataset) -> None:
    # a ranking on its own is labelled with its name, a dict entry with its key
    ax = PreHocAnalysis(synthetic2d).plot_importance_strip(FeatureRanking.CORR)
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        f"CORR: {name}" for name in synthetic2d.Y_names
    ]
    ax = PreHocAnalysis(synthetic2d).plot_importance_strip(
        {"F": FeatureRanking.F_STATISTIC}
    )
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        f"F: {name}" for name in synthetic2d.Y_names
    ]
    values = np.random.default_rng(0).random((2, synthetic2d.n_features))
    ax = PostHocAnalysis(synthetic2d).plot_importance_strip(values)
    assert [t.get_text() for t in ax.get_yticklabels()] == list(synthetic2d.Y_names)


def test_column_oriented_arrays_are_accepted(synthetic: Dataset) -> None:
    values = np.random.default_rng(1).random((synthetic.n_features, 3))
    ax = PostHocAnalysis(synthetic).plot_importance_strip(values)
    assert _mesh(ax).shape == (3, synthetic.n_features)


def test_fitted_model(synthetic: Dataset) -> None:
    model = StandardModel(Model.PLS, init_hyperparameters={"n_components": 3})
    model.fit(synthetic)
    vip, _ = model.feature_importance()
    assert vip is not None
    ax = PostHocAnalysis(synthetic).plot_importance_strip(model)
    assert np.argmax(_mesh(ax)[0]) == np.argmax(vip)


def test_top_k_draws_lines_for_the_most_important_bands(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    ax = PostHocAnalysis(synthetic).plot_importance_strip(importance, top_k=5)
    assert not [c for c in ax.collections if isinstance(c, QuadMesh)]
    (positions,) = _line_positions(ax)
    wavelengths = synthetic.wavelengths
    assert wavelengths is not None
    expected = np.sort(wavelengths[np.argsort(np.abs(importance))[-5:]])
    np.testing.assert_allclose(positions, expected)


def test_line_darkness_follows_importance(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    ax = PostHocAnalysis(synthetic).plot_importance_strip(importance, top_k=3)
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    darkness = 1 - np.asarray(lines.get_colors())[:, :3].mean(axis=1)
    x = np.array([seg[0, 0] for seg in lines.get_segments()])
    wavelengths = synthetic.wavelengths
    assert wavelengths is not None
    magnitude = np.abs(importance)[
        [int(np.flatnonzero(wavelengths == v)[0]) for v in x]
    ]
    assert np.all(np.diff(darkness[np.argsort(magnitude)]) > 0)


def test_threshold(synthetic: Dataset, importance: np.ndarray) -> None:
    ax = PostHocAnalysis(synthetic).plot_importance_strip(importance, threshold=1.0)
    (positions,) = _line_positions(ax)
    assert positions.size == int(np.sum(np.abs(importance) >= 1.0))


def test_selectors_and_masks_draw_their_bands(synthetic: Dataset) -> None:
    selector = SelectKBest(k=6).fit(synthetic)
    mask = np.zeros(synthetic.n_features, dtype=bool)
    mask[[0, 10, 20]] = True
    ax = PreHocAnalysis(synthetic).plot_importance_strip(
        {"selector": selector, "mask": mask}
    )
    sel_rows, mask_rows = _line_positions(ax)
    wavelengths = synthetic.wavelengths
    assert wavelengths is not None
    np.testing.assert_allclose(sel_rows, np.sort(wavelengths[selector.get_support()]))
    np.testing.assert_allclose(mask_rows, np.sort(wavelengths[mask]))


def test_excluded_bands_are_left_blank(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    post = PostHocAnalysis(synthetic)
    shade = _mesh(post.plot_importance_strip(importance, excluded_bands=[(1300, 1500)]))
    wavelengths = synthetic.wavelengths
    assert wavelengths is not None
    inside = (wavelengths >= 1300) & (wavelengths <= 1500)
    mask = np.ma.getmaskarray(shade)[0]
    assert mask[inside].all() and not mask[~inside].any()
    ax = post.plot_importance_strip(
        importance, top_k=synthetic.n_features, excluded_bands=[(1500, 1300)]
    )
    (positions,) = _line_positions(ax)
    assert positions.size == int(np.sum(~inside))


def test_with_spectrum_shares_the_spectral_axis(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    ax = PostHocAnalysis(synthetic).plot_importance_strip(
        importance, with_spectrum=True, colorbar=True
    )
    spectrum_ax = ax.figure.axes[0]
    assert spectrum_ax is not ax and spectrum_ax.has_data()
    assert spectrum_ax.get_shared_x_axes().joined(spectrum_ax, ax)
    # the colorbar spans both panels, which therefore stay aligned
    assert spectrum_ax.get_position().x1 == pytest.approx(ax.get_position().x1)


def test_colorbar(synthetic: Dataset, importance: np.ndarray) -> None:
    ax = PostHocAnalysis(synthetic).plot_importance_strip(importance, colorbar=True)
    assert len(ax.figure.axes) == 2


def test_wavenumbers_are_drawn_high_to_low(synthetic: Dataset) -> None:
    wavenumbers = np.linspace(4000, 600, synthetic.n_features)
    mir = Dataset(synthetic.X, synthetic.Y, wavenumbers, ["y"], spectral_unit="cm-1")
    importance = np.zeros(mir.n_features)
    importance[0] = 1  # the 4000 cm-1 band
    for top_k in (None, 1):
        ax = PreHocAnalysis(mir).plot_importance_strip(importance, top_k=top_k)
        assert ax.xaxis_inverted()
        assert ax.get_xlabel() == mir.spectral_label
    ax = PreHocAnalysis(mir).plot_importance_strip(importance)
    assert np.argmax(_mesh(ax)[0]) == mir.n_features - 1  # sorted ascending


def test_kwargs_reach_the_artists(synthetic: Dataset, importance: np.ndarray) -> None:
    post = PostHocAnalysis(synthetic)
    ax = post.plot_importance_strip(importance, cmap="viridis", alpha=0.5)
    (mesh,) = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert mesh.get_cmap().name == "viridis" and mesh.get_alpha() == 0.5
    ax = post.plot_importance_strip(importance, top_k=2, lw=4)
    (lines,) = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert np.asarray(lines.get_linewidth()).ravel()[0] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"importance": np.ones(3)},
        {"importance": {}},
        {"importance": np.ones(42), "top_k": 0},
        {"importance": np.ones(42), "top_k": 1.5},
    ],
)
def test_invalid_arguments(synthetic: Dataset, kwargs: dict) -> None:
    with pytest.raises(SpectraXAIValueError):
        PostHocAnalysis(synthetic).plot_importance_strip(**kwargs)


def test_with_spectrum_needs_its_own_figure(
    synthetic: Dataset, importance: np.ndarray
) -> None:
    _, ax = plt.subplots()
    with pytest.raises(SpectraXAIValueError, match="with_spectrum"):
        PostHocAnalysis(synthetic).plot_importance_strip(
            importance, ax=ax, with_spectrum=True
        )


def test_rankings_need_prehoc(synthetic: Dataset) -> None:
    with pytest.raises(SpectraXAIValueError, match="PreHocAnalysis"):
        PostHocAnalysis(synthetic).plot_importance_strip(FeatureRanking.CORR)

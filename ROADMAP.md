<!--
SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# SpectraXAI 1.3.0 roadmap

Goal: smarter model building and more trustworthy predictions: Bayesian and
random hyperparameter search, joint search over pre-processing and model,
fractional-order derivatives, local PLS, applicability-domain checks, conformal
prediction intervals, and an importance "strip" plot. **Code that works on 1.2.0
keeps working unchanged.**

## Status

Phase 1 (fractional-order derivatives) is implemented; phases 2–7 to follow.

## Ground rules

| # | Decision |
|---|----------|
| D1 | Release as **1.3.0** (new features; 1.2.x patches are for fixes only). The 1.2.0 ground rules still hold: keyword-only additions with defaults, `DeprecationWarning` for anything retired, `random_state` for every source of randomness, `SpectraXAIValueError` for invalid arguments, regression only. |
| D2 | Optuna is an **optional extra** (`spectraxai[optuna]`). Grid and random search, and every other feature, need no new dependency. Using `search="bayes"` without Optuna raises an error saying how to install it. scikit-optimize is not used (archived, broken with current NumPy). |
| D3 | Every phase ships with **tests and an example** (script or notebook section), runs in CI, and updates the CHANGELOG and docs. |
| D4 | One branch and commit per phase, as for 1.2.0; docs regenerated at the end of each phase. |

## Phase 1: Fractional-order derivatives

- New `SpectralPreprocessing.FD`, used as `(SP.FD, {"order": 0.5})`; stateless, so it also works in `SpectralPreprocessor` and `fit_and_predict`.
- Grünwald–Letnikov definition (the one used in soil spectroscopy, e.g. Hong et al. 2018), implemented in NumPy: coefficients by the recurrence w₀ = 1, wₘ = wₘ₋₁·(m − 1 − v)/m, applied as a backward convolution along the spectral axis.
- Options: `order` (required, real, 0 ≤ order ≤ 2), `h=1` (step; the default counts bands, as in the literature and like the SG derivatives here), `n_terms=None` (full memory by default; an integer truncates it).
- No built-in smoothing: chain it, e.g. `[SP.ABS, (SP.SG0, {...}), (SP.FD, {"order": 1.5})]`.
- As with SG1, the sign follows the stored column order; documented for MIR data stored high to low.

**Tests:** order 0 returns the spectrum; order 1 equals the first backward difference and order 2 the second; known closed forms (fractional derivative of a power function); `n_terms` truncation; `h` scaling (result scales by h^(−v)); `init_class` round-trip of `str((SP.FD, {...}))`; works in `fit_and_predict`, `SpectralPreprocessor` and a `Pipeline`; invalid orders raise.
**Example:** `examples/fractional_derivatives.py`: PLS cross-validated RMSE for orders 0–2 in steps of 0.25, plotted against order, plus the spectra at a few orders.

## Phase 2: Importance strip plot

- New `plot_importance_strip` on `PreHocAnalysis` and `PostHocAnalysis`: a thin band along the spectral axis, darker where a wavelength is more important (like an absorption spectrum).
- `importance`: an array, a fitted model (its own importance, as in `plot_importance_spectrum`), a fitted selector (its selected bands), or a dict `{label: importance}` for one stacked row per method or output.
- Continuous mode (default) draws with `pcolormesh` over band edges, so dense spectra have no gaps or overlaps. Line mode (`top_k=` or `threshold=`, or automatically for a selector) draws only the chosen wavelengths as dark lines on a light strip, darkness still showing importance.
- Darkness: |importance| min–max scaled per row (`shared_scale=True` for one scale across rows); `cmap="Greys"`, optional `colorbar=False`.
- `with_spectrum=False`: when True on a figure the library creates, the mean spectrum is drawn above the strip(s), sharing the x axis. With a user `ax`, only the strip is drawn.
- Follows the plot contract (`ax=`, `**kwargs`, no global state), the wavenumber axis direction and `excluded_bands`.

**Tests:** artists drawn on the passed `ax`; one row per dict entry with labels; darkness ordering matches importance; line mode draws exactly `top_k` lines / the selector's bands; `shared_scale`; MIR axis reversed; excluded bands blank; `with_spectrum` creates two axes sharing x.
**Example:** a section in `modelInterpretability.ipynb` comparing VIP, permutation, SHAP and correlation as stacked strips, and a selector in line mode.

## Phase 3: Random and Bayesian hyperparameter search

- `StandardModel(..., *, search="grid", n_iter=50, search_space=None, sampler=None)`:
  - `"grid"`: unchanged behaviour (`GridSearchCV` over `grid_search_hyperparameters`);
  - `"random"`: `RandomizedSearchCV`, no new dependency;
  - `"bayes"`: Optuna (D2), TPE sampler by default, any Optuna sampler via `sampler=` (e.g. `GPSampler`).
- Default search spaces per model, next to `_DEFAULT_GRIDS`, covering the same ranges (log scale where the grid is log-spaced): e.g. SVR `C` log-uniform 2⁻²–2⁹, `epsilon` 0.01–0.5, `gamma` from `sigest`'s range at fit time; PLS integer `n_components` 1 to min(100, features − 1); XGBoost `learning_rate` log 0.01–0.3, `max_depth` 2–10. GPR keeps tuning its own kernel.
- Our own small search class (`spectraxai.search`) instead of the experimental `OptunaSearchCV`: each trial is scored by cross-validation; trials clearly behind are stopped after each fold (median pruning).
- `search_space`: a dict of Optuna-style distributions or simple specs (`("log", 1e-3, 1e3)`, `("int", 1, 30)`, a list of choices). A list in `grid_search_hyperparameters` is treated as choices for random and Bayesian search.
- Results filled the same way for every strategy: `best_hyperparameters`, `best_score`, `search_space_` (`param_grid_` kept as an alias), and a new `search_results_` table of all trials.

**Tests:** grid results unchanged from 1.2.0; random and Bayesian search are reproducible with `random_state`; best score at least as good as a coarse grid on synthetic data; default spaces exist for every tunable model; pruning stops trials; clear error without Optuna (simulated); `clone` and `get_params` round-trip the new parameters; sklearn estimator checks still pass.
**Example:** `examples/hyperparameter_search.py`: grid vs random vs Bayesian search for SVR on the bundled SSL, comparing RMSE, number of fits and time.
**CI:** a test job with the `optuna` extra installed, and one without it.

## Phase 4: Joint pre-processing and model search

- New `spectraxai.search.PreprocessingSearch(models=[...], preprocessings=[...] or a space, *, search="bayes", n_iter=100, cv=5, random_state=None)`: one study that searches the pre-processing chain, its continuous parameters (FD order, SG window and polynomial order), the model, and its hyperparameters.
- Pre-processing in every trial is learned on the training folds only, as `fit_and_predict` does.
- Returns a fitted best pipeline (`best_estimator_`, a scikit-learn `Pipeline` of `SpectralPreprocessor` and model) and a `results_` table of every trial; `search="grid"` enumerates the given list, which replaces hand-written loops over `fit_and_predict_multiple`.
- Works on a `Dataset` or on X, y.

**Tests:** finds the planted best pre-processing on a synthetic dataset; no leakage (stateful MSC fitted on training folds only); reproducible with `random_state`; the best pipeline predicts on new data; grid mode equals `fit_and_predict_multiple` for the same list.
**Example:** `examples/automl_search.py` on the bundled SSL, and a section in `modelExamples.ipynb`.

## Phase 5: Local PLS (memory-based learning)

- New `Model.LOCAL_PLS`, backed by a scikit-learn `LocalPLSRegressor(k=100, n_components=10, *, metric="pca-mahalanobis", n_pca_components=None, weights="uniform")`: for each spectrum to predict, a PLS model is fitted on its `k` nearest training spectra (the memory-based learning approach of the R package resemble).
- Distances in PCA-score space (Mahalanobis) or by correlation; `n_components` capped at `k − 1`.
- Usable through `StandardModel` (grid, random or Bayesian search over `k` and `n_components`) and in `PreprocessingSearch`.
- Feature importance: the mean absolute VIP of the local models over the predicted samples.

**Tests:** with `k` equal to the training size it matches global PLS; outperforms global PLS on a synthetic dataset with locally different relationships; deterministic; sklearn estimator checks pass; works through `StandardModel` and with `fit_and_predict`.
**Example:** a section in `modelExamples.ipynb` comparing local and global PLS.

## Phase 6: Applicability domain and prediction intervals

- New `spectraxai.diagnostics.ApplicabilityDomain(n_components=None, *, confidence=0.99)`: PCA on the training spectra, Hotelling T² and Q residual statistics with critical limits; `predict` flags spectra outside the domain, `score_samples` returns both statistics, and `plot` draws the T²–Q chart with the limits (plot contract).
- New `spectraxai.uncertainty.ConformalRegressor(model, *, method="split", alpha=0.1, calibration_size=0.2, random_state=None)`: split-conformal and CV+ prediction intervals around any `StandardModel` or scikit-learn regressor; `predict(X, return_interval=True)`. No new dependency.
- `fit_and_predict(..., *, return_intervals=False)` adds interval width and empirical coverage to the results.

**Tests:** T² and Q limits match reference formulas; training spectra are mostly inside the domain at the chosen confidence; shifted or noisy spectra are flagged; conformal coverage on synthetic data within tolerance of 1 − alpha; reproducible; works with every model.
**Example:** `examples/prediction_reliability.py` (flag unusual spectra and report predictions with intervals) and a section in `mirSpectra.ipynb`.

## Phase 7: Wrap-up

- Notebooks re-run with the new features; README "What's new in 1.3.0"; CHANGELOG; docs rebuilt.
- CI: the examples job installs the `optuna` extra.

## Out of scope for 1.3.0

Calibration transfer (DS/PDS), spectral augmentation, more representative-sampling methods (Duplex, Puchwein) and 1D-CNN models were considered and rejected.

# SpectraXAI 1.2.0 roadmap (released)

Goal: make `spectraxai` more solid — consistent, sklearn-compatible interfaces,
robust and composable plots, real test coverage, feature selection, more models —
**without breaking any code that works on v1.1.0**.

## Status

Phases 1–6 are implemented (see [CHANGELOG.md](CHANGELOG.md)). Still open: the
`spectraxai[pycaret]` extra and its CI job, to be added once PyCaret 4.0 is
released as stable (the example notebook targets the 4.0 API and was run with
4.0.0a8).

## Ground rules

| # | Decision |
|---|----------|
| D1 | Release as **1.2.0**. Every public name and signature from v1.1.0 keeps working. New parameters are keyword-only with defaults. Anything to retire emits `DeprecationWarning` and stays until 2.0. |
| D2 | Bug fixes that change numeric results are allowed and listed in `CHANGELOG.md`. Every source of randomness gets `random_state=None` (default preserves current behaviour). |
| D3 | Validation errors raise `SpectraXAIValueError(ValueError, AssertionError)` so existing `except AssertionError` still catches them. |
| D4 | Scope stays **regression only**; fix docstrings that mention classification. Keep interfaces open for classification later. |
| D5 | Move packaging from `setup.py` to `pyproject.toml`, same dependencies (xgboost, cubist, sage remain required). New heavy deps are optional extras only. |
| D6 | Keep Azure Pipelines (superseded: CI moved to GitHub Actions after the fork). Keep committing `docs/` and regenerate it at the end of each phase. |
| D7 | Deliver as six phased PRs, each releasable on its own, in the order below. |

## Phase 1 — Test infrastructure and bug fixes

**Testing**
- Switch the runner to pytest (existing `unittest` classes keep running).
- Add a small, seeded synthetic spectral fixture (`tests/conftest.py`) for fast unit tests; keep `load_GR_SSL` for a few integration tests.
- Force the `Agg` backend; close figures after each test.
- Plot tests assert on drawn artists (lines, bars, NaN gaps) and that a passed `ax` is the one drawn on. No image-comparison tests.
- CI: add pyright (basic) and flake8 jobs; coverage floor ≈ 85 %.
- CI: add an `nbmake` job running `examples/*.ipynb` and `examples/*.py` at coarse subsampling.

**Bugs to fix (one regression test each)**
- `StandardModel.fit` writes SVR `gamma` and PLS `n_components` into `self.grid_search_hyperparameters`, so later fits (folds, `fit_and_predict_multiple`) reuse the grid of the first dataset/pre-processing. Compute per-fit grids locally.
- Off-by-one bounds checks (`idx <= n_samples` → `<`) in `Dataset.subset` and `Dataset.train_test_split_explicit`.
- Unseeded randomness: RF, XGBoost, `KFold(shuffle=True)`, `sigest`, `estimate_epsilon`, random split → `random_state`.
- `PreHocAnalysis._mi` / `_f_statistic` divide by `max` without guarding zero.
- `bar_plot_permutation_importance` y-label says "Mean decrease in impurity"; should be "Mean decrease in score".
- XGBoost rows in `fit_and_predict` lack hyperparameters and `feature_importance`.
- `DatasetSplit.STRATIFIED` runs `StratifiedKFold` on continuous Y. Stratify on quantile bins of Y (`n_bins=10` default).
- `SpectralPreprocessing.init_class` uses fragile string replacement. Replace with a proper parser that round-trips everything `str(preprocess)` produces (property-based test); same signature and output.
- Replace validation `AssertionError`s with `SpectraXAIValueError` (D3).

**Done when:** all tests pass on 3.13–3.15, lint + type-check jobs green, CHANGELOG lists every behavioural fix.

## Phase 2 — Plot contract

Every plotting function:
- accepts `ax=` (or `axes=` for multi-panel) and returns what it drew on;
- never mutates global state — no `plt.style.use`, `plt.axis`, or `tight_layout` on a figure it didn't create;
- forwards `**kwargs` to the main matplotlib call;
- keeps today's default `figsize`/`dpi` only when it creates the figure itself.

Styling moves from `_Explain.__init__` into an opt-in `spectraxai.plotting.style()` context manager, applied automatically only when the library creates the figure.

Functions to change: `PreHocAnalysis.correlogram` (add `axes=`), `PreHocAnalysis.mean_spectrum_by_range` (add `axes=`), `PostHocAnalysis.circular_bar_plot_importance` (drop `plt.axis("off")` on the current axes; require/create a polar `ax`), `utils.scatter_plot` (no `plt.figure()`), plus audit of `bar_plot_*` and `mean_spectrum`.

**Done when:** every plot is tested drawing into a user-supplied axes inside a multi-panel figure, and `plt.rcParams` is unchanged after constructing `PreHocAnalysis`/`PostHocAnalysis`.

## Phase 3 — sklearn interface

**`StandardModel`**
- Becomes an sklearn estimator (`BaseEstimator`, `RegressorMixin`). `fit` accepts a `Dataset` *or* `(X, y)` (detected).
- `get_params` covers `model`, `init_hyperparameters`, `grid_search_hyperparameters`, `estimator`, `cv`, `random_state`, `n_jobs` (default `-1`, as today).
- `best_model` stays the bare estimator; new `pipeline_` holds scaler + estimator; `predict` goes through `pipeline_`.
- `Model.CUSTOM` + `estimator=` lets users bring any sklearn regressor; `model` stays the first positional argument.
- `PostHocAnalysis` methods accept a `StandardModel` directly — fixes SVR receiving unscaled X in permutation importance and SAGE.

**Pre-processing and data**
- New `SpectralPreprocessor` sklearn transformer built on `SpectralPreprocessingSequence`.
- New `SpectralPreprocessing` members: `MSC`, `DETREND`, `MEAN_CENTER`. MSC learns its reference at `fit`; `fit_and_predict` uses the transformer so the reference never sees test data.
- `Dataset.wavelengths` property (floats from `X_names`, or `None`) and optional `wavelengths=` constructor argument; `X_names` unchanged.
- `continuum_removal` / `Spectra.cr` accept `wavelengths=`.
- `Dataset.to_dataframe()`.

**Unified results schema** — every `fit_and_predict` row has `model_name`, `best_hyperparameters` (dict), `feature_importance` (native or `None`), `importance_method` (`"VIP"`, `"impurity"`, `"coef"`, `None`), `random_state`. Existing flat columns (`C`, `n_components`, `SVs`, …) and the `model` column from `get_model=True` stay. New `return_permutation_importance=False` computes test-set importance for models without native importance.

**Done when:** `sklearn.utils.estimator_checks` passes for the relevant checks, `cross_val_score(StandardModel(...), X, y)` works, and all v1.1.0 usage in the tests and examples runs unchanged.

## Phase 4 — Feature selection (+ CLHS)

New module `spectraxai.selection`:
- `RFE` / `RFECV` wrapping sklearn's.
- VIP filtering for PLS (default threshold 1).
- `SelectKBest` driven by the existing `FeatureRanking` methods (correlation, mutual information, F-statistic).

Each selector is an sklearn transformer **and** offers `selector.fit(dataset).transform(dataset) -> Dataset`, keeping `X_names`/`wavelengths` in sync. Importance getter by model: VIP for PLS, `feature_importances_` for RF/XGBoost/Cubist, coefficients for linear models, permutation importance fallback for SVR and others.

New plot: `plot_selected_bands(ax=...)` shading kept wavelengths over the mean spectrum.

Implement `DatasetSplit.CLHS` (conditioned Latin hypercube sampling, pure numpy annealing, seeded).

Later (out of scope): iPLS, CARS, SPA, GA.

## Phase 5 — Models and explanations

- New `Model` members (sklearn only, no new deps): `RIDGE`, `LASSO`, `ELASTIC_NET`, `KNN`, `GPR`, `MLP`, each with a default grid and importance where available.
- New explanation plots (all following the Phase 2 contract): importance overlaid on the mean spectrum, `plot_pls_coefficients`, and a SAGE plot helper.
- Optional extra `spectraxai[shap]`: SHAP summary across wavelengths.

## Phase 6 — pycaret, docs, examples

- No pycaret dependency for now (stable 3.x doesn't support Python ≥ 3.12 well; 4.0 with 3.13 support is still alpha as of May 2026).
- Example notebook showing `SpectralPreprocessor` and selectors used via `pycaret.regression.setup(custom_pipeline=...)` and `Dataset.to_dataframe()`.
- When pycaret 4.0 is stable: add `spectraxai[pycaret]` extra and an optional, non-blocking CI job.
- Update notebooks and README to the new APIs (`ax=`, `random_state`, selectors); regenerate `docs/`.

## Out of scope for 1.2.0

Classification models, deep learning / 1D-CNNs, LightGBM/CatBoost, iPLS/CARS/SPA/GA, image-comparison plot tests.

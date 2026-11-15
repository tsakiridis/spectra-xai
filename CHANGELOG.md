<!--
SPDX-FileCopyrightText: 2026 Nikos Tsakiridis <tsakirin@auth.gr>

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to this project are documented here. The decisions behind
1.2.0 are in [ROADMAP.md](https://github.com/tsakiridis/spectra-xai/blob/main/ROADMAP.md).

## [Unreleased] — 1.3.0

### Added
- **Fractional-order derivatives**: `SpectralPreprocessing.FD` and `Spectra.fd`
  (Grünwald–Letnikov), e.g. `(SpectralPreprocessing.FD, {"order": 0.5})`. Any
  order from 0 to 2; optional `h` (band step, default 1) and `n_terms` (truncated
  memory). Stateless, so it works in `Dataset.preprocess`, `fit_and_predict`,
  `SpectralPreprocessor` and scikit-learn pipelines. New example
  `fractional_derivatives.py` chooses the order by cross-validation.
- **Importance strip plot**: `plot_importance_strip` on `PreHocAnalysis` and
  `PostHocAnalysis` draws importance as a strip along the spectrum, darker where
  a wavelength matters more. It takes an array, a fitted model, a selector or
  boolean mask, a `FeatureRanking` (pre-hoc), or a dict of these for stacked,
  labelled rows (one per method or output). Line mode (`top_k=`, `threshold=`,
  or any selector or mask) draws only the chosen wavelengths as dark lines.
  Options: `shared_scale`, `colorbar`, `with_spectrum` (mean spectrum above,
  sharing the axis), `excluded_bands`, `cmap`. Follows the plot contract and the
  wavenumber axis direction. New section in `modelInterpretability.ipynb`.

- **Random and Bayesian hyperparameter search**: `StandardModel(...,
  search="random" | "bayes", n_iter=50, search_space=None, sampler=None)`. Grid
  search stays the default and is unchanged. Every tunable model has a default
  search space covering its default grid (log scale where the grid is
  log-spaced; PLS `n_components` and SVR `gamma` derived from the data). Spaces
  use simple entries (`("log", lo, hi)`, `("int", lo, hi)`, lists of choices) or
  Optuna/SciPy distributions; a custom `grid_search_hyperparameters` is read as
  choices.
- `spectraxai.search.BayesianSearchCV`: Optuna-based search with
  cross-validation for any scikit-learn estimator, TPE by default (any Optuna
  sampler via `sampler=`), stopping unpromising trials after each fold (median
  pruning). Needs the new optional extra: `pip install spectraxai[optuna]`.
- `StandardModel.search_results_` (one row per candidate) and `search_space_`
  (`param_grid_` is kept as an alias).
- New example `hyperparameter_search.py`: grid vs random vs Bayesian search
  for SVR.
- CI: tests run with the `optuna` extra, plus a job without it.
- **Joint search of pre-processing and model**:
  `spectraxai.search.PreprocessingSearch(models, preprocessings, search="bayes")`
  searches the pre-processing, its options, the model and its hyperparameters
  together. Pre-processing options written as ranges (e.g.
  `(SP.FD, {"order": ("float", 0, 2)})` or an SG `window_length`) are tuned by
  Bayesian search; `"grid"` and `"random"` search the listed pre-processings
  with every model's grid or space. Every candidate is cross-validated as a
  pipeline (`SpectralPreprocessor` → scaler → model), so stateful steps never
  see the validation folds. Invalid option combinations fail single trials.
  Results: `best_estimator_` (a fitted pipeline), `best_params_`,
  `best_score_` and `results_` (one row per candidate). New example
  `automl_search.py` and a section in `modelExamples.ipynb`.
- `BayesianSearchCV.results_` (and so `StandardModel.search_results_` for
  Bayesian search) has a `rank_test_score` column, ranking complete trials only.
- **Local models (memory-based learning)**: `Model.LOCAL_PLS` and
  `Model.LOCAL_GPR`, backed by `spectraxai.local.LocalPLSRegressor(k=100,
  n_components=10)` and `LocalGPRRegressor(k=100, noise_variance=0.001)`. Each
  spectrum is predicted by a PLS model, or a linear-kernel Gaussian process as in
  resemble's `local_fit_gpr`, fitted on its `k` nearest training spectra
  (`dissimilarity=`, see below; Mahalanobis distance in principal-component
  space by default). Default grids and search spaces over `k` and `n_components` /
  `noise_variance`; local PLS reports the mean VIP of its local models as
  feature importance (`LocalPLSRegressor.vip(X)` for given spectra). New section
  in `modelExamples.ipynb`.
- **Spectral dissimilarities**: new module `spectraxai.dissimilarity`. The
  `Dissimilarity` enum names each measure: `EUCLIDEAN`; `PCA_EUCLIDEAN`,
  `PCA_MAHALANOBIS` and `PLS_MAHALANOBIS` (on principal-component or PLS
  scores); `CORRELATION` and `MOVING_CORRELATION`; the spectral angle `SAM`, the
  spectral information divergence `SID`, and their combinations `SID_SAM_TAN`
  and `SID_SAM_SIN`. `SpectralDissimilarity(method, n_components=None,
  window=11)` sets the options and computes matrices against reference spectra
  (`pairwise_dissimilarity` in one call). Local models take any of them as
  `dissimilarity=` (also searchable), or `Dissimilarity.PRECOMPUTED` with a
  matrix computed elsewhere, e.g. from other data than the spectra, passed to
  `predict(X, dissimilarity=D)`, or as a matrix over all samples to
  `fit_and_predict` / `fit_and_predict_multiple`.

### Changed
- `SpectralPreprocessingOptions` is typed `dict[str, Any]` (options may be floats).

## [1.2.0] — 2026-11-14

### Added
- `spectraxai.exceptions.SpectraXAIValueError`, raised for invalid arguments. It
  subclasses both `ValueError` and `AssertionError`, so existing
  `except AssertionError` / `except ValueError` handlers keep working.
- `random_state` keyword argument (default `None`, i.e. the previous behaviour) on
  `StandardModel`, `Dataset.train_test_split`, `PreHocAnalysis`,
  `PostHocAnalysis`, `utils.sigest` and `utils.estimate_epsilon`.
- `StandardModel.param_grid_`: the grid actually searched by the last `fit`.
- `Dataset.train_test_split(..., n_bins=10)` for stratifying continuous outputs.
- `SpectralPreprocessing.init_class` now parses everything `str()` produces for a
  pre-processing sequence (single members, members with options, nested lists,
  names or values), e.g. the `pre_process` column of `fit_and_predict`.

- **Plotting contract** (`spectraxai.plotting`). Every plot accepts axes to draw
  into, returns what it drew on, forwards extra keyword arguments to its main
  matplotlib call, and never changes global matplotlib state:
  - `PreHocAnalysis.correlogram(..., axes=)` and
    `PreHocAnalysis.mean_spectrum_by_range(..., axes=)` are new (keyword-only).
  - `**kwargs` on `PreHocAnalysis.bar_plot_importance`, `correlogram`,
    `mean_spectrum`, `mean_spectrum_by_range`, `PostHocAnalysis.bar_plot_importance`,
    `circular_bar_plot_importance`, `bar_plot_permutation_importance` and
    `utils.scatter_plot` (e.g. `color=`, `lw=`, `s=`, `label=`).
  - `spectraxai.plotting.style()` context manager to apply the library's look to
    your own figures.
  - Multi-output plots accept axes in any shape (e.g. from `plt.subplots(1, 2)`).

- **scikit-learn interface.** `StandardModel` is now a scikit-learn regressor
  (`BaseEstimator`, `RegressorMixin`): it works with `clone`, pipelines,
  `GridSearchCV`, `cross_val_score`, `permutation_importance` and SAGE.
  - `fit` also accepts `fit(X, y)` and returns `self`; `fit(dataset)` and
    `fit(dataset, cv=...)` work as before.
  - New constructor keywords: `estimator=`, `cv=` (default 5), `n_jobs=` (default
    -1).
  - `Model.CUSTOM` + `estimator=` trains any scikit-learn regressor (cloned, never
    modified), with or without a grid search.
  - New `pipeline_` (SVR scaler + `best_model`), used by `predict`; `predict`
    also accepts a `Dataset` and checks the number of features.
  - New `StandardModel.feature_importance()` returning `(importance, method)`.
  - `PostHocAnalysis.bar_plot_permutation_importance` and `sage_importance`
    accept a `StandardModel` directly. For SVR this is the correct way: passing
    `best_model` skips the input scaling.
- `spectraxai.preprocessing.SpectralPreprocessor`, a scikit-learn transformer for
  any `SpectralPreprocessingSequence`.
- New pre-processing: `SpectralPreprocessing.MSC`, `DETREND` and `MEAN_CENTER`
  (and `Spectra.msc`, `detrend`, `mean_center`). The stateful ones learn their
  reference from the training samples in `fit_and_predict` and in
  `SpectralPreprocessor`, and from the given spectra in `Spectra.preprocess`.
- `Dataset.wavelengths` (from numeric `X_names`, or the new `wavelengths=`
  constructor argument), used by continuum removal and the spectral plots;
  `Spectra(X, wavelengths=)` and `Spectra.cr(wavelengths=)`.
- `Dataset.to_dataframe()`.
- **Unified results.** Every `fit_and_predict` row now also has `model_name`,
  `best_hyperparameters` (dict), `feature_importance` (array or None, now also
  for SVR), `importance_method` (`"VIP"`, `"impurity"`, `"usage"`, `"gain"`,
  `"coef"`, `"native"`, `"permutation"` or None) and `random_state`. All existing
  columns are kept. `return_permutation_importance=True` fills in the test-set
  permutation importance for models without one.

- **Feature selection** (`spectraxai.selection`), scikit-learn compatible and
  usable directly on a `Dataset` (`selector.fit(dataset).transform(dataset)`
  returns a `Dataset` with matching `X_names` and `wavelengths`):
  - `RFE` / `RFECV`: recursive feature elimination for any model, including
    `StandardModel`. Importances are VIP for PLS, the model's own for RF,
    XGBoost and Cubist, coefficients for linear models, and permutation
    importance on the training data otherwise (e.g. SVR).
  - `VIPSelector`: keep wavelengths with PLS VIP above a threshold (default 1).
  - `SelectKBest`: keep the `k` wavelengths ranked highest by a pre-hoc
    `FeatureRanking`.
- `Dataset.select_features(mask_or_indices)`.
- `plot_selected_bands(support, ax=...)` on `PreHocAnalysis` and
  `PostHocAnalysis`: shades the selected wavelengths over the mean spectrum.
  `mean_spectrum` is now also available on `PostHocAnalysis`.
- `DatasetSplit.CLHS` is implemented: conditioned Latin hypercube sampling of the
  training set on the first principal components of the spectra
  (`train_test_split(DatasetSplit.CLHS, 0.8, random_state=...,
  clhs_iterations=..., clhs_components=...)`).

- **New models**: `Model.RIDGE`, `LASSO`, `ELASTIC_NET`, `KNN`, `GPR` (scaled RBF
  plus white-noise kernel, tuned by marginal likelihood) and `MLP`, each with a
  default grid. They all standardize their inputs (the scaler is part of
  `pipeline_`) and support several outputs; the linear models report their
  absolute standardized coefficients as feature importance (`"coef"`).
- **New explanation plots** on `PostHocAnalysis` (all following the plotting
  contract):
  - `plot_importance_spectrum(importance_or_model, ax=...)`: importance on a twin
    axis over the mean spectrum ± SD; returns `(ax, importance_ax)`.
  - `plot_pls_coefficients(model, ax=...)`: PLS regression coefficients per
    wavelength, one line per output.
  - `plot_sage_importance(explanation_or_model, ax=...)`: SAGE values with their
    standard deviation as error bars.
- **Optional SHAP support** (`pip install spectraxai[shap]`):
  `PostHocAnalysis.shap_importance(model, X=None, background=50)` (model-agnostic
  permutation explainer) and `plot_shap_summary(explanation_or_model, ax=...)`
  (mean |SHAP| per wavelength).

- `SpectralPreprocessor` supports `get_feature_names_out` and
  `set_output(transform="pandas")`.
- **Documentation and examples**: a new
  [PyCaret example](https://github.com/tsakiridis/spectra-xai/blob/main/examples/pycaret/pycaretIntegration.ipynb) (PyCaret 4.0
  `RegressionExperiment` with spectraxai pipelines and `StandardModel`); the example
  notebooks cover the new models, pipelines, stateful pre-processing, feature
  selection, cLHS and the new plots; README and package quickstart updated.

- **Mid-infrared spectra.** `Dataset(..., spectral_unit="cm-1")` (also `"nm"`,
  `"um"`; new `spectral_unit`, `is_wavenumber` and `spectral_label` attributes,
  kept by `subset`, `preprocess` and `select_features`). Spectral plots are
  labelled "Wavenumber (cm⁻¹)" and drawn from high to low wavenumber whatever
  the column order, including bar plots; the default (no unit) is unchanged.
  `load_GR_SSL` now sets `spectral_unit="nm"` (axes read "Wavelength (nm)").
- `spectraxai.data.load_GB_MIR()`: MIR absorbance spectra (4000–400 cm⁻¹) of 427
  soils of the Countryside Survey of Great Britain 2007 (Keith et al., 2020,
  Open Government Licence v3.0), stored from high to low wavenumber.
- `excluded_bands` pairs may be given in either order, e.g. `(2400, 2280)`.
- New example scripts with descriptive names: `quickstart.py` and
  `compare_pretreatments.py` (formerly `example2.py` and `example1.py`) and
  `select_wavelengths.py`; new `mirSpectra.ipynb` notebook.

### Changed (models)
- `fit_and_predict` fits the pre-processing on the training samples and applies
  it to the test samples. For stateless pre-processing the results are identical
  to 1.1.0.
- Continuum removal in `Dataset.preprocess` / `fit_and_predict` uses the
  dataset's wavelengths when known. For evenly spaced wavelengths (e.g. the
  bundled SSL) results are identical; for unevenly spaced ones they are now
  correct.
- Passing `cv` positionally to `StandardModel.fit(dataset, cv)` is deprecated
  (`DeprecationWarning`); use `cv=`.
- Requires scikit-learn >= 1.6.

### Changed (plots)
- Constructing `PreHocAnalysis` / `PostHocAnalysis` no longer calls
  `plt.style.use("seaborn-v0_8-whitegrid")` globally. Figures the library creates
  are still drawn in that style; to keep it for your own figures use
  `with spectraxai.plotting.style():` (or call `plt.style.use` yourself).
- Plots drawn into axes you supply keep your style; `correlogram` only calls
  `tight_layout` on figures it created.
- `circular_bar_plot_importance` no longer hides the frame of whatever axes was
  current; it raises if the supplied `ax` is not polar.
- `utils.scatter_plot` accepts `(n, 1)` columns (e.g. PLS predictions) and is
  styled like the other plots when it creates the figure.

### Fixed
These fixes may change numeric results compared to 1.1.0.
- **Grid search state leaked between fits.** `StandardModel.fit` stored the
  dataset-dependent SVR `gamma` grid and PLS `n_components` grid in
  `grid_search_hyperparameters`, so every later fit (other folds, other
  pre-processings in `fit_and_predict_multiple`, other datasets) reused the grid
  computed for the first one. Grids are now computed per fit.
- Index bounds checks in `Dataset.subset` and `Dataset.train_test_split_explicit`
  accepted `n_samples` as a valid index.
- `DatasetSplit.STRATIFIED` failed on continuous outputs; it now stratifies on
  quantile bins of the (first) output. Class-like outputs are stratified on their
  labels exactly as before.
- `fit_and_predict` raised `KeyError` for SVR/RF/Cubist when `init_hyperparameters`
  did not contain every reported hyperparameter; missing values now fall back to
  the estimator's own.
- `fit_and_predict` now reports hyperparameters and `feature_importance` for
  XGBoost.
- `fit_and_predict(..., get_model=True)` ignored `get_model` for cross-validation
  folds.
- `sklearn.base.clone` of a `StandardModel` with `init_hyperparameters` failed for
  models with a default grid (SVR, RF, Cubist, XGBoost).
- Mutual-information and F-statistic rankings divided by zero (NaN) for
  uninformative outputs.
- `PostHocAnalysis.bar_plot_permutation_importance` labelled the y axis "Mean
  decrease in impurity"; it is now "Mean decrease in score".
- `continuum_removal` ignored `wvs` for 2-D inputs.
- `continuum_removal` gave wrong results for wavelengths stored in decreasing
  order (e.g. MIR wavenumbers from 4000 to 400 cm⁻¹).
- In Jupyter, when the first figure of a session was created by a spectraxai plot,
  no later figure was displayed (`spectraxai.plotting.style()` now keeps the
  interactive mode that the notebook backend switches on).
- `Spectra.preprocess` silently returned `None` for unknown methods; it now raises.

### Changed
- Project URLs (README, package metadata) point to
  https://github.com/tsakiridis/spectra-xai and its GitHub Pages documentation.
- Packaging moved from `setup.py` to `pyproject.toml` (same dependencies).
- Tests run with pytest, with fast synthetic fixtures; the SSL-based model tests
  are marked `slow`. CI also runs flake8, black, pyright and the examples.
- CI moved from Azure Pipelines to GitHub Actions (`.github/workflows/ci.yml`):
  tests on Python 3.13–3.15, lint and type checks, REUSE licensing, and the
  example scripts and notebooks.

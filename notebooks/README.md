# Scikit-learn SQL notebooks

[A closer training fit can hide larger unseen-row errors](01_tree_capacity_and_held_out_error.ipynb) fits a regression tree on synthetic training rows, scores separate held-out samples, reconciles worker metrics with SQL arithmetic, and examines how tree depth changes the fitted curve and error distributions.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and Git. The notebook uses Python 3.13 or 3.14; the worker explicitly uses Python 3.13. First use needs network access for dependencies, the published Haybarn/VGI extension and the pinned Git worker. Computation then uses local synthetic inputs. A copy of this directory works without the worker source checkout or a local build.

```bash
uv sync --frozen
uv run --frozen python execute.py
```

The visible SQL launches the remote worker revision `9ebed7eb059fd80aa7306b9e87e23f6958491334` using `uvx`, with VGI SDK 0.51.0 / RPC 0.53.1, scikit-learn 1.9.0, skops 0.14.0, NumPy 2.5.0 and SciPy 1.18.0. These library versions match the tested worker lock. The independent notebook environment pins Haybarn 2.0.0rc11. Rerun default and experiment after changing any pin.

For interactive use, register this environment and select it in Jupyter:

```bash
uv run --frozen python -m ipykernel install --user --name vgi-sklearn-notebooks --display-name 'Python (vgi-sklearn notebooks)'
```

Restart and run all after editing `MAX_DEPTH`. Parameters, connection SQL and analytical SQL remain visible; Python setup, plots, interpretation source and cleanup are folded. The runner creates a temporary kernel with this directory as cwd, saves only after success, and requires a separate destination for parameter overrides. It rejects unknown parameters and a destination equal to the source directory.

## Focused experiment and input provenance

```bash
uv run --frozen python execute.py --parameters '{"MAX_DEPTH":12}' --output-dir validation/deeper-tree
```

The default depth is 3; the experiment uses 12. The fixed 80 training and 120 test rows are separate draws from the same synthetic distribution. The fit sees only x and training targets. Prediction receives only an identifier and x; targets and split labels are joined back afterward. The known conditional mean is displayed only because the generator is known, and never enters training.

[Provenance](fixtures/provenance.json) records the CSV hash, exact generator, NumPy version and units. All data are invented, with no observed subjects or timestamps. x is uniform on [−3,3], the mean is `sin(1.5*x)+0.2*x`, and independent normal noise has standard deviation 0.35. Both feature and target are dimensionless. The committed CSV is sufficient; to regenerate its bytes:

```bash
uv run --frozen python generate_fixture.py
```

## Model storage, checks and limits

`NotebookRegistry` in [notebook_support.py](notebook_support.py) creates a temporary directory and points the worker's supported `SKLEARN_MODELS_DIR` setting to it. Fitting stores only `notebook_tree` there; cleanup closes the connection, restores the environment and removes the directory. The fit call is materialized once before display. The model BLOB remains in an in-memory SQL table and is not displayed or committed. No shared registry or remote service is written.

Thirteen SQL checks cover row identity, fit metadata, feature selection, registry isolation, prediction completeness, finite values, numerical metrics and CDF endpoints. Worker RMSE/MAE are independently reconciled with explicit SQL arithmetic. The synthetic fixture's partition sizes and hash are also checked. All estimator fitting, prediction and worker metrics pass through VGI; Python only creates the documented fixture and presents returned results.

The two predeclared depths illustrate a training/test distinction on one fixed sample, not a population accuracy guarantee or a hyperparameter search. Viewing held-out results does not leave them untouched for later model selection. A real selection workflow needs separate validation and final test data. R² can be negative; missing predictions are not silently dropped or treated as zero. Serialized models are coupled to the pinned scikit-learn version.

Saved default outputs are embedded in the notebook. Generated logs, charts and experiment copies are ignored under `validation/`; the runner copies fixtures, branding and runtime companions with experiments. [STANDARD.md](STANDARD.md) records presentation conventions. Source checks passed 248 pytest tests, 389 stdio assertions and 389 Unix-launch assertions across 17 cases, plus Ruff/format, mypy and isolated pydoclint. The shipped HTTP suite passed 360 assertions in 16 cases and skipped its composed case; that file omits httpfs setup. Adding signed INSTALL/LOAD httpfs only to a disposable validation copy made the composed case pass 33 assertions (29 original assertions plus four loader assertions). No source tests were changed.

The existing metadata-quality gate fails at 85/100, L1: 103 warnings and 146 info findings, with zero errors. Warnings concern dynamic-result schema documentation and empty registry examples. These source metadata findings are separate from the successful notebook checks; worker code and root dependency locks remain unchanged.

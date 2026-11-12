# IMPORTS
import os
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapiq
import sdm
import torch

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabdpt import TabDPTRegressor
from tabicl import TabICLRegressor
from xgboost import XGBRegressor


# SETTINGS

# Synthetic dataset and evaluation settings
n_samples = 500                    # Observations in each fixed synthetic feature matrix.
n_features = 10                    # Total predictors, including irrelevant ones (>= 6).
test_fraction = 0.25               # Fraction of observations held out for R² and SHAP.
feature_mean = 0.0                 # Mean of continuous predictors before rho transformation.
feature_std = 1.0                  # SD of continuous predictors before rho transformation.
n_samples_test = 100               # Held-out observations explained with SHAP per fit.
shap_budget = 32                   # Model evaluation budget per SHAP explanation.
shap_background_size = None        # None uses shapiq's default background sample size.
sim_1_form = "nonlinear"           # "linear" or "nonlinear"

# Repeated experiments
n_replicates = 10                  # Scores averaged for each model and parameter level.
noise_variance_levels = [0.25, 0.50, 1.00, 2.00, 4.00]
rho = 0.0                          # Fixed predictor rho in noise-variance sweeps.
rho_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
rho_noise_variance = 1.0           # Fixed response-error variance in rho sweeps.

# SYNTHETIC DATASET FORMS

sim_1_x1_coefficient = 2.0
sim_1_x2_coefficient = 1.5

def synthetic_1_signal(X):
    x1 = X[:, 0]
    x2 = X[:, 1]

    if sim_1_form == "linear":
        return (
            sim_1_x1_coefficient * x1
            + sim_1_x2_coefficient * x2
        )

    elif sim_1_form == "nonlinear":
        return (
            2.0 * np.sin(2.0 * x1)
            + 1.5 * x2**2
            + x1 * x2
        )

    else:
        raise ValueError("Unknown simulation 1 form")

sim_2_indicator_cutoff = 0.0
sim_2_indicator_effect = 1.0

def synthetic_2_signal(X):
    """Dataset 2: y = sin(x1)+x2²+x3+1{x4>0}+x5+x6+x5*x6+epsilon."""
    indicator = (X[:, 3] > sim_2_indicator_cutoff).astype(float)
    return (
        np.sin(X[:, 0])
        + X[:, 1] ** 2
        + X[:, 2]
        + sim_2_indicator_effect * indicator
        + X[:, 4]
        + X[:, 5]
        + X[:, 4] * X[:, 5]
    )

# Zero-based true feature indices (for SHAP ranking accuracy).
SIMULATION_1_TRUE_FEATURE_IDS = (0, 1)           # x1, x2
SIMULATION_2_TRUE_FEATURE_IDS = (0, 1, 2, 3, 4, 5)  # x1 ... x6
SIMULATION_2_BINARY_FEATURE_ID = 5               # x6 is Bernoulli(0.5), not Gaussian.

# ICE feature selections (replicate 1, when detailed plots are enabled)
SYNTHETIC_1_ICE_FEATURES = ("x1", "x2")
SYNTHETIC_2_ICE_FEATURES = ("x1", "x2", "x3", "x4", "x5", "x6")

# Random seeds
synthetic_1_seed = 42
synthetic_2_seed = 84
split_seed = 42
ranking_seed_base = 10000
model_seed = 42

# Model parameters:
tabicl_n_estimators = 8
tabicl_kv_cache = False
tabdpt_n_ensembles = 1
tabdpt_context_size = None
tabdpt_use_flash = False
tabdpt_compile = False
kumo_size = "medium"              # "small", "medium", or "large".
kumo_n_estimators = 1
kumo_query_batch_size = 128
xgb_n_estimators = 100
xgb_max_depth = 4
xgb_learning_rate = 0.1

# Plot parameters:
ice_max_levels = 10
ice_lower_percentile = 5
ice_upper_percentile = 95
graph_dpi = 300
show_graphs = False

# Execution toggles
RUN_SMOKE_TEST_ONLY = False        # True: one model, one dataset, one replicate, both sweeps.
SMOKE_TEST_MODEL = "Kumo"       # Choose one of MODEL_NAMES below.
SMOKE_TEST_SIMULATION = 2          # 1 or 2. The smoke test checks all configured levels.
RUN_VARIANCE_SWEEP = True
RUN_RHO_SWEEP = True
RUN_SYNTHETIC_1 = True
RUN_SYNTHETIC_2 = True
RUN_DETAILED_PLOTS = False         # True: SHAP bars/beeswarms, ICE, LOCO for replicate 1.
save_csv = True                   # Write sweep_runs.csv and sweep_summary.csv.
standardize_signal = True

OUTPUT_DIR = Path(os.environ.get("MODEL_OUTPUT_DIR") or
                  str(Path.home() / "model_outputs")).expanduser().resolve()

MODEL_NAMES = ("XGBoost", "TabICL", "TabDPT", "Kumo")
MODEL_COLORS = {"XGBoost": "tab:green", "TabICL": "tab:blue",
                "TabDPT": "tab:red", "Kumo": "tab:purple"}
MODEL_MARKERS = {"XGBoost": "X", "TabICL": "o", "TabDPT": "D", "Kumo": "s"}

graph_number = 1
original_show = plt.show

def save_and_show(*args, **kwargs):
    global graph_number
    filename = OUTPUT_DIR / f"Graph_{graph_number:02d}.png"
    plt.savefig(filename, dpi=graph_dpi, bbox_inches="tight")
    print(f"Saved: {filename}", flush=True)
    graph_number += 1
    if show_graphs:
        original_show(*args, **kwargs)

plt.show = save_and_show


# SETTINGS VALIDATION / CSV HELPERS
def validate_settings():
    if not isinstance(n_samples, int) or n_samples < 20 or n_features < 6:
        raise ValueError("n_samples >= 20 and n_features >= 6 are required")
    if n_replicates < 1 or n_samples_test < 1 or shap_budget < 1:
        raise ValueError("n_replicates, n_samples_test and shap_budget must be positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if feature_std <= 0 or not np.isfinite(feature_std) or not np.isfinite(feature_mean):
        raise ValueError("feature_std must be positive; feature_mean must be finite")
    if n_samples_test > int(np.ceil(n_samples * test_fraction)):
        raise ValueError("n_samples_test exceeds the synthetic test-set size")
    if not noise_variance_levels or any(not np.isfinite(v) or v < 0 for v in noise_variance_levels):
        raise ValueError("noise_variance_levels must be nonempty, finite, nonnegative")
    if not rho_levels or not np.isfinite(rho_noise_variance) or rho_noise_variance < 0:
        raise ValueError("rho_levels must be nonempty; rho_noise_variance must be nonnegative")
    if len(set(noise_variance_levels)) != len(noise_variance_levels):
        raise ValueError("Noise-variance levels must be unique")
    if len(set(rho_levels)) != len(rho_levels):
        raise ValueError("Rho levels must be unique")
    if not isinstance(SIMULATION_2_BINARY_FEATURE_ID, int) or not 0 <= SIMULATION_2_BINARY_FEATURE_ID < n_features:
        raise ValueError("SIMULATION_2_BINARY_FEATURE_ID is outside the feature range")
    for sim, indices in ((1, SIMULATION_1_TRUE_FEATURE_IDS), (2, SIMULATION_2_TRUE_FEATURE_IDS)):
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"Simulation {sim} must have unique true feature indices")
        if any(not isinstance(i, int) or not 0 <= i < n_features for i in indices):
            raise ValueError(f"Invalid true feature index in simulation {sim}")
    for sim, ice_features in ((1, SYNTHETIC_1_ICE_FEATURES), (2, SYNTHETIC_2_ICE_FEATURES)):
        if any(not f.startswith("x") or not f[1:].isdigit() or not 1 <= int(f[1:]) <= n_features
               for f in ice_features):
            raise ValueError(f"Invalid ICE feature name in simulation {sim}")
    for sim in (1, 2):
        n_continuous = n_features - (1 if sim == 2 else 0)
        minimum_rho = -1 / (n_continuous - 1)
        for r in [rho] + list(rho_levels):
            if not np.isfinite(r) or not minimum_rho < r < 1:
                raise ValueError(f"rho for simulation {sim} must be in ({minimum_rho:.4f}, 1)")
    if kumo_size not in ("small", "medium", "large"):
        raise ValueError("kumo_size must be small, medium, or large")
    if kumo_query_batch_size < 1 or kumo_n_estimators < 1 or tabdpt_n_ensembles < 1:
        raise ValueError("Invalid ensemble or batch size")
    if ice_max_levels < 2 or not 0 <= ice_lower_percentile < ice_upper_percentile <= 100:
        raise ValueError("Invalid ICE levels or percentiles")
    if SMOKE_TEST_MODEL not in MODEL_NAMES or SMOKE_TEST_SIMULATION not in (1, 2):
        raise ValueError("SMOKE_TEST_MODEL must be a known model; simulation must be 1 or 2")
    if not RUN_SMOKE_TEST_ONLY and not (RUN_VARIANCE_SWEEP or RUN_RHO_SWEEP):
        raise ValueError("Enable at least one sweep")
    if not RUN_SMOKE_TEST_ONLY and not (RUN_SYNTHETIC_1 or RUN_SYNTHETIC_2):
        raise ValueError("Enable at least one synthetic dataset")


def write_csv_rows(rows, filename):
    if save_csv and rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / filename, index=False)


# MODEL CONSTRUCTORS:
def make_tabicl_regressor():
    return TabICLRegressor(
        n_estimators=tabicl_n_estimators,
        kv_cache=tabicl_kv_cache,
    )


def make_xgboost_regressor():
    return XGBRegressor(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        random_state=model_seed,
    )


class TabDPTRegressorAdapter:
    def __init__(self):
        self.model = TabDPTRegressor(
            use_flash=tabdpt_use_flash,
            compile=tabdpt_compile,
        )

    def fit(self, X, y):
        self.model.fit(np.asarray(X), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        result = self.model.predict(
            np.asarray(X),
            context_size=tabdpt_context_size,
            n_ensembles=tabdpt_n_ensembles,
            seed=model_seed,
        )
        return np.asarray(result, dtype=float).reshape(-1)


@lru_cache(maxsize=3)
def get_shared_kumo_model(size, device):
    return sdm.models.KumoTabular(
        task="regression",
        size=size,
        pretrained=True,
        device=device,
    ).eval()

KUMO_ACTIVE_ADAPTER = None

class KumoRegressorAdapter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x_context = None
        self.y_context = None
        self.feature_names = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.feature_names = [f"feature_{j}" for j in range(X.shape[1])]

        df = pd.DataFrame(X, columns=self.feature_names)
        df["target"] = y
        table = sdm.TableTensor.from_pandas(
            df=df,
            stypes={column: "numerical" for column in df.columns},
            device=self.device,
        )

        self.x_context = table.drop_columns("target")
        self.y_context = table["target"]
        return self

    def predict(self, X):
        global KUMO_ACTIVE_ADAPTER

        if self.x_context is None:
            raise RuntimeError("Call Kumo.fit() before predicting")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) == 0:
            return np.array([], dtype=float)

        model = get_shared_kumo_model(kumo_size, self.device)
        if KUMO_ACTIVE_ADAPTER is not self:
            model.clear()
            generator = torch.Generator(device=self.device).manual_seed(model_seed)
            with torch.inference_mode():
                model.fit(
                    x=self.x_context,
                    y=self.y_context,
                    num_estimators=kumo_n_estimators,
                    generator=generator,
                )
            KUMO_ACTIVE_ADAPTER = self

        preds = []
        for start in range(0, len(X), kumo_query_batch_size):
            part = X[start:start + kumo_query_batch_size]
            df = pd.DataFrame(part, columns=self.feature_names)
            query = sdm.TableTensor.from_pandas(
                df=df,
                stypes={column: "numerical" for column in df.columns},
                device=self.device,
            )
            with torch.inference_mode():
                result = model.predict(query)
            quantiles = result.numerical.detach().cpu().numpy()
            quantiles = np.asarray(quantiles, dtype=float)
            if quantiles.ndim == 3 and quantiles.shape[0] == 1:
                quantiles = quantiles[0]
            if quantiles.ndim != 2 or quantiles.shape != (len(part), 999):
                raise ValueError(f"Unexpected Kumo quantiles shape: {quantiles.shape}")
            preds.append(quantiles.mean(axis=-1))

        return np.concatenate(preds)


def make_model(model_name):
    if model_name == "XGBoost":
        return make_xgboost_regressor()
    if model_name == "TabICL":
        return make_tabicl_regressor()
    if model_name == "TabDPT":
        return TabDPTRegressorAdapter()
    if model_name == "Kumo":
        return KumoRegressorAdapter()
    raise ValueError(f"Unknown model: {model_name}")


def predict_flat(model, X):
    y_pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if len(y_pred) != len(X) or not np.isfinite(y_pred).all():
        raise ValueError("Predictions must be finite and have one value per observation")
    return y_pred

# SYNTHETIC GENERATION / EXPLANATION HELPERS

# SYNTHETIC DATA GENERATION
def build_simulation_signal(X, simulation_number):
    """Dispatch to the target equations in Settings, without target noise."""
    if simulation_number == 1:
        return synthetic_1_signal(X)
    if simulation_number == 2:
        return synthetic_2_signal(X)
    raise ValueError("Unknown simulation number")


def generate_synthetic_features(simulation_number):
    """Generate one X matrix per simulation and sweep, as in V6.

    Replicates reuse X; training/test splits and target-error draws vary by
    replicate. This preserves V6's comparison across parameter levels.
    """
    seed = synthetic_1_seed if simulation_number == 1 else synthetic_2_seed
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=feature_mean, scale=feature_std,
                   size=(n_samples, n_features))
    if simulation_number == 2:
        X[:, SIMULATION_2_BINARY_FEATURE_ID] = rng.integers(
            low=0, high=2, size=n_samples)
    return X


def get_continuous_feature_ids(simulation_number):
    if simulation_number == 1:
        return list(range(n_features))
    return [i for i in range(n_features) if i != SIMULATION_2_BINARY_FEATURE_ID]


def get_simulation_configuration(simulation_number):
    if simulation_number == 1:
        return SIMULATION_1_TRUE_FEATURE_IDS, SYNTHETIC_1_ICE_FEATURES
    return SIMULATION_2_TRUE_FEATURE_IDS, SYNTHETIC_2_ICE_FEATURES


# SHAP EXPLANATIONS
def make_shap_explanations(model, X_train, X_explain):
    explainer = shapiq.TabularExplainer(
        model=lambda X_values: predict_flat(model, X_values),
        data=X_train,
        index="SV",
        max_order=1,
        random_state=model_seed,
        sample_size=shap_background_size,
    )
    return [
        explainer.explain(x=x, budget=shap_budget)
        for x in X_explain
    ]


def get_mean_absolute_shap_ranking(explanations):
    shap_values = np.vstack([
        explanation.get_n_order_values(1)
        for explanation in explanations
    ])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return np.argsort(-mean_abs_shap)

# LOCO AND ICE:
def plot_loco(model_factory, X_train, y_train, X_test, y_test,
              feature_names, title):
    model = model_factory()
    model.fit(X_train, y_train)
    baseline_loss = mean_squared_error(y_test, predict_flat(model, X_test))
    scores = []

    for feature_id in range(X_train.shape[1]):
        X_train_reduced = np.delete(X_train, feature_id, axis=1)
        X_test_reduced = np.delete(X_test, feature_id, axis=1)
        reduced_model = model_factory()
        reduced_model.fit(X_train_reduced, y_train)
        reduced_loss = mean_squared_error(
            y_test, predict_flat(reduced_model, X_test_reduced)
        )
        scores.append(reduced_loss - baseline_loss)

    scores = np.asarray(scores)
    order = np.argsort(scores)
    plt.figure(figsize=(10, 7))
    plt.barh(np.asarray(feature_names)[order], scores[order])
    plt.xlabel("Increase in Loss When Feature Is Removed")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()
    return int(np.argmax(scores))


def plot_ice(model, X, feature_id, feature_names, title,
             max_levels=None):
    if max_levels is None:
        max_levels = ice_max_levels
    if feature_id is None:
        print(f"{title} skipped: enter feature ID from LOCO plot.")
        return

    unique_values = np.unique(X[:, feature_id])
    if len(unique_values) <= 2 and set(unique_values).issubset({0.0, 1.0}):
        feature_values = np.array([0.0, 1.0])
    else:
        feature_values = np.linspace(
            np.percentile(X[:, feature_id], ice_lower_percentile),
            np.percentile(X[:, feature_id], ice_upper_percentile),
            max_levels,
        )

    ice_predictions = []
    for value in feature_values:
        X_changed = X.copy()
        X_changed[:, feature_id] = value
        ice_predictions.append(predict_flat(model, X_changed))
    ice_predictions = np.asarray(ice_predictions).T

    plt.figure(figsize=(9, 6))
    plt.plot(feature_values, ice_predictions.T, alpha=0.25, linewidth=1)
    plt.plot(
        feature_values,
        ice_predictions.mean(axis=0),
        linewidth=3,
        label="Average",
    )
    plt.xlabel(feature_names[feature_id])
    plt.ylabel("Predicted Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# RHO AND RANKING METRICS:
def apply_rho_to_existing_data(X, continuous_feature_ids, rho_value,
                               transform_at_zero=False):
    X_correlated = np.asarray(X, dtype=float).copy()
    if rho_value == 0 and not transform_at_zero:
        return X_correlated
    n_continuous = len(continuous_feature_ids)
    minimum_rho = -1 / (n_continuous - 1)
    if not minimum_rho < rho_value < 1:
        raise ValueError(f"rho must be between {minimum_rho:.4f} and 1")

    X_continuous = X_correlated[:, continuous_feature_ids].copy()
    X_continuous = (
        X_continuous - X_continuous.mean(axis=0)
    ) / X_continuous.std(axis=0, ddof=1)

    sample_cholesky = np.linalg.cholesky(
        np.corrcoef(X_continuous, rowvar=False)
    )
    X_white = np.linalg.solve(sample_cholesky, X_continuous.T).T
    target_corr = np.full((n_continuous, n_continuous), rho_value)
    np.fill_diagonal(target_corr, 1.0)
    target_cholesky = np.linalg.cholesky(target_corr)
    X_correlated[:, continuous_feature_ids] = X_white @ target_cholesky.T
    return X_correlated


def calculate_top_k_accuracy(ranking, true_feature_ids):
    k = len(true_feature_ids)
    return len(set(ranking[:k]) & set(true_feature_ids)) / k


def calculate_rank_sum(ranking, true_feature_ids):
    rank_positions = np.empty(len(ranking), dtype=int)
    rank_positions[ranking] = np.arange(1, len(ranking) + 1)
    return float(sum(rank_positions[i] for i in true_feature_ids))

# FIRST-REPLICATE SHAP / ICE / LOCO GRAPHS
def plot_detailed_interpretability(model_name, model, X_train, X_test,
                                  y_train, y_test, X_explain, explanations,
                                  simulation_number, sweep_type, level):
    """Plot detailed diagnostics once for each model and level (replicate 1)."""
    feature_names = [f"x{i+1}" for i in range(n_features)]
    _, ice_features = get_simulation_configuration(simulation_number)
    sweep_label = "Noise Variance" if sweep_type == "noise_variance" else "Rho"
    context = (f"{model_name} - Synthetic Dataset {simulation_number} - "
               f"{sweep_label} = {level:g} - Replicate 1")

    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(explanations, feature_names=feature_names, show=False)
    plt.title(f"{context}\nSHAP Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(explanations, data=X_explain,
                             feature_names=feature_names, show=False)
    plt.title(f"{context}\nSHAP Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()
    
    for feature_name in ice_features:
        plot_ice(model, X_test, feature_names.index(feature_name),
                 feature_names, f"{context}\nICE - {feature_name}")

    plot_loco(lambda: make_model(model_name), X_train, y_train, X_test,
              y_test, feature_names, f"{context}\nLOCO")


# COMBINED SWEEP RUNNER
SWEEP_RUNS = []

def save_sweep_outputs():
    """Persist partial run scores and current averages, so jobs may be inspected."""
    if not SWEEP_RUNS:
        return pd.DataFrame()
    write_csv_rows(SWEEP_RUNS, "sweep_runs.csv")
    summary = pd.DataFrame(SWEEP_RUNS).groupby(
        ["sweep", "simulation", "level", "noise_variance", "rho", "model"],
        as_index=False,
    ).agg(
        r2=("r2", "mean"),
        top_k=("top_k", "mean"),
        rank_sum=("rank_sum", "mean"),
        replicates_completed=("replicate", "count"),
    )
    write_csv_rows(summary.to_dict("records"), "sweep_summary.csv")
    return summary


def run_parameter_ranking_experiment(X_base, simulation_number, true_feature_ids,
                                     sweep_type, model_names, replicates):
    """V6's parameter sweep, now recording held-out R² in the same rows.

    For each replicate, X/split/base_noise remain paired across levels,
    matching the original code. Detailed plots use only replicate 1.
    """
    levels = noise_variance_levels if sweep_type == "noise_variance" else rho_levels
    continuous_feature_ids = get_continuous_feature_ids(simulation_number)

    for replicate_id in range(replicates):
        sample_ids = np.arange(len(X_base))
        train_ids, test_ids = train_test_split(
            sample_ids, test_size=test_fraction,
            random_state=split_seed + replicate_id,
        )
        rng = np.random.default_rng(
            ranking_seed_base + 1000 * simulation_number + replicate_id
        )
        base_noise = rng.normal(loc=0.0, scale=1.0, size=len(X_base))

        for level in levels:
            current_rho = rho if sweep_type == "noise_variance" else level
            current_variance = (
                level if sweep_type == "noise_variance" else rho_noise_variance
            )
            X = apply_rho_to_existing_data(
                X_base, continuous_feature_ids, current_rho,
                transform_at_zero=(sweep_type == "rho"),
            )
            signal = build_simulation_signal(X, simulation_number)
            if standardize_signal:
                signal = (
                (signal - signal.mean())
                / signal.std(ddof=0)
                )
            y = signal + np.sqrt(current_variance) * base_noise
            X_train, X_test = X[train_ids], X[test_ids]
            y_train, y_test = y[train_ids], y[test_ids]
            X_explain = X_test[:n_samples_test]

            print(f"Simulation {simulation_number} | {sweep_type} = {level:g} | "
                  f"replicate {replicate_id + 1}/{replicates}", flush=True)

            for model_name in model_names:
                model = make_model(model_name)
                model.fit(X_train, y_train)
                predictions = predict_flat(model, X_test)
                r2_value = float(r2_score(y_test, predictions))

                explanations = make_shap_explanations(model, X_train, X_explain)
                ranking = get_mean_absolute_shap_ranking(explanations)
                row = {
                    "sweep": sweep_type,
                    "simulation": simulation_number,
                    "replicate": replicate_id + 1,
                    "level": level,
                    "noise_variance": current_variance,
                    "rho": current_rho,
                    "model": model_name,
                    "r2": r2_value,
                    "top_k": calculate_top_k_accuracy(ranking, true_feature_ids),
                    "rank_sum": calculate_rank_sum(ranking, true_feature_ids),
                }
                SWEEP_RUNS.append(row)
                save_sweep_outputs()
                print(f"  {model_name}: R²={r2_value:.4f}, "
                      f"top-k={row['top_k']:.4f}, "
                      f"rank sum={row['rank_sum']:.2f}", flush=True)

                if RUN_DETAILED_PLOTS and replicate_id == 0:
                    plot_detailed_interpretability(
                        model_name, model, X_train, X_test, y_train, y_test,
                        X_explain, explanations, simulation_number,
                        sweep_type, level,
                    )

    summary = save_sweep_outputs()
    return summary.loc[
        (summary["sweep"] == sweep_type)
        & (summary["simulation"] == simulation_number)
    ].copy()


# SUMMARY PLOTTING:
def format_rank_sum_label(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}"


def add_coordinate_labels(x_values, y_values, metric, vertical_offset):
    for x_value, y_value in zip(x_values, y_values):
        if metric == "rank_sum":
            y_text = format_rank_sum_label(y_value)
        elif metric == "r2":
            y_text = f"{y_value:.3f}"
        else:
            y_text = f"{y_value:.2f}"
        plt.annotate(
            f"({x_value:g}, {y_text})",
            xy=(x_value, y_value),
            xytext=(0, vertical_offset),
            textcoords="offset points", ha="center", va="center",
            fontsize=8, zorder=10,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.95},
        )


def plot_parameter_metric(results, simulation_number, metric, sweep_type,
                          model_names, replicates):
    x_values = noise_variance_levels if sweep_type == "noise_variance" else rho_levels
    plt.figure(figsize=(10, 6))
    for model_number, model_name in enumerate(model_names):
        model_results = results[results["model"] == model_name].set_index("level")
        y_values = [float(model_results.loc[level, metric]) for level in x_values]
        plt.plot(
            x_values, y_values, marker=MODEL_MARKERS[model_name], linewidth=2,
            label=model_name, color=MODEL_COLORS[model_name], zorder=2,
        )
        add_coordinate_labels(
            x_values, y_values, metric,
            vertical_offset=12 if model_number % 2 == 0 else -13,
        )
    metric_names = {
        "r2": ("Mean Test R² (Higher = Better)", "Mean Test R²"),
        "top_k": ("Top-k Accuracy (Higher = Better)", "Mean SHAP Top-k Accuracy"),
        "rank_sum": ("Rank Sum (Lower = Better)", "Mean SHAP Rank Sum"),
    }
    x_title = "Noise Variance" if sweep_type == "noise_variance" else "Rho"
    plt.xlabel(x_title)
    plt.ylabel(metric_names[metric][0])
    plt.title(
        f"Synthetic Dataset {simulation_number} - {metric_names[metric][1]} "
        f"vs {x_title} ({replicates} Replicate{'s' if replicates != 1 else ''})"
    )
    plt.xticks(x_values)
    if metric == "top_k":
        plt.ylim(-0.08, 1.12)
    else:
        plt.margins(y=0.15)
        if metric == "r2":
            plt.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    plt.legend()
    plt.grid(alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.show()
    plt.close()


def run_sweep(sweep_type, simulations, model_names, replicates):
    """Run either V7 experiment and save all 3 averaged comparison metrics."""
    x_title = "Noise Variance" if sweep_type == "noise_variance" else "Rho"
    print(f"\n===== {x_title.upper()} SWEEP =====", flush=True)
    for simulation_number in simulations:
        X_base = generate_synthetic_features(simulation_number)
        true_feature_ids, _ = get_simulation_configuration(simulation_number)
        results = run_parameter_ranking_experiment(
            X_base, simulation_number, true_feature_ids, sweep_type,
            model_names, replicates,
        )
        for metric in ("top_k", "rank_sum", "r2"):
            plot_parameter_metric(results, simulation_number, metric,
                                  sweep_type, model_names, replicates)


# EXECUTION
def main():
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"V7 output directory: {OUTPUT_DIR}", flush=True)
    print(f"Detailed plots: {RUN_DETAILED_PLOTS}; save_csv: {save_csv}", flush=True)

    if RUN_SMOKE_TEST_ONLY:
        print(f"SMOKE TEST: {SMOKE_TEST_MODEL}, Synthetic Dataset "
              f"{SMOKE_TEST_SIMULATION}, 1 replicate; both sweeps", flush=True)
        # Runs the actual sweep/plot functions, not the V6 installation-only test.
        for sweep_type in ("noise_variance", "rho"):
            run_sweep(sweep_type, [SMOKE_TEST_SIMULATION],
                      [SMOKE_TEST_MODEL], replicates=1)
        print("V7 smoke test completed successfully.", flush=True)
        return

    simulations = ([1] if RUN_SYNTHETIC_1 else []) + ([2] if RUN_SYNTHETIC_2 else [])
    if RUN_VARIANCE_SWEEP:
        run_sweep("noise_variance", simulations, MODEL_NAMES, n_replicates)
    if RUN_RHO_SWEEP:
        run_sweep("rho", simulations, MODEL_NAMES, n_replicates)
    print(f"Finished V7. Graphs and optional CSVs are in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
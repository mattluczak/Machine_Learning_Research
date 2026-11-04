# IMPORTS

from functools import lru_cache
from itertools import combinations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapiq
import sdm
import torch

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabdpt import TabDPTRegressor
from tabicl import TabICLRegressor
from xgboost import XGBRegressor

matplotlib.use("Agg")

# AUTO SAVE ALL GRAPHS

graph_number = 1
original_show = plt.show

def save_and_show(*args, **kwargs):
    global graph_number

    filename = OUTPUT_DIR / f"Graph_{graph_number:02d}.png"

    plt.savefig(
        filename,
        dpi=graph_dpi,
        bbox_inches="tight"
    )

    print(f"Saved: {filename}", flush=True)

    graph_number += 1
    if show_graphs:
        original_show(*args, **kwargs)

plt.show = save_and_show

# SETTINGS

# Dataset and SHAP settings
n_samples_test = 2                 # Test observations explained with SHAP per model / condition.
shap_budget = 16                   # Model-evaluation budget per SHAP explanation (not total budget).
shap_background_size = None       # SHAP background sample count; None uses the shapiq default.
n_samples = 500                    # Rows in each synthetic dataset.
n_features = 10                    # Predictors, including noise features; must be at least 6.
test_fraction = 0.25               # Fraction of observations held out for evaluation.
feature_mean = 0.0                 # Mean of simulated continuous predictor distributions.
feature_std = 1.0                  # Standard deviation of simulated continuous predictors.
target_noise_std = 1.0             # Target error standard deviation in baseline experiments.

# Repeated-experiment settings
n_replicates = 1                  # Independent replicate scores averaged for R2 and rankings.
noise_variance_levels = [0.25, 0.50, 1.00, 2.00, 4.00]  # X-axis values in variance sweeps.
rho = 0.0                          # Fixed predictor correlation in noise-variance sweeps.
rho_levels = [0.0, 0.2, 0.4, 0.6, 0.8]  # X-axis values in correlation sweeps.
rho_noise_variance = 1.0           # Fixed response-error variance in correlation sweeps.

# Model parameters
tabicl_n_estimators = 8           # Number of TabICL ensemble estimators.
tabicl_kv_cache = False           # Whether TabICL uses its key/value prediction cache.
tabdpt_n_ensembles = 1            # Number of TabDPT inference ensembles for fast initial tests.
tabdpt_context_size = None        # TabDPT max context rows; None uses all available context.
tabdpt_use_flash = False
tabdpt_compile = False
kumo_size = "medium"              # Kumo Tabular size: "small", "medium", or "large".
kumo_n_estimators = 1             # Kumo inference ensemble size (higher is slower).
kumo_query_batch_size = 128       # Maximum Kumo query rows per prediction batch.
model_seed = 42                   # Reproducibility seed used by models where supported.
xgb_n_estimators = 100            # Number of XGBoost boosting trees.
xgb_max_depth = 4                 # Maximum depth of each XGBoost tree.
xgb_learning_rate = 0.1           # XGBoost learning rate / contribution of each tree.

# Plot and method parameters
ice_max_levels = 10               # ICE prediction grid size for continuous features.
ice_lower_percentile = 5          # Lower percentile endpoint of continuous ICE grid.
ice_upper_percentile = 95        # Upper percentile endpoint of continuous ICE grid.
sensitivity_perturbation = 0.10    # Sensitivity perturbation as a fraction of feature SD.
graph_dpi = 300                    # Exported PNG resolution.
show_graphs = False               # Show interactive windows; False is suitable for HPC nodes.
OUTPUT_DIR = Path.home() / "model_outputs"  # Folder containing Graph_XX.png and optional CSVs.
BOSTON_DATA_CACHE = OUTPUT_DIR / "housing.data"  # Local Boston data cache for offline nodes.

# Random seeds, controls and target functions
synthetic_1_seed = 42             # Base seed for synthetic dataset 1 and its first noise draw.
synthetic_2_seed = 84             # Base seed for synthetic dataset 2 and its first noise draw.
boston_sim_1_seed = 142           # Boston simulation 1 noise / first target-error seed.
boston_sim_2_seed = 184           # Boston simulation 2 noise / first target-error seed.
split_seed = 42                   # Train/test split seed for replicate 1; incremented thereafter.
ranking_seed_base = 10000         # Noise-seed offset for repeated SHAP ranking experiments.
sim_1_x1_coefficient = 2.0        # Coefficient of x1 in first simulation target formula.
sim_1_x2_coefficient = 1.0        # Coefficient of x2 in first simulation target formula.
sim_2_indicator_cutoff = 0.0      # Threshold for 1{x4 > cutoff} in simulation 2.
sim_2_indicator_effect = 1.0      # Target effect when x4 exceeds cutoff.
sim_1_boston_noise_features = 8   # Irrelevant predictors added to Boston simulation 1.
sim_2_boston_noise_features = 4   # Irrelevant predictors added to Boston simulation 2.

# ICE feature selections
SYNTHETIC_1_ICE_FEATURES = ["x1", "x2"]
SYNTHETIC_2_ICE_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]

# Execution toggles
RUN_SMOKE_TEST_ONLY = False       # Runs a test for the model installations.
RUN_BASELINE = True               # Master toggle for baseline datasets, model plots and R2.
RUN_SYNTHETIC_1 = True            # Include the first synthetic baseline dataset.
RUN_SYNTHETIC_2 = True            # Include the second synthetic baseline dataset.
RUN_BOSTON = True                 # Include all four Boston Housing baseline experiments.
RUN_BOSTON_CORRELATION = True     # Generate the original Boston correlation matrix.
RUN_R2_PLOTS = True              # Compute and graph mean R2 over n_replicates for baselines.
RUN_SHAP_PLOTS = True             # Plot SHAP bars and beeswarms for replicate 1 only.
RUN_LOCO = True                   # Make LOCO plots using baseline replicate 1.
RUN_ICE = True                    # Make ICE plots using baseline replicate 1.
RUN_SENSITIVITY = True            # Make sensitivity boxplots using baseline replicate 1.
RUN_VARIANCE_SWEEP = True         # Run averaged top-k / rank-sum versus noise variance.
RUN_RHO_SWEEP = True              # Run averaged top-k / rank-sum versus rho.
save_csv = True                   # Write the four result CSV files alongside the graphs.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ("XGBoost", "TabICL", "TabDPT", "Kumo")
MODEL_COLORS = {"XGBoost": "tab:green", "TabICL": "tab:blue",
                "TabDPT": "tab:red", "Kumo": "tab:purple"}
MODEL_MARKERS = {"XGBoost": "X", "TabICL": "I", "TabDPT": "D", "Kumo": "K"}

# HELPER FUNCTIONS

# SETTINGS VALIDATION AND NUMERICAL OUTPUT

def validate_settings():
    if n_samples < 20 or n_features < 6:
        raise ValueError("n_samples >= 20 and n_features >= 6 are required")
    if n_replicates < 1 or n_samples_test < 1 or shap_budget < 1:
        raise ValueError("n_replicates, n_samples_test and shap_budget must be positive")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if kumo_size not in ("small", "medium", "large"):
        raise ValueError("kumo_size must be small, medium, or large")
    if not noise_variance_levels or any(v < 0 for v in noise_variance_levels):
        raise ValueError("noise_variance_levels must be nonempty and nonnegative")
    if not rho_levels or rho_noise_variance < 0:
        raise ValueError("rho_levels must be nonempty; rho_noise_variance >= 0")
    if sim_1_boston_noise_features < 0 or sim_2_boston_noise_features < 0:
        raise ValueError("The number of Boston noise features cannot be negative")
    if n_samples_test > int(np.ceil(n_samples * test_fraction)):
        raise ValueError("n_samples_test exceeds the synthetic test-set size")
    for r in [rho] + list(rho_levels):
        if not -1 / (n_features - 2) < r < 1:
            raise ValueError("rho must be between -1/(n_features-2) and 1")
    if ice_max_levels < 2 or not 0 <= ice_lower_percentile < ice_upper_percentile <= 100:
        raise ValueError("Check ICE levels and percentiles")
    if sensitivity_perturbation <= 0 or feature_std <= 0 or target_noise_std < 0:
        raise ValueError("Check feature_std, target_noise_std and sensitivity perturbation")
    if kumo_query_batch_size < 1 or kumo_n_estimators < 1 or tabdpt_n_ensembles < 1:
        raise ValueError("Invalid new-model ensemble or batch size")


def write_csv_rows(rows, filename):
    if save_csv and rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / filename, index=False)


R2_RESULTS = []
RANKING_RESULTS = []

# MODEL CONSTRUCTORS

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

KUMO_ACTIVE_ADAPTER_ID = None

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
        global KUMO_ACTIVE_ADAPTER_ID

        if self.x_context is None:
            raise RuntimeError("Call Kumo.fit() before predicting")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) == 0:
            return np.array([], dtype=float)

        model = get_shared_kumo_model(kumo_size, self.device)
        if KUMO_ACTIVE_ADAPTER_ID != id(self):
            model.clear()
            generator = torch.Generator(device=self.device).manual_seed(model_seed)
            with torch.inference_mode():
                model.fit(
                    x=self.x_context,
                    y=self.y_context,
                    num_estimators=kumo_n_estimators,
                    generator=generator,
                )
            KUMO_ACTIVE_ADAPTER_ID = id(self)

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

# DATA GENERATION AND REPEATED R2

def build_simulation_signal(X, simulation_number):
    if simulation_number == 1:
        return sim_1_x1_coefficient * X[:, 0] + sim_1_x2_coefficient * X[:, 1]
    if simulation_number == 2:
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
    raise ValueError("Unknown simulation number")


def create_baseline_replicate(X_base, simulation_number, kind, base_seed, replicate_id):
    rng = np.random.default_rng(base_seed + replicate_id)

    if kind == "synthetic":
        X = rng.normal(
            loc=feature_mean,
            scale=feature_std,
            size=(n_samples, n_features),
        )
        if simulation_number == 2:
            X[:, 5] = rng.integers(low=0, high=2, size=n_samples)
        epsilon = rng.normal(loc=0.0, scale=target_noise_std, size=n_samples)
    elif kind == "boston":
        X = np.array(X_base, copy=True)
        epsilon = rng.normal(loc=0.0, scale=target_noise_std, size=len(X))
    else:
        raise ValueError("kind must be synthetic or boston")

    y = build_simulation_signal(X, simulation_number) + epsilon
    return train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=split_seed + replicate_id,
    )


def calculate_mean_r2(
    first_models, first_split, X_base, simulation_number, kind,
    base_seed, dataset_label,
):
    dataset_rows = []
    X_train, X_test, y_train, y_test = first_split

    for replicate_id in range(n_replicates):
        if replicate_id != 0:
            X_train, X_test, y_train, y_test = create_baseline_replicate(
                X_base, simulation_number, kind, base_seed, replicate_id,
            )

        for model_name in MODEL_NAMES:
            if replicate_id == 0:
                model = first_models[model_name]
            else:
                model = make_model(model_name)
                model.fit(X_train, y_train)

            r2_value = float(r2_score(y_test, predict_flat(model, X_test)))
            row = {
                "dataset": dataset_label,
                "replicate": replicate_id + 1,
                "model": model_name,
                "r2": r2_value,
            }
            dataset_rows.append(row)
            R2_RESULTS.append(row)
            write_csv_rows(R2_RESULTS, "baseline_r2_runs.csv")
            print(f"{dataset_label} | replicate {replicate_id + 1}/{n_replicates} | "
                  f"{model_name}: R^2 = {r2_value:.4f}", flush=True)

    summary = pd.DataFrame(R2_RESULTS).groupby(
        ["dataset", "model"], as_index=False
    )["r2"].agg(mean_r2="mean", std_r2="std")
    write_csv_rows(summary.to_dict("records"), "baseline_r2_summary.csv")

    return pd.DataFrame(dataset_rows)


def plot_average_r2(dataset_rows, dataset_label):
    means = []
    deviations = []
    for model_name in MODEL_NAMES:
        scores = dataset_rows.loc[dataset_rows["model"] == model_name, "r2"].to_numpy()
        means.append(float(np.mean(scores)))
        deviations.append(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        MODEL_NAMES,
        means,
        color=[MODEL_COLORS[model_name] for model_name in MODEL_NAMES],
        yerr=deviations,
        capsize=5,
    )
    plt.bar_label(bars, fmt="%.4f", padding=3)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    low = min(np.array(means) - np.array(deviations))
    high = max(np.array(means) + np.array(deviations))
    plt.ylim(min(0.0, low - 0.05), max(1.0, high + 0.05))
    plt.ylabel("Mean R^2 (error bars: 1 SD)")
    plt.title(f"{dataset_label} - R^2 Comparison ({n_replicates} Replicates)")
    plt.tight_layout()
    plt.show()
    plt.close()

# SHAP HELPERS

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


def get_mean_absolute_shap_ranking(model, X_train, X_explain):
    explanations = make_shap_explanations(model, X_train, X_explain)
    shap_values = np.vstack([
        explanation.get_n_order_values(1)
        for explanation in explanations
    ])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return np.argsort(-mean_abs_shap)

# LOCO, ICE AND SENSITIVITY HELPERS

def plot_loco(model_factory, X_train, y_train, X_test, y_test,
              feature_names, task, title):
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


def plot_ice(model, X, feature_id, feature_names, task, title,
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


def plot_sensitivity(model, X, feature_names, task, title,
                     perturbation=None):
    if perturbation is None:
        perturbation = sensitivity_perturbation
    baseline = predict_flat(model, X)
    sensitivities = np.zeros((X.shape[0], X.shape[1]))

    for feature_id in range(X.shape[1]):
        step = perturbation * np.std(X[:, feature_id])
        if step == 0:
            continue
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, feature_id] += step
        X_minus[:, feature_id] -= step
        plus = predict_flat(model, X_plus)
        minus = predict_flat(model, X_minus)
        sensitivities[:, feature_id] = (
            np.abs(plus - baseline) + np.abs(minus - baseline)
        ) / 2

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [sensitivities[:, i] for i in range(X.shape[1])],
        tick_labels=feature_names,
        showfliers=False,
    )
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Change in Prediction")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON FEATURE-SELECTION HELPERS

def get_correlation_pairs(correlation_matrix):
    pairs = []
    feature_names = list(correlation_matrix.columns)
    for i, feature_1 in enumerate(feature_names):
        for feature_2 in feature_names[i + 1:]:
            correlation = correlation_matrix.loc[feature_1, feature_2]
            pairs.append((abs(correlation), correlation, feature_1, feature_2))
    return pairs


def average_absolute_correlation(feature_names):
    correlation_matrix = X_boston_standardized[feature_names].corr()
    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool), k=1
    )
    values = np.abs(correlation_matrix.to_numpy()[upper_triangle])
    return float(values.mean())


def make_boston_simulation_1(feature_pair, random_seed=None):
    if random_seed is None:
        random_seed = boston_sim_1_seed
    selected_features = list(feature_pair)
    X = X_boston_standardized[selected_features].copy()
    X.columns = ["x1", "x2"]
    rng = np.random.default_rng(random_seed)
    epsilon = rng.normal(loc=0.0, scale=target_noise_std, size=len(X))
    noise_features = rng.normal(
        loc=feature_mean,
        scale=feature_std,
        size=(len(X), sim_1_boston_noise_features),
    )
    for noise_id in range(sim_1_boston_noise_features):
        X[f"x{noise_id + 3}"] = noise_features[:, noise_id]

    y = (sim_1_x1_coefficient * X["x1"].to_numpy()
         + sim_1_x2_coefficient * X["x2"].to_numpy() + epsilon)
    mapping = {"x1": selected_features[0], "x2": selected_features[1]}
    mapping.update({f"x{i+3}": "Noise" for i in range(sim_1_boston_noise_features)})
    return X, pd.Series(y, name="Target"), mapping


def make_boston_simulation_2(feature_set, random_seed=None):
    if random_seed is None:
        random_seed = boston_sim_2_seed
    selected_features = list(feature_set)
    X = X_boston_standardized[selected_features].copy()
    X.columns = ["x1", "x2", "x3", "x4", "x5", "x6"]
    rng = np.random.default_rng(random_seed)
    epsilon = rng.normal(loc=0.0, scale=target_noise_std, size=len(X))
    noise_features = rng.normal(
        loc=feature_mean,
        scale=feature_std,
        size=(len(X), sim_2_boston_noise_features),
    )
    for noise_id in range(sim_2_boston_noise_features):
        X[f"x{noise_id + 7}"] = noise_features[:, noise_id]
    y = build_simulation_signal(X.to_numpy(), simulation_number=2) + epsilon
    mapping = {f"x{i+1}": feature for i, feature in enumerate(selected_features)}
    mapping.update({f"x{i+7}": "Noise" for i in range(sim_2_boston_noise_features)})
    return X, pd.Series(y, name="Target"), mapping

# NOISE VARIANCE / RHO RANKING HELPERS

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


def run_parameter_ranking_experiment(X_base, simulation_number, true_feature_ids,
                                     sweep_type):
    levels = noise_variance_levels if sweep_type == "noise_variance" else rho_levels
    continuous_feature_ids = (
        list(range(X_base.shape[1])) if simulation_number == 1
        else [i for i in range(X_base.shape[1]) if i != 5]
    )

    for replicate_id in range(n_replicates):
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
            y = signal + np.sqrt(current_variance) * base_noise
            X_train, X_test, y_train = X[train_ids], X[test_ids], y[train_ids]
            X_explain = X_test[:n_samples_test]

            print(f"Simulation {simulation_number} | {sweep_type} = {level:g} | "
                  f"replicate {replicate_id + 1}/{n_replicates}", flush=True)

            for model_name in MODEL_NAMES:
                model = make_model(model_name)
                model.fit(X_train, y_train)
                ranking = get_mean_absolute_shap_ranking(
                    model, X_train, X_explain,
                )
                RANKING_RESULTS.append({
                    "sweep": sweep_type,
                    "simulation": simulation_number,
                    "replicate": replicate_id + 1,
                    "level": level,
                    "noise_variance": current_variance,
                    "rho": current_rho,
                    "model": model_name,
                    "top_k": calculate_top_k_accuracy(ranking, true_feature_ids),
                    "rank_sum": calculate_rank_sum(ranking, true_feature_ids),
                })
                write_csv_rows(RANKING_RESULTS, "ranking_sweep_runs.csv")

    rows = pd.DataFrame(RANKING_RESULTS)
    summary = rows.groupby(
        ["sweep", "simulation", "level", "model"], as_index=False
    )[["top_k", "rank_sum"]].mean()
    write_csv_rows(summary.to_dict("records"), "ranking_sweep_summary.csv")
    return summary[
        (summary["sweep"] == sweep_type)
        & (summary["simulation"] == simulation_number)
    ].copy()


def format_rank_sum_label(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}"


def add_coordinate_labels(x_values, y_values, metric, vertical_offset):
    for x_value, y_value in zip(x_values, y_values):
        y_text = f"{y_value:.2f}" if metric == "top_k" else format_rank_sum_label(y_value)
        plt.annotate(
            f"({x_value:g}, {y_text})",
            xy=(x_value, y_value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            zorder=10,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.95},
        )


def plot_parameter_metric(results, simulation_number, metric, sweep_type):
    x_values = noise_variance_levels if sweep_type == "noise_variance" else rho_levels
    plt.figure(figsize=(10, 6))
    for model_number, model_name in enumerate(MODEL_NAMES):
        model_results = results[results["model"] == model_name].set_index("level")
        y_values = [model_results.loc[level, metric] for level in x_values]
        plt.plot(
            x_values,
            y_values,
            marker=MODEL_MARKERS[model_name],
            linewidth=2,
            label=model_name,
            color=MODEL_COLORS[model_name],
            zorder=2,
        )
        add_coordinate_labels(
            x_values, y_values, metric,
            vertical_offset=12 if model_number % 2 == 0 else -13,
        )
    plt.xlabel("Noise Variance" if sweep_type == "noise_variance" else "Rho")
    plt.ylabel("Top-k Accuracy (Higher = Better)" if metric == "top_k"
               else "Rank Sum (Lower = Better)")
    metric_title = "SHAP Top-k Accuracy" if metric == "top_k" else "SHAP Rank Sum"
    x_title = "Noise Variance" if sweep_type == "noise_variance" else "Rho"
    plt.title(
        f"Synthetic Dataset {simulation_number} - {metric_title} vs {x_title} "
        f"({n_replicates} Replicates)"
    )
    plt.xticks(x_values)
    if metric == "top_k":
        plt.ylim(-0.08, 1.12)
    else:
        plt.margins(y=0.15)
    plt.legend()
    plt.grid(alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.show()
    plt.close()

# SMOKE TEST HELPER

def smoke_test_models():
    rng = np.random.default_rng(synthetic_1_seed)
    X = rng.normal(loc=feature_mean, scale=feature_std, size=(40, n_features))
    y = build_simulation_signal(X, 1)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=test_fraction, random_state=split_seed,
    )
    for model_name in MODEL_NAMES:
        print(f"Smoke test: {model_name}", flush=True)
        model = make_model(model_name)
        model.fit(X_train, y_train)
        prediction = predict_flat(model, X_test[:3])
        print(f"  Predicted {len(prediction)} rows: {prediction}", flush=True)
    print("Four-model smoke test completed.", flush=True)


validate_settings()
if RUN_SMOKE_TEST_ONLY:
    smoke_test_models()
    raise SystemExit(0)

# SYNTHETIC DATASET 1
# y_i = 2*x_i1 + x_i2 + epsilon_i
# epsilon_i ~ N(0, 1)
# All x_ij ~ N(0, 1)

if RUN_BASELINE and RUN_SYNTHETIC_1:
    rng_1 = np.random.default_rng(synthetic_1_seed)

    X_synthetic_1_array = rng_1.normal(
        loc=feature_mean,
        scale=feature_std,
        size=(n_samples, n_features),
    )

    epsilon_1 = rng_1.normal(
        loc=0.0,
        scale=target_noise_std,
        size=n_samples,
    )

    y_synthetic_1_array = (
        sim_1_x1_coefficient * X_synthetic_1_array[:, 0]
        + sim_1_x2_coefficient * X_synthetic_1_array[:, 1]
        + epsilon_1
    )

    feature_names_1 = [f"x{i+1}" for i in range(n_features)]
    X_synthetic_1 = pd.DataFrame(
        X_synthetic_1_array,
        columns=feature_names_1,
    )
    y_synthetic_1 = pd.Series(y_synthetic_1_array, name="Target")
    synthetic_1_target_label = "Synthetic Dataset 1 Target"

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X_synthetic_1, y_synthetic_1,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_1 = X_train_1.to_numpy()
    X_test_1 = X_test_1.to_numpy()
    y_train_1 = y_train_1.to_numpy()
    y_test_1 = y_test_1.to_numpy()
    X_explain_1 = X_test_1[:n_samples_test]

# SYNTHETIC DATASET 1 - XGBOOST

if RUN_BASELINE and RUN_SYNTHETIC_1:
    xgb_regressor_1 = make_model("XGBoost")
    xgb_regressor_1.fit(X_train_1, y_train_1)

# SYNTHETIC DATASET 1 - XGBOOST SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    xgboost_shap_1 = make_shap_explanations(
        xgb_regressor_1, X_train_1, X_explain_1,
    )

# Plot: XGBoost Synthetic Dataset 1 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgboost_shap_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("XGBoost - Synthetic Dataset 1 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Synthetic Dataset 1 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgboost_shap_1,
        data=X_explain_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("XGBoost - Synthetic Dataset 1 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 1 - TABICL

if RUN_BASELINE and RUN_SYNTHETIC_1:
    tabicl_regressor_1 = make_model("TabICL")
    tabicl_regressor_1.fit(X_train_1, y_train_1)

# SYNTHETIC DATASET 1 - TABICL SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    tabicl_shap_1 = make_shap_explanations(
        tabicl_regressor_1, X_train_1, X_explain_1,
    )

# Plot: TabICL Synthetic Dataset 1 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("TabICL - Synthetic Dataset 1 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Synthetic Dataset 1 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_1,
        data=X_explain_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("TabICL - Synthetic Dataset 1 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 1 - TABDPT

if RUN_BASELINE and RUN_SYNTHETIC_1:
    tabdpt_regressor_1 = make_model("TabDPT")
    tabdpt_regressor_1.fit(X_train_1, y_train_1)

# SYNTHETIC DATASET 1 - TABDPT SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    tabdpt_shap_1 = make_shap_explanations(
        tabdpt_regressor_1, X_train_1, X_explain_1,
    )

# Plot: TabDPT Synthetic Dataset 1 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("TabDPT - Synthetic Dataset 1 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Synthetic Dataset 1 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_1,
        data=X_explain_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("TabDPT - Synthetic Dataset 1 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 1 - KUMO

if RUN_BASELINE and RUN_SYNTHETIC_1:
    kumo_regressor_1 = make_model("Kumo")
    kumo_regressor_1.fit(X_train_1, y_train_1)

# SYNTHETIC DATASET 1 - KUMO SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    kumo_shap_1 = make_shap_explanations(
        kumo_regressor_1, X_train_1, X_explain_1,
    )

# Plot: Kumo Synthetic Dataset 1 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("Kumo - Synthetic Dataset 1 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Synthetic Dataset 1 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_1,
        data=X_explain_1,
        feature_names=feature_names_1,
        show=False,
    )
    plt.title("Kumo - Synthetic Dataset 1 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 1 MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_R2_PLOTS:
    r2_results_1 = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_1,
            "TabICL": tabicl_regressor_1,
            "TabDPT": tabdpt_regressor_1,
            "Kumo": kumo_regressor_1,
        },
        first_split=(X_train_1, X_test_1, y_train_1, y_test_1),
        X_base=X_synthetic_1_array,
        simulation_number=1,
        kind="synthetic",
        base_seed=synthetic_1_seed,
        dataset_label="Synthetic Dataset 1",
    )

# Plot: Synthetic Dataset 1 Averaged R^2 Comparison

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_R2_PLOTS:
    plot_average_r2(r2_results_1, "Synthetic Dataset 1")

# SYNTHETIC DATASET 2
# y_i = sin(x_i1) + x_i2^2 + x_i3 + I{x_i4 > 0}
#       + x_i5 + x_i6 + x_i5*x_i6 + epsilon_i
# epsilon_i ~ N(0, 1)
# x_i6 is binary; all other x_ij ~ N(0, 1)

if RUN_BASELINE and RUN_SYNTHETIC_2:
    rng_2 = np.random.default_rng(synthetic_2_seed)
    X_synthetic_2_array = rng_2.normal(
        loc=feature_mean,
        scale=feature_std,
        size=(n_samples, n_features),
    )

    X_synthetic_2_array[:, 5] = rng_2.integers(
        low=0,
        high=2,
        size=n_samples,
    )

    epsilon_2 = rng_2.normal(
        loc=0.0,
        scale=target_noise_std,
        size=n_samples,
    )
    y_synthetic_2_array = build_simulation_signal(X_synthetic_2_array, 2) + epsilon_2
    feature_names_2 = [f"x{i+1}" for i in range(n_features)]
    X_synthetic_2 = pd.DataFrame(
        X_synthetic_2_array, columns=feature_names_2,
    )
    y_synthetic_2 = pd.Series(y_synthetic_2_array, name="Target")
    synthetic_2_target_label = "Synthetic Dataset 2 Target"

    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X_synthetic_2, y_synthetic_2,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_2 = X_train_2.to_numpy()
    X_test_2 = X_test_2.to_numpy()
    y_train_2 = y_train_2.to_numpy()
    y_test_2 = y_test_2.to_numpy()
    X_explain_2 = X_test_2[:n_samples_test]

# SYNTHETIC DATASET 2 - XGBOOST

if RUN_BASELINE and RUN_SYNTHETIC_2:
    xgb_regressor_2 = make_model("XGBoost")
    xgb_regressor_2.fit(X_train_2, y_train_2)

# SYNTHETIC DATASET 2 - XGBOOST SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    xgboost_shap_2 = make_shap_explanations(
        xgb_regressor_2, X_train_2, X_explain_2,
    )

# Plot: XGBoost Synthetic Dataset 2 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgboost_shap_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("XGBoost - Synthetic Dataset 2 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Synthetic Dataset 2 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgboost_shap_2,
        data=X_explain_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("XGBoost - Synthetic Dataset 2 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 2 - TABICL

if RUN_BASELINE and RUN_SYNTHETIC_2:
    tabicl_regressor_2 = make_model("TabICL")
    tabicl_regressor_2.fit(X_train_2, y_train_2)

# SYNTHETIC DATASET 2 - TABICL SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    tabicl_shap_2 = make_shap_explanations(
        tabicl_regressor_2, X_train_2, X_explain_2,
    )

# Plot: TabICL Synthetic Dataset 2 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("TabICL - Synthetic Dataset 2 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Synthetic Dataset 2 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_2,
        data=X_explain_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("TabICL - Synthetic Dataset 2 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 2 - TABDPT

if RUN_BASELINE and RUN_SYNTHETIC_2:
    tabdpt_regressor_2 = make_model("TabDPT")
    tabdpt_regressor_2.fit(X_train_2, y_train_2)

# SYNTHETIC DATASET 2 - TABDPT SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    tabdpt_shap_2 = make_shap_explanations(
        tabdpt_regressor_2, X_train_2, X_explain_2,
    )

# Plot: TabDPT Synthetic Dataset 2 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("TabDPT - Synthetic Dataset 2 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Synthetic Dataset 2 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_2,
        data=X_explain_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("TabDPT - Synthetic Dataset 2 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 2 - KUMO

if RUN_BASELINE and RUN_SYNTHETIC_2:
    kumo_regressor_2 = make_model("Kumo")
    kumo_regressor_2.fit(X_train_2, y_train_2)

# SYNTHETIC DATASET 2 - KUMO SHAP EXPLANATIONS (REPLICATE 1 ONLY)

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    kumo_shap_2 = make_shap_explanations(
        kumo_regressor_2, X_train_2, X_explain_2,
    )

# Plot: Kumo Synthetic Dataset 2 Bar Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("Kumo - Synthetic Dataset 2 - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Synthetic Dataset 2 Beeswarm Plot

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_2,
        data=X_explain_2,
        feature_names=feature_names_2,
        show=False,
    )
    plt.title("Kumo - Synthetic Dataset 2 - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# SYNTHETIC DATASET 2 MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_R2_PLOTS:
    r2_results_2 = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_2,
            "TabICL": tabicl_regressor_2,
            "TabDPT": tabdpt_regressor_2,
            "Kumo": kumo_regressor_2,
        },
        first_split=(X_train_2, X_test_2, y_train_2, y_test_2),
        X_base=X_synthetic_2_array,
        simulation_number=2,
        kind="synthetic",
        base_seed=synthetic_2_seed,
        dataset_label="Synthetic Dataset 2",
    )

# Plot: Synthetic Dataset 2 Averaged R^2 Comparison

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_R2_PLOTS:
    plot_average_r2(r2_results_2, "Synthetic Dataset 2")

# INTERPRETABILITY GRAPHS - SYNTHETIC DATASETS

# LOCO GRAPHS - SYNTHETIC DATASETS

# Plot: TabICL Synthetic Dataset 1 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_1, y_train_1, X_test_1, y_test_1,
        feature_names_1,
        "regression",
        "TabICL - Synthetic Dataset 1 - LOCO",
    )

# Plot: XGBoost Synthetic Dataset 1 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_1, y_train_1, X_test_1, y_test_1,
        feature_names_1,
        "regression",
        "XGBoost - Synthetic Dataset 1 - LOCO",
    )

# Plot: TabDPT Synthetic Dataset 1 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_1, y_train_1, X_test_1, y_test_1,
        feature_names_1,
        "regression",
        "TabDPT - Synthetic Dataset 1 - LOCO",
    )

# Plot: Kumo Synthetic Dataset 1 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_1, y_train_1, X_test_1, y_test_1,
        feature_names_1,
        "regression",
        "Kumo - Synthetic Dataset 1 - LOCO",
    )

# Plot: TabICL Synthetic Dataset 2 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_2, y_train_2, X_test_2, y_test_2,
        feature_names_2,
        "regression",
        "TabICL - Synthetic Dataset 2 - LOCO",
    )

# Plot: XGBoost Synthetic Dataset 2 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_2, y_train_2, X_test_2, y_test_2,
        feature_names_2,
        "regression",
        "XGBoost - Synthetic Dataset 2 - LOCO",
    )

# Plot: TabDPT Synthetic Dataset 2 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_2, y_train_2, X_test_2, y_test_2,
        feature_names_2,
        "regression",
        "TabDPT - Synthetic Dataset 2 - LOCO",
    )

# Plot: Kumo Synthetic Dataset 2 LOCO

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_2, y_train_2, X_test_2, y_test_2,
        feature_names_2,
        "regression",
        "Kumo - Synthetic Dataset 2 - LOCO",
    )

# ICE GRAPHS - SYNTHETIC DATASETS

# ICE Graphs - TabICL, Synthetic Dataset 1

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_ICE:
    for feature_name in SYNTHETIC_1_ICE_FEATURES:
        plot_ice(
            tabicl_regressor_1,
            X_test_1,
            feature_names_1.index(feature_name),
            feature_names_1,
            "regression",
            f"TabICL - Synthetic Dataset 1 - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Synthetic Dataset 1

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_ICE:
    for feature_name in SYNTHETIC_1_ICE_FEATURES:
        plot_ice(
            xgb_regressor_1,
            X_test_1,
            feature_names_1.index(feature_name),
            feature_names_1,
            "regression",
            f"XGBoost - Synthetic Dataset 1 - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Synthetic Dataset 1

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_ICE:
    for feature_name in SYNTHETIC_1_ICE_FEATURES:
        plot_ice(
            tabdpt_regressor_1,
            X_test_1,
            feature_names_1.index(feature_name),
            feature_names_1,
            "regression",
            f"TabDPT - Synthetic Dataset 1 - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Synthetic Dataset 1

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_ICE:
    for feature_name in SYNTHETIC_1_ICE_FEATURES:
        plot_ice(
            kumo_regressor_1,
            X_test_1,
            feature_names_1.index(feature_name),
            feature_names_1,
            "regression",
            f"Kumo - Synthetic Dataset 1 - ICE - {feature_name}",
        )

# ICE Graphs - TabICL, Synthetic Dataset 2

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_ICE:
    for feature_name in SYNTHETIC_2_ICE_FEATURES:
        plot_ice(
            tabicl_regressor_2,
            X_test_2,
            feature_names_2.index(feature_name),
            feature_names_2,
            "regression",
            f"TabICL - Synthetic Dataset 2 - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Synthetic Dataset 2

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_ICE:
    for feature_name in SYNTHETIC_2_ICE_FEATURES:
        plot_ice(
            xgb_regressor_2,
            X_test_2,
            feature_names_2.index(feature_name),
            feature_names_2,
            "regression",
            f"XGBoost - Synthetic Dataset 2 - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Synthetic Dataset 2

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_ICE:
    for feature_name in SYNTHETIC_2_ICE_FEATURES:
        plot_ice(
            tabdpt_regressor_2,
            X_test_2,
            feature_names_2.index(feature_name),
            feature_names_2,
            "regression",
            f"TabDPT - Synthetic Dataset 2 - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Synthetic Dataset 2

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_ICE:
    for feature_name in SYNTHETIC_2_ICE_FEATURES:
        plot_ice(
            kumo_regressor_2,
            X_test_2,
            feature_names_2.index(feature_name),
            feature_names_2,
            "regression",
            f"Kumo - Synthetic Dataset 2 - ICE - {feature_name}",
        )

# SENSITIVITY GRAPHS - SYNTHETIC DATASETS

# Plot: TabICL Synthetic Dataset 1 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_1,
        X_test_1, feature_names_1,
        "regression",
        "TabICL - Synthetic Dataset 1 - Sensitivity",
    )

# Plot: XGBoost Synthetic Dataset 1 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_1,
        X_test_1, feature_names_1,
        "regression",
        "XGBoost - Synthetic Dataset 1 - Sensitivity",
    )

# Plot: TabDPT Synthetic Dataset 1 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_1,
        X_test_1, feature_names_1,
        "regression",
        "TabDPT - Synthetic Dataset 1 - Sensitivity",
    )

# Plot: Kumo Synthetic Dataset 1 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_1 and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_1,
        X_test_1, feature_names_1,
        "regression",
        "Kumo - Synthetic Dataset 1 - Sensitivity",
    )

# Plot: TabICL Synthetic Dataset 2 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_2,
        X_test_2, feature_names_2,
        "regression",
        "TabICL - Synthetic Dataset 2 - Sensitivity",
    )

# Plot: XGBoost Synthetic Dataset 2 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_2,
        X_test_2, feature_names_2,
        "regression",
        "XGBoost - Synthetic Dataset 2 - Sensitivity",
    )

# Plot: TabDPT Synthetic Dataset 2 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_2,
        X_test_2, feature_names_2,
        "regression",
        "TabDPT - Synthetic Dataset 2 - Sensitivity",
    )

# Plot: Kumo Synthetic Dataset 2 Sensitivity

if RUN_BASELINE and RUN_SYNTHETIC_2 and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_2,
        X_test_2, feature_names_2,
        "regression",
        "Kumo - Synthetic Dataset 2 - Sensitivity",
    )

# BOSTON HOUSING CORRELATION EXPERIMENTS

if (RUN_BASELINE and RUN_BOSTON) or RUN_BOSTON_CORRELATION:
    boston_columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV",
    ]

    regression_feature_labels = {
        "CRIM": "Crime Rate",
        "ZN": "Large-Lot Residential Zoning",
        "INDUS": "Non-Retail Business Land",
        "CHAS": "Borders Charles River",
        "NOX": "Nitrogen Oxide Pollution",
        "RM": "Average Number of Rooms",
        "AGE": "Older Homes (%)",
        "DIS": "Distance to Employment Centers",
        "RAD": "Highway Accessibility",
        "TAX": "Property Tax Rate",
        "PTRATIO": "Student-Teacher Ratio",
        "B": "Black Population (%)",
        "LSTAT": "Lower-Income Population (%)",
        "MEDV": "Median Home Value (1000s of $)",
    }

    boston_url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/housing/housing.data"
    )

    boston_source = BOSTON_DATA_CACHE if BOSTON_DATA_CACHE.exists() else boston_url
    try:
        boston = pd.read_csv(
            boston_source,
            sep=r"\s+",
            header=None,
            names=boston_columns,
        )
    except Exception as error:
        raise RuntimeError(
            f"Boston data unavailable. Put housing.data at {BOSTON_DATA_CACHE}"
        ) from error
    if boston_source == boston_url:
        boston.to_csv(BOSTON_DATA_CACHE, sep=" ", header=False, index=False)

    X_boston = boston.drop(columns=["MEDV"]).copy()
    continuous_boston_features = [
        feature for feature in X_boston.columns if feature != "CHAS"
    ]
    boston_scaler = StandardScaler()
    X_boston_standardized = X_boston.copy()
    X_boston_standardized[continuous_boston_features] = (
        boston_scaler.fit_transform(X_boston[continuous_boston_features])
    )
    boston_correlation = X_boston_standardized.corr()

# BOSTON HOUSING CORRELATION MATRIX

if RUN_BOSTON_CORRELATION:
    boston_correlation_display = boston_correlation.rename(
        index=regression_feature_labels,
        columns=regression_feature_labels,
    )
    fig, ax = plt.subplots(figsize=(18, 14))
    correlation_image = ax.imshow(
        boston_correlation_display,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )
    ax.set_xticks(range(len(boston_correlation_display.columns)))
    ax.set_yticks(range(len(boston_correlation_display.index)))
    ax.set_xticklabels(
        boston_correlation_display.columns, rotation=75, ha="right",
    )
    ax.set_yticklabels(boston_correlation_display.index)
    for i in range(len(boston_correlation_display.index)):
        for j in range(len(boston_correlation_display.columns)):
            ax.text(
                j, i,
                f"{boston_correlation_display.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
            )
    fig.colorbar(correlation_image, ax=ax, label="Correlation")
    plt.title("Boston Housing Predictor Correlation Matrix")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON HOUSING - SIMULATION 1 CORRELATION SELECTION

if RUN_BASELINE and RUN_BOSTON:
    boston_pairs = get_correlation_pairs(boston_correlation)
    boston_high_pair = max(boston_pairs, key=lambda pair: pair[0])
    boston_low_pair = min(boston_pairs, key=lambda pair: pair[0])

    print("\nBoston Housing - Simulation 1 Correlation Selection")
    print(
        f"High-correlation pair: {boston_high_pair[2]} and {boston_high_pair[3]} "
        f"(correlation = {boston_high_pair[1]:.4f})"
    )
    print(
        f"Low-correlation pair: {boston_low_pair[2]} and {boston_low_pair[3]} "
        f"(correlation = {boston_low_pair[1]:.4f})"
    )

    X_boston_sim_1_high, y_boston_sim_1_high, mapping_boston_sim_1_high = (
        make_boston_simulation_1((boston_high_pair[2], boston_high_pair[3]))
    )
    X_boston_sim_1_low, y_boston_sim_1_low, mapping_boston_sim_1_low = (
        make_boston_simulation_1((boston_low_pair[2], boston_low_pair[3]))
    )
    print("Boston Sim 1 High mapping:", mapping_boston_sim_1_high)
    print("Boston Sim 1 Low mapping:", mapping_boston_sim_1_low)

# BOSTON HOUSING - SIMULATION 2 CORRELATION SELECTION

if RUN_BASELINE and RUN_BOSTON:
    boston_sim_2_candidates = [
        feature for feature in X_boston_standardized.columns
        if feature != "CHAS"
    ]
    boston_sim_2_sets = []
    for feature_combination in combinations(boston_sim_2_candidates, 5):
        feature_set = list(feature_combination) + ["CHAS"]
        boston_sim_2_sets.append(
            (average_absolute_correlation(feature_set), feature_set)
        )
    boston_sim_2_high = max(boston_sim_2_sets, key=lambda result: result[0])
    boston_sim_2_low = min(boston_sim_2_sets, key=lambda result: result[0])
    print("\nBoston Housing - Simulation 2 Correlation Selection")
    print("High average absolute correlation:", boston_sim_2_high)
    print("Low average absolute correlation:", boston_sim_2_low)

    X_boston_sim_2_high, y_boston_sim_2_high, mapping_boston_sim_2_high = (
        make_boston_simulation_2(boston_sim_2_high[1])
    )
    X_boston_sim_2_low, y_boston_sim_2_low, mapping_boston_sim_2_low = (
        make_boston_simulation_2(boston_sim_2_low[1])
    )
    print("Boston Sim 2 High mapping:", mapping_boston_sim_2_high)
    print("Boston Sim 2 Low mapping:", mapping_boston_sim_2_low)

# BOSTON SIMULATION 1 - HIGH CORRELATION - TRAIN / TEST SPLIT

if RUN_BASELINE and RUN_BOSTON:
    feature_names_boston_sim_1_high = list(X_boston_sim_1_high.columns)
    X_train_boston_sim_1_high, X_test_boston_sim_1_high, y_train_boston_sim_1_high, y_test_boston_sim_1_high = train_test_split(
        X_boston_sim_1_high, y_boston_sim_1_high,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_boston_sim_1_high = X_train_boston_sim_1_high.to_numpy()
    X_test_boston_sim_1_high = X_test_boston_sim_1_high.to_numpy()
    y_train_boston_sim_1_high = y_train_boston_sim_1_high.to_numpy()
    y_test_boston_sim_1_high = y_test_boston_sim_1_high.to_numpy()
    X_explain_boston_sim_1_high = X_test_boston_sim_1_high[:n_samples_test]

# BOSTON SIMULATION 1 - HIGH CORRELATION - XGBOOST

if RUN_BASELINE and RUN_BOSTON:
    xgb_regressor_boston_sim_1_high = make_model("XGBoost")
    xgb_regressor_boston_sim_1_high.fit(X_train_boston_sim_1_high, y_train_boston_sim_1_high)

# BOSTON SIMULATION 1 - HIGH CORRELATION - XGBOOST SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    xgb_shap_boston_sim_1_high = make_shap_explanations(
        xgb_regressor_boston_sim_1_high, X_train_boston_sim_1_high, X_explain_boston_sim_1_high,
    )

# Plot: XGBoost Boston Simulation 1 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgb_shap_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 1 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Boston Simulation 1 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgb_shap_boston_sim_1_high,
        data=X_explain_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 1 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - HIGH CORRELATION - TABICL

if RUN_BASELINE and RUN_BOSTON:
    tabicl_regressor_boston_sim_1_high = make_model("TabICL")
    tabicl_regressor_boston_sim_1_high.fit(X_train_boston_sim_1_high, y_train_boston_sim_1_high)

# BOSTON SIMULATION 1 - HIGH CORRELATION - TABICL SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabicl_shap_boston_sim_1_high = make_shap_explanations(
        tabicl_regressor_boston_sim_1_high, X_train_boston_sim_1_high, X_explain_boston_sim_1_high,
    )

# Plot: TabICL Boston Simulation 1 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 1 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Boston Simulation 1 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_boston_sim_1_high,
        data=X_explain_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 1 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - HIGH CORRELATION - TABDPT

if RUN_BASELINE and RUN_BOSTON:
    tabdpt_regressor_boston_sim_1_high = make_model("TabDPT")
    tabdpt_regressor_boston_sim_1_high.fit(X_train_boston_sim_1_high, y_train_boston_sim_1_high)

# BOSTON SIMULATION 1 - HIGH CORRELATION - TABDPT SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabdpt_shap_boston_sim_1_high = make_shap_explanations(
        tabdpt_regressor_boston_sim_1_high, X_train_boston_sim_1_high, X_explain_boston_sim_1_high,
    )

# Plot: TabDPT Boston Simulation 1 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 1 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Boston Simulation 1 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_boston_sim_1_high,
        data=X_explain_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 1 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - HIGH CORRELATION - KUMO

if RUN_BASELINE and RUN_BOSTON:
    kumo_regressor_boston_sim_1_high = make_model("Kumo")
    kumo_regressor_boston_sim_1_high.fit(X_train_boston_sim_1_high, y_train_boston_sim_1_high)

# BOSTON SIMULATION 1 - HIGH CORRELATION - KUMO SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    kumo_shap_boston_sim_1_high = make_shap_explanations(
        kumo_regressor_boston_sim_1_high, X_train_boston_sim_1_high, X_explain_boston_sim_1_high,
    )

# Plot: Kumo Boston Simulation 1 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 1 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Boston Simulation 1 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_boston_sim_1_high,
        data=X_explain_boston_sim_1_high,
        feature_names=feature_names_boston_sim_1_high,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 1 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - HIGH CORRELATION MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    r2_results_boston_sim_1_high = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_boston_sim_1_high,
            "TabICL": tabicl_regressor_boston_sim_1_high,
            "TabDPT": tabdpt_regressor_boston_sim_1_high,
            "Kumo": kumo_regressor_boston_sim_1_high,
        },
        first_split=(
            X_train_boston_sim_1_high, X_test_boston_sim_1_high,
            y_train_boston_sim_1_high, y_test_boston_sim_1_high,
        ),
        X_base=X_boston_sim_1_high.to_numpy(),
        simulation_number=1,
        kind="boston",
        base_seed=boston_sim_1_seed,
        dataset_label="Boston Simulation 1 - High Correlation",
    )

# Plot: Boston Simulation 1 - High Correlation Averaged R^2 Comparison

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    plot_average_r2(r2_results_boston_sim_1_high, "Boston Simulation 1 - High Correlation")

# Plot: TabICL Boston Simulation 1 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_boston_sim_1_high, y_train_boston_sim_1_high, X_test_boston_sim_1_high, y_test_boston_sim_1_high,
        feature_names_boston_sim_1_high, "regression",
        "TabICL - Boston Simulation 1 - High Correlation - LOCO",
    )

# Plot: XGBoost Boston Simulation 1 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_boston_sim_1_high, y_train_boston_sim_1_high, X_test_boston_sim_1_high, y_test_boston_sim_1_high,
        feature_names_boston_sim_1_high, "regression",
        "XGBoost - Boston Simulation 1 - High Correlation - LOCO",
    )

# Plot: TabDPT Boston Simulation 1 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_boston_sim_1_high, y_train_boston_sim_1_high, X_test_boston_sim_1_high, y_test_boston_sim_1_high,
        feature_names_boston_sim_1_high, "regression",
        "TabDPT - Boston Simulation 1 - High Correlation - LOCO",
    )

# Plot: Kumo Boston Simulation 1 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_boston_sim_1_high, y_train_boston_sim_1_high, X_test_boston_sim_1_high, y_test_boston_sim_1_high,
        feature_names_boston_sim_1_high, "regression",
        "Kumo - Boston Simulation 1 - High Correlation - LOCO",
    )

# ICE Graphs - TabICL, Boston Simulation 1 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            tabicl_regressor_boston_sim_1_high,
            X_test_boston_sim_1_high,
            feature_names_boston_sim_1_high.index(feature_name),
            feature_names_boston_sim_1_high, "regression",
            f"TabICL - Boston Simulation 1 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Boston Simulation 1 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            xgb_regressor_boston_sim_1_high,
            X_test_boston_sim_1_high,
            feature_names_boston_sim_1_high.index(feature_name),
            feature_names_boston_sim_1_high, "regression",
            f"XGBoost - Boston Simulation 1 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Boston Simulation 1 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            tabdpt_regressor_boston_sim_1_high,
            X_test_boston_sim_1_high,
            feature_names_boston_sim_1_high.index(feature_name),
            feature_names_boston_sim_1_high, "regression",
            f"TabDPT - Boston Simulation 1 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Boston Simulation 1 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            kumo_regressor_boston_sim_1_high,
            X_test_boston_sim_1_high,
            feature_names_boston_sim_1_high.index(feature_name),
            feature_names_boston_sim_1_high, "regression",
            f"Kumo - Boston Simulation 1 - High Correlation - ICE - {feature_name}",
        )

# Plot: TabICL Boston Simulation 1 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_boston_sim_1_high, X_test_boston_sim_1_high, feature_names_boston_sim_1_high,
        "regression", "TabICL - Boston Simulation 1 - High Correlation - Sensitivity",
    )

# Plot: XGBoost Boston Simulation 1 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_boston_sim_1_high, X_test_boston_sim_1_high, feature_names_boston_sim_1_high,
        "regression", "XGBoost - Boston Simulation 1 - High Correlation - Sensitivity",
    )

# Plot: TabDPT Boston Simulation 1 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_boston_sim_1_high, X_test_boston_sim_1_high, feature_names_boston_sim_1_high,
        "regression", "TabDPT - Boston Simulation 1 - High Correlation - Sensitivity",
    )

# Plot: Kumo Boston Simulation 1 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_boston_sim_1_high, X_test_boston_sim_1_high, feature_names_boston_sim_1_high,
        "regression", "Kumo - Boston Simulation 1 - High Correlation - Sensitivity",
    )

# BOSTON SIMULATION 1 - LOW CORRELATION - TRAIN / TEST SPLIT

if RUN_BASELINE and RUN_BOSTON:
    feature_names_boston_sim_1_low = list(X_boston_sim_1_low.columns)
    X_train_boston_sim_1_low, X_test_boston_sim_1_low, y_train_boston_sim_1_low, y_test_boston_sim_1_low = train_test_split(
        X_boston_sim_1_low, y_boston_sim_1_low,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_boston_sim_1_low = X_train_boston_sim_1_low.to_numpy()
    X_test_boston_sim_1_low = X_test_boston_sim_1_low.to_numpy()
    y_train_boston_sim_1_low = y_train_boston_sim_1_low.to_numpy()
    y_test_boston_sim_1_low = y_test_boston_sim_1_low.to_numpy()
    X_explain_boston_sim_1_low = X_test_boston_sim_1_low[:n_samples_test]

# BOSTON SIMULATION 1 - LOW CORRELATION - XGBOOST

if RUN_BASELINE and RUN_BOSTON:
    xgb_regressor_boston_sim_1_low = make_model("XGBoost")
    xgb_regressor_boston_sim_1_low.fit(X_train_boston_sim_1_low, y_train_boston_sim_1_low)

# BOSTON SIMULATION 1 - LOW CORRELATION - XGBOOST SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    xgb_shap_boston_sim_1_low = make_shap_explanations(
        xgb_regressor_boston_sim_1_low, X_train_boston_sim_1_low, X_explain_boston_sim_1_low,
    )

# Plot: XGBoost Boston Simulation 1 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgb_shap_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 1 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Boston Simulation 1 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgb_shap_boston_sim_1_low,
        data=X_explain_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 1 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - LOW CORRELATION - TABICL

if RUN_BASELINE and RUN_BOSTON:
    tabicl_regressor_boston_sim_1_low = make_model("TabICL")
    tabicl_regressor_boston_sim_1_low.fit(X_train_boston_sim_1_low, y_train_boston_sim_1_low)

# BOSTON SIMULATION 1 - LOW CORRELATION - TABICL SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabicl_shap_boston_sim_1_low = make_shap_explanations(
        tabicl_regressor_boston_sim_1_low, X_train_boston_sim_1_low, X_explain_boston_sim_1_low,
    )

# Plot: TabICL Boston Simulation 1 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 1 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Boston Simulation 1 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_boston_sim_1_low,
        data=X_explain_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 1 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - LOW CORRELATION - TABDPT

if RUN_BASELINE and RUN_BOSTON:
    tabdpt_regressor_boston_sim_1_low = make_model("TabDPT")
    tabdpt_regressor_boston_sim_1_low.fit(X_train_boston_sim_1_low, y_train_boston_sim_1_low)

# BOSTON SIMULATION 1 - LOW CORRELATION - TABDPT SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabdpt_shap_boston_sim_1_low = make_shap_explanations(
        tabdpt_regressor_boston_sim_1_low, X_train_boston_sim_1_low, X_explain_boston_sim_1_low,
    )

# Plot: TabDPT Boston Simulation 1 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 1 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Boston Simulation 1 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_boston_sim_1_low,
        data=X_explain_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 1 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - LOW CORRELATION - KUMO

if RUN_BASELINE and RUN_BOSTON:
    kumo_regressor_boston_sim_1_low = make_model("Kumo")
    kumo_regressor_boston_sim_1_low.fit(X_train_boston_sim_1_low, y_train_boston_sim_1_low)

# BOSTON SIMULATION 1 - LOW CORRELATION - KUMO SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    kumo_shap_boston_sim_1_low = make_shap_explanations(
        kumo_regressor_boston_sim_1_low, X_train_boston_sim_1_low, X_explain_boston_sim_1_low,
    )

# Plot: Kumo Boston Simulation 1 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 1 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Boston Simulation 1 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_boston_sim_1_low,
        data=X_explain_boston_sim_1_low,
        feature_names=feature_names_boston_sim_1_low,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 1 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 1 - LOW CORRELATION MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    r2_results_boston_sim_1_low = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_boston_sim_1_low,
            "TabICL": tabicl_regressor_boston_sim_1_low,
            "TabDPT": tabdpt_regressor_boston_sim_1_low,
            "Kumo": kumo_regressor_boston_sim_1_low,
        },
        first_split=(
            X_train_boston_sim_1_low, X_test_boston_sim_1_low,
            y_train_boston_sim_1_low, y_test_boston_sim_1_low,
        ),
        X_base=X_boston_sim_1_low.to_numpy(),
        simulation_number=1,
        kind="boston",
        base_seed=boston_sim_1_seed,
        dataset_label="Boston Simulation 1 - Low Correlation",
    )

# Plot: Boston Simulation 1 - Low Correlation Averaged R^2 Comparison

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    plot_average_r2(r2_results_boston_sim_1_low, "Boston Simulation 1 - Low Correlation")

# Plot: TabICL Boston Simulation 1 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_boston_sim_1_low, y_train_boston_sim_1_low, X_test_boston_sim_1_low, y_test_boston_sim_1_low,
        feature_names_boston_sim_1_low, "regression",
        "TabICL - Boston Simulation 1 - Low Correlation - LOCO",
    )

# Plot: XGBoost Boston Simulation 1 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_boston_sim_1_low, y_train_boston_sim_1_low, X_test_boston_sim_1_low, y_test_boston_sim_1_low,
        feature_names_boston_sim_1_low, "regression",
        "XGBoost - Boston Simulation 1 - Low Correlation - LOCO",
    )

# Plot: TabDPT Boston Simulation 1 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_boston_sim_1_low, y_train_boston_sim_1_low, X_test_boston_sim_1_low, y_test_boston_sim_1_low,
        feature_names_boston_sim_1_low, "regression",
        "TabDPT - Boston Simulation 1 - Low Correlation - LOCO",
    )

# Plot: Kumo Boston Simulation 1 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_boston_sim_1_low, y_train_boston_sim_1_low, X_test_boston_sim_1_low, y_test_boston_sim_1_low,
        feature_names_boston_sim_1_low, "regression",
        "Kumo - Boston Simulation 1 - Low Correlation - LOCO",
    )

# ICE Graphs - TabICL, Boston Simulation 1 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            tabicl_regressor_boston_sim_1_low,
            X_test_boston_sim_1_low,
            feature_names_boston_sim_1_low.index(feature_name),
            feature_names_boston_sim_1_low, "regression",
            f"TabICL - Boston Simulation 1 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Boston Simulation 1 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            xgb_regressor_boston_sim_1_low,
            X_test_boston_sim_1_low,
            feature_names_boston_sim_1_low.index(feature_name),
            feature_names_boston_sim_1_low, "regression",
            f"XGBoost - Boston Simulation 1 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Boston Simulation 1 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            tabdpt_regressor_boston_sim_1_low,
            X_test_boston_sim_1_low,
            feature_names_boston_sim_1_low.index(feature_name),
            feature_names_boston_sim_1_low, "regression",
            f"TabDPT - Boston Simulation 1 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Boston Simulation 1 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2']:
        plot_ice(
            kumo_regressor_boston_sim_1_low,
            X_test_boston_sim_1_low,
            feature_names_boston_sim_1_low.index(feature_name),
            feature_names_boston_sim_1_low, "regression",
            f"Kumo - Boston Simulation 1 - Low Correlation - ICE - {feature_name}",
        )

# Plot: TabICL Boston Simulation 1 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_boston_sim_1_low, X_test_boston_sim_1_low, feature_names_boston_sim_1_low,
        "regression", "TabICL - Boston Simulation 1 - Low Correlation - Sensitivity",
    )

# Plot: XGBoost Boston Simulation 1 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_boston_sim_1_low, X_test_boston_sim_1_low, feature_names_boston_sim_1_low,
        "regression", "XGBoost - Boston Simulation 1 - Low Correlation - Sensitivity",
    )

# Plot: TabDPT Boston Simulation 1 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_boston_sim_1_low, X_test_boston_sim_1_low, feature_names_boston_sim_1_low,
        "regression", "TabDPT - Boston Simulation 1 - Low Correlation - Sensitivity",
    )

# Plot: Kumo Boston Simulation 1 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_boston_sim_1_low, X_test_boston_sim_1_low, feature_names_boston_sim_1_low,
        "regression", "Kumo - Boston Simulation 1 - Low Correlation - Sensitivity",
    )

# BOSTON SIMULATION 2 - HIGH CORRELATION - TRAIN / TEST SPLIT

if RUN_BASELINE and RUN_BOSTON:
    feature_names_boston_sim_2_high = list(X_boston_sim_2_high.columns)
    X_train_boston_sim_2_high, X_test_boston_sim_2_high, y_train_boston_sim_2_high, y_test_boston_sim_2_high = train_test_split(
        X_boston_sim_2_high, y_boston_sim_2_high,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_boston_sim_2_high = X_train_boston_sim_2_high.to_numpy()
    X_test_boston_sim_2_high = X_test_boston_sim_2_high.to_numpy()
    y_train_boston_sim_2_high = y_train_boston_sim_2_high.to_numpy()
    y_test_boston_sim_2_high = y_test_boston_sim_2_high.to_numpy()
    X_explain_boston_sim_2_high = X_test_boston_sim_2_high[:n_samples_test]

# BOSTON SIMULATION 2 - HIGH CORRELATION - XGBOOST

if RUN_BASELINE and RUN_BOSTON:
    xgb_regressor_boston_sim_2_high = make_model("XGBoost")
    xgb_regressor_boston_sim_2_high.fit(X_train_boston_sim_2_high, y_train_boston_sim_2_high)

# BOSTON SIMULATION 2 - HIGH CORRELATION - XGBOOST SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    xgb_shap_boston_sim_2_high = make_shap_explanations(
        xgb_regressor_boston_sim_2_high, X_train_boston_sim_2_high, X_explain_boston_sim_2_high,
    )

# Plot: XGBoost Boston Simulation 2 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgb_shap_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 2 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Boston Simulation 2 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgb_shap_boston_sim_2_high,
        data=X_explain_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 2 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - HIGH CORRELATION - TABICL

if RUN_BASELINE and RUN_BOSTON:
    tabicl_regressor_boston_sim_2_high = make_model("TabICL")
    tabicl_regressor_boston_sim_2_high.fit(X_train_boston_sim_2_high, y_train_boston_sim_2_high)

# BOSTON SIMULATION 2 - HIGH CORRELATION - TABICL SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabicl_shap_boston_sim_2_high = make_shap_explanations(
        tabicl_regressor_boston_sim_2_high, X_train_boston_sim_2_high, X_explain_boston_sim_2_high,
    )

# Plot: TabICL Boston Simulation 2 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 2 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Boston Simulation 2 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_boston_sim_2_high,
        data=X_explain_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 2 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - HIGH CORRELATION - TABDPT

if RUN_BASELINE and RUN_BOSTON:
    tabdpt_regressor_boston_sim_2_high = make_model("TabDPT")
    tabdpt_regressor_boston_sim_2_high.fit(X_train_boston_sim_2_high, y_train_boston_sim_2_high)

# BOSTON SIMULATION 2 - HIGH CORRELATION - TABDPT SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabdpt_shap_boston_sim_2_high = make_shap_explanations(
        tabdpt_regressor_boston_sim_2_high, X_train_boston_sim_2_high, X_explain_boston_sim_2_high,
    )

# Plot: TabDPT Boston Simulation 2 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 2 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Boston Simulation 2 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_boston_sim_2_high,
        data=X_explain_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 2 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - HIGH CORRELATION - KUMO

if RUN_BASELINE and RUN_BOSTON:
    kumo_regressor_boston_sim_2_high = make_model("Kumo")
    kumo_regressor_boston_sim_2_high.fit(X_train_boston_sim_2_high, y_train_boston_sim_2_high)

# BOSTON SIMULATION 2 - HIGH CORRELATION - KUMO SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    kumo_shap_boston_sim_2_high = make_shap_explanations(
        kumo_regressor_boston_sim_2_high, X_train_boston_sim_2_high, X_explain_boston_sim_2_high,
    )

# Plot: Kumo Boston Simulation 2 - High Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 2 - High Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Boston Simulation 2 - High Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_boston_sim_2_high,
        data=X_explain_boston_sim_2_high,
        feature_names=feature_names_boston_sim_2_high,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 2 - High Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - HIGH CORRELATION MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    r2_results_boston_sim_2_high = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_boston_sim_2_high,
            "TabICL": tabicl_regressor_boston_sim_2_high,
            "TabDPT": tabdpt_regressor_boston_sim_2_high,
            "Kumo": kumo_regressor_boston_sim_2_high,
        },
        first_split=(
            X_train_boston_sim_2_high, X_test_boston_sim_2_high,
            y_train_boston_sim_2_high, y_test_boston_sim_2_high,
        ),
        X_base=X_boston_sim_2_high.to_numpy(),
        simulation_number=2,
        kind="boston",
        base_seed=boston_sim_2_seed,
        dataset_label="Boston Simulation 2 - High Correlation",
    )

# Plot: Boston Simulation 2 - High Correlation Averaged R^2 Comparison

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    plot_average_r2(r2_results_boston_sim_2_high, "Boston Simulation 2 - High Correlation")

# Plot: TabICL Boston Simulation 2 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_boston_sim_2_high, y_train_boston_sim_2_high, X_test_boston_sim_2_high, y_test_boston_sim_2_high,
        feature_names_boston_sim_2_high, "regression",
        "TabICL - Boston Simulation 2 - High Correlation - LOCO",
    )

# Plot: XGBoost Boston Simulation 2 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_boston_sim_2_high, y_train_boston_sim_2_high, X_test_boston_sim_2_high, y_test_boston_sim_2_high,
        feature_names_boston_sim_2_high, "regression",
        "XGBoost - Boston Simulation 2 - High Correlation - LOCO",
    )

# Plot: TabDPT Boston Simulation 2 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_boston_sim_2_high, y_train_boston_sim_2_high, X_test_boston_sim_2_high, y_test_boston_sim_2_high,
        feature_names_boston_sim_2_high, "regression",
        "TabDPT - Boston Simulation 2 - High Correlation - LOCO",
    )

# Plot: Kumo Boston Simulation 2 - High Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_boston_sim_2_high, y_train_boston_sim_2_high, X_test_boston_sim_2_high, y_test_boston_sim_2_high,
        feature_names_boston_sim_2_high, "regression",
        "Kumo - Boston Simulation 2 - High Correlation - LOCO",
    )

# ICE Graphs - TabICL, Boston Simulation 2 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            tabicl_regressor_boston_sim_2_high,
            X_test_boston_sim_2_high,
            feature_names_boston_sim_2_high.index(feature_name),
            feature_names_boston_sim_2_high, "regression",
            f"TabICL - Boston Simulation 2 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Boston Simulation 2 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            xgb_regressor_boston_sim_2_high,
            X_test_boston_sim_2_high,
            feature_names_boston_sim_2_high.index(feature_name),
            feature_names_boston_sim_2_high, "regression",
            f"XGBoost - Boston Simulation 2 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Boston Simulation 2 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            tabdpt_regressor_boston_sim_2_high,
            X_test_boston_sim_2_high,
            feature_names_boston_sim_2_high.index(feature_name),
            feature_names_boston_sim_2_high, "regression",
            f"TabDPT - Boston Simulation 2 - High Correlation - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Boston Simulation 2 - High Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            kumo_regressor_boston_sim_2_high,
            X_test_boston_sim_2_high,
            feature_names_boston_sim_2_high.index(feature_name),
            feature_names_boston_sim_2_high, "regression",
            f"Kumo - Boston Simulation 2 - High Correlation - ICE - {feature_name}",
        )

# Plot: TabICL Boston Simulation 2 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_boston_sim_2_high, X_test_boston_sim_2_high, feature_names_boston_sim_2_high,
        "regression", "TabICL - Boston Simulation 2 - High Correlation - Sensitivity",
    )

# Plot: XGBoost Boston Simulation 2 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_boston_sim_2_high, X_test_boston_sim_2_high, feature_names_boston_sim_2_high,
        "regression", "XGBoost - Boston Simulation 2 - High Correlation - Sensitivity",
    )

# Plot: TabDPT Boston Simulation 2 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_boston_sim_2_high, X_test_boston_sim_2_high, feature_names_boston_sim_2_high,
        "regression", "TabDPT - Boston Simulation 2 - High Correlation - Sensitivity",
    )

# Plot: Kumo Boston Simulation 2 - High Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_boston_sim_2_high, X_test_boston_sim_2_high, feature_names_boston_sim_2_high,
        "regression", "Kumo - Boston Simulation 2 - High Correlation - Sensitivity",
    )

# BOSTON SIMULATION 2 - LOW CORRELATION - TRAIN / TEST SPLIT

if RUN_BASELINE and RUN_BOSTON:
    feature_names_boston_sim_2_low = list(X_boston_sim_2_low.columns)
    X_train_boston_sim_2_low, X_test_boston_sim_2_low, y_train_boston_sim_2_low, y_test_boston_sim_2_low = train_test_split(
        X_boston_sim_2_low, y_boston_sim_2_low,
        test_size=test_fraction,
        random_state=split_seed,
    )
    X_train_boston_sim_2_low = X_train_boston_sim_2_low.to_numpy()
    X_test_boston_sim_2_low = X_test_boston_sim_2_low.to_numpy()
    y_train_boston_sim_2_low = y_train_boston_sim_2_low.to_numpy()
    y_test_boston_sim_2_low = y_test_boston_sim_2_low.to_numpy()
    X_explain_boston_sim_2_low = X_test_boston_sim_2_low[:n_samples_test]

# BOSTON SIMULATION 2 - LOW CORRELATION - XGBOOST

if RUN_BASELINE and RUN_BOSTON:
    xgb_regressor_boston_sim_2_low = make_model("XGBoost")
    xgb_regressor_boston_sim_2_low.fit(X_train_boston_sim_2_low, y_train_boston_sim_2_low)

# BOSTON SIMULATION 2 - LOW CORRELATION - XGBOOST SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    xgb_shap_boston_sim_2_low = make_shap_explanations(
        xgb_regressor_boston_sim_2_low, X_train_boston_sim_2_low, X_explain_boston_sim_2_low,
    )

# Plot: XGBoost Boston Simulation 2 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgb_shap_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 2 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: XGBoost Boston Simulation 2 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgb_shap_boston_sim_2_low,
        data=X_explain_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("XGBoost - Boston Simulation 2 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - LOW CORRELATION - TABICL

if RUN_BASELINE and RUN_BOSTON:
    tabicl_regressor_boston_sim_2_low = make_model("TabICL")
    tabicl_regressor_boston_sim_2_low.fit(X_train_boston_sim_2_low, y_train_boston_sim_2_low)

# BOSTON SIMULATION 2 - LOW CORRELATION - TABICL SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabicl_shap_boston_sim_2_low = make_shap_explanations(
        tabicl_regressor_boston_sim_2_low, X_train_boston_sim_2_low, X_explain_boston_sim_2_low,
    )

# Plot: TabICL Boston Simulation 2 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 2 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabICL Boston Simulation 2 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap_boston_sim_2_low,
        data=X_explain_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("TabICL - Boston Simulation 2 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - LOW CORRELATION - TABDPT

if RUN_BASELINE and RUN_BOSTON:
    tabdpt_regressor_boston_sim_2_low = make_model("TabDPT")
    tabdpt_regressor_boston_sim_2_low.fit(X_train_boston_sim_2_low, y_train_boston_sim_2_low)

# BOSTON SIMULATION 2 - LOW CORRELATION - TABDPT SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    tabdpt_shap_boston_sim_2_low = make_shap_explanations(
        tabdpt_regressor_boston_sim_2_low, X_train_boston_sim_2_low, X_explain_boston_sim_2_low,
    )

# Plot: TabDPT Boston Simulation 2 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabdpt_shap_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 2 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: TabDPT Boston Simulation 2 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabdpt_shap_boston_sim_2_low,
        data=X_explain_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("TabDPT - Boston Simulation 2 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - LOW CORRELATION - KUMO

if RUN_BASELINE and RUN_BOSTON:
    kumo_regressor_boston_sim_2_low = make_model("Kumo")
    kumo_regressor_boston_sim_2_low.fit(X_train_boston_sim_2_low, y_train_boston_sim_2_low)

# BOSTON SIMULATION 2 - LOW CORRELATION - KUMO SHAP (REPLICATE 1)

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    kumo_shap_boston_sim_2_low = make_shap_explanations(
        kumo_regressor_boston_sim_2_low, X_train_boston_sim_2_low, X_explain_boston_sim_2_low,
    )

# Plot: Kumo Boston Simulation 2 - Low Correlation Feature Importance

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        kumo_shap_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 2 - Low Correlation - Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()

# Plot: Kumo Boston Simulation 2 - Low Correlation Beeswarm

if RUN_BASELINE and RUN_BOSTON and RUN_SHAP_PLOTS:
    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        kumo_shap_boston_sim_2_low,
        data=X_explain_boston_sim_2_low,
        feature_names=feature_names_boston_sim_2_low,
        show=False,
    )
    plt.title("Kumo - Boston Simulation 2 - Low Correlation - Beeswarm")
    plt.tight_layout()
    plt.show()
    plt.close()

# BOSTON SIMULATION 2 - LOW CORRELATION MODEL PERFORMANCE COMPARISON

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    r2_results_boston_sim_2_low = calculate_mean_r2(
        first_models={
            "XGBoost": xgb_regressor_boston_sim_2_low,
            "TabICL": tabicl_regressor_boston_sim_2_low,
            "TabDPT": tabdpt_regressor_boston_sim_2_low,
            "Kumo": kumo_regressor_boston_sim_2_low,
        },
        first_split=(
            X_train_boston_sim_2_low, X_test_boston_sim_2_low,
            y_train_boston_sim_2_low, y_test_boston_sim_2_low,
        ),
        X_base=X_boston_sim_2_low.to_numpy(),
        simulation_number=2,
        kind="boston",
        base_seed=boston_sim_2_seed,
        dataset_label="Boston Simulation 2 - Low Correlation",
    )

# Plot: Boston Simulation 2 - Low Correlation Averaged R^2 Comparison

if RUN_BASELINE and RUN_BOSTON and RUN_R2_PLOTS:
    plot_average_r2(r2_results_boston_sim_2_low, "Boston Simulation 2 - Low Correlation")

# Plot: TabICL Boston Simulation 2 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabICL"),
        X_train_boston_sim_2_low, y_train_boston_sim_2_low, X_test_boston_sim_2_low, y_test_boston_sim_2_low,
        feature_names_boston_sim_2_low, "regression",
        "TabICL - Boston Simulation 2 - Low Correlation - LOCO",
    )

# Plot: XGBoost Boston Simulation 2 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("XGBoost"),
        X_train_boston_sim_2_low, y_train_boston_sim_2_low, X_test_boston_sim_2_low, y_test_boston_sim_2_low,
        feature_names_boston_sim_2_low, "regression",
        "XGBoost - Boston Simulation 2 - Low Correlation - LOCO",
    )

# Plot: TabDPT Boston Simulation 2 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("TabDPT"),
        X_train_boston_sim_2_low, y_train_boston_sim_2_low, X_test_boston_sim_2_low, y_test_boston_sim_2_low,
        feature_names_boston_sim_2_low, "regression",
        "TabDPT - Boston Simulation 2 - Low Correlation - LOCO",
    )

# Plot: Kumo Boston Simulation 2 - Low Correlation LOCO

if RUN_BASELINE and RUN_BOSTON and RUN_LOCO:
    plot_loco(
        lambda: make_model("Kumo"),
        X_train_boston_sim_2_low, y_train_boston_sim_2_low, X_test_boston_sim_2_low, y_test_boston_sim_2_low,
        feature_names_boston_sim_2_low, "regression",
        "Kumo - Boston Simulation 2 - Low Correlation - LOCO",
    )

# ICE Graphs - TabICL, Boston Simulation 2 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            tabicl_regressor_boston_sim_2_low,
            X_test_boston_sim_2_low,
            feature_names_boston_sim_2_low.index(feature_name),
            feature_names_boston_sim_2_low, "regression",
            f"TabICL - Boston Simulation 2 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - XGBoost, Boston Simulation 2 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            xgb_regressor_boston_sim_2_low,
            X_test_boston_sim_2_low,
            feature_names_boston_sim_2_low.index(feature_name),
            feature_names_boston_sim_2_low, "regression",
            f"XGBoost - Boston Simulation 2 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - TabDPT, Boston Simulation 2 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            tabdpt_regressor_boston_sim_2_low,
            X_test_boston_sim_2_low,
            feature_names_boston_sim_2_low.index(feature_name),
            feature_names_boston_sim_2_low, "regression",
            f"TabDPT - Boston Simulation 2 - Low Correlation - ICE - {feature_name}",
        )

# ICE Graphs - Kumo, Boston Simulation 2 - Low Correlation

if RUN_BASELINE and RUN_BOSTON and RUN_ICE:
    for feature_name in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
        plot_ice(
            kumo_regressor_boston_sim_2_low,
            X_test_boston_sim_2_low,
            feature_names_boston_sim_2_low.index(feature_name),
            feature_names_boston_sim_2_low, "regression",
            f"Kumo - Boston Simulation 2 - Low Correlation - ICE - {feature_name}",
        )

# Plot: TabICL Boston Simulation 2 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabicl_regressor_boston_sim_2_low, X_test_boston_sim_2_low, feature_names_boston_sim_2_low,
        "regression", "TabICL - Boston Simulation 2 - Low Correlation - Sensitivity",
    )

# Plot: XGBoost Boston Simulation 2 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        xgb_regressor_boston_sim_2_low, X_test_boston_sim_2_low, feature_names_boston_sim_2_low,
        "regression", "XGBoost - Boston Simulation 2 - Low Correlation - Sensitivity",
    )

# Plot: TabDPT Boston Simulation 2 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        tabdpt_regressor_boston_sim_2_low, X_test_boston_sim_2_low, feature_names_boston_sim_2_low,
        "regression", "TabDPT - Boston Simulation 2 - Low Correlation - Sensitivity",
    )

# Plot: Kumo Boston Simulation 2 - Low Correlation Sensitivity

if RUN_BASELINE and RUN_BOSTON and RUN_SENSITIVITY:
    plot_sensitivity(
        kumo_regressor_boston_sim_2_low, X_test_boston_sim_2_low, feature_names_boston_sim_2_low,
        "regression", "Kumo - Boston Simulation 2 - Low Correlation - Sensitivity",
    )

# NOISE VARIANCE / SHAP RANKING EXPERIMENTS

if RUN_VARIANCE_SWEEP:
    rng_variance_1 = np.random.default_rng(synthetic_1_seed)
    X_variance_sim_1 = rng_variance_1.normal(
        loc=feature_mean, scale=feature_std,
        size=(n_samples, n_features),
    )
    variance_results_1 = run_parameter_ranking_experiment(
        X_variance_sim_1,
        simulation_number=1,
        true_feature_ids=[0, 1],
        sweep_type="noise_variance",
    )

if RUN_VARIANCE_SWEEP:
    rng_variance_2 = np.random.default_rng(synthetic_2_seed)
    X_variance_sim_2 = rng_variance_2.normal(
        loc=feature_mean, scale=feature_std,
        size=(n_samples, n_features),
    )
    X_variance_sim_2[:, 5] = rng_variance_2.integers(
        low=0, high=2, size=n_samples,
    )
    variance_results_2 = run_parameter_ranking_experiment(
        X_variance_sim_2,
        simulation_number=2,
        true_feature_ids=[0, 1, 2, 3, 4, 5],
        sweep_type="noise_variance",
    )

# Plot: Synthetic Dataset 1 - Top-k Accuracy vs Noise Variance

if RUN_VARIANCE_SWEEP:
    plot_parameter_metric(
        variance_results_1,
        simulation_number=1,
        metric="top_k",
        sweep_type="noise_variance",
    )

# Plot: Synthetic Dataset 2 - Top-k Accuracy vs Noise Variance

if RUN_VARIANCE_SWEEP:
    plot_parameter_metric(
        variance_results_2,
        simulation_number=2,
        metric="top_k",
        sweep_type="noise_variance",
    )

# Plot: Synthetic Dataset 1 - Rank Sum vs Noise Variance

if RUN_VARIANCE_SWEEP:
    plot_parameter_metric(
        variance_results_1,
        simulation_number=1,
        metric="rank_sum",
        sweep_type="noise_variance",
    )

# Plot: Synthetic Dataset 2 - Rank Sum vs Noise Variance

if RUN_VARIANCE_SWEEP:
    plot_parameter_metric(
        variance_results_2,
        simulation_number=2,
        metric="rank_sum",
        sweep_type="noise_variance",
    )

# RHO / SHAP RANKING EXPERIMENTS

if RUN_RHO_SWEEP:
    rng_rho_1 = np.random.default_rng(synthetic_1_seed)
    X_rho_sim_1 = rng_rho_1.normal(
        loc=feature_mean, scale=feature_std,
        size=(n_samples, n_features),
    )
    rho_results_1 = run_parameter_ranking_experiment(
        X_rho_sim_1,
        simulation_number=1,
        true_feature_ids=[0, 1],
        sweep_type="rho",
    )

if RUN_RHO_SWEEP:
    rng_rho_2 = np.random.default_rng(synthetic_2_seed)
    X_rho_sim_2 = rng_rho_2.normal(
        loc=feature_mean, scale=feature_std,
        size=(n_samples, n_features),
    )
    X_rho_sim_2[:, 5] = rng_rho_2.integers(
        low=0, high=2, size=n_samples,
    )
    rho_results_2 = run_parameter_ranking_experiment(
        X_rho_sim_2,
        simulation_number=2,
        true_feature_ids=[0, 1, 2, 3, 4, 5],
        sweep_type="rho",
    )

# Plot: Synthetic Dataset 1 - Top-k Accuracy vs Rho

if RUN_RHO_SWEEP:
    plot_parameter_metric(
        rho_results_1,
        simulation_number=1,
        metric="top_k",
        sweep_type="rho",
    )

# Plot: Synthetic Dataset 2 - Top-k Accuracy vs Rho

if RUN_RHO_SWEEP:
    plot_parameter_metric(
        rho_results_2,
        simulation_number=2,
        metric="top_k",
        sweep_type="rho",
    )

# Plot: Synthetic Dataset 1 - Rank Sum vs Rho

if RUN_RHO_SWEEP:
    plot_parameter_metric(
        rho_results_1,
        simulation_number=1,
        metric="rank_sum",
        sweep_type="rho",
    )

# Plot: Synthetic Dataset 2 - Rank Sum vs Rho

if RUN_RHO_SWEEP:
    plot_parameter_metric(
        rho_results_2,
        simulation_number=2,
        metric="rank_sum",
        sweep_type="rho",
    )

print(f"Finished V7. Graphs and optional CSV files are in {OUTPUT_DIR}", flush=True)
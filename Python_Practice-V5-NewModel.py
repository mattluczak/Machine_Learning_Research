# IMPORTS

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapiq
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor
from xgboost import XGBRegressor

# AUTO SAVE ALL GRAPHS

OUTPUT_DIR = Path.home() / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

graph_number = 1
original_show = plt.show

def save_and_show(*args, **kwargs):
    global graph_number

    filename = OUTPUT_DIR / f"Graph_{graph_number:02d}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {filename}")

    graph_number += 1
    original_show(*args, **kwargs)

plt.show = save_and_show

# SETTINGS

n_samples_test = 2
shap_budget = 16
shap_background_size = None

n_samples = 500
n_features = 10

tabicl_n_estimators = 8

def make_tabicl_regressor():
    return TabICLRegressor(
        n_estimators=tabicl_n_estimators,
        kv_cache=False,
    )

# SYNTHETIC DATASET 1
# y_i = 2*x_i1 + x_i2 + epsilon_i
# epsilon_i ~ N(0, 1)
# All x_ij ~ N(0, 1)

rng_1 = np.random.default_rng(42)

X_synthetic_1_array = rng_1.normal(
    loc=0.0,
    scale=1.0,
    size=(n_samples, n_features),
)

epsilon_1 = rng_1.normal(
    loc=0.0,
    scale=1.0,
    size=n_samples,
)

y_synthetic_1_array = (
    2 * X_synthetic_1_array[:, 0]
    + X_synthetic_1_array[:, 1]
    + epsilon_1
)

feature_names_synthetic_1 = [
    f"x{i + 1}" for i in range(n_features)
]

X_synthetic_1 = pd.DataFrame(
    X_synthetic_1_array,
    columns=feature_names_synthetic_1,
)

y_synthetic_1 = pd.Series(
    y_synthetic_1_array,
    name="Target",
)

synthetic_1_target_label = "Synthetic Dataset 1 Target"

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_synthetic_1,
    y_synthetic_1,
    test_size=0.25,
    random_state=42,
)

X_train_1 = X_train_1.to_numpy()
X_test_1 = X_test_1.to_numpy()
y_train_1 = y_train_1.to_numpy()

X_explain_1 = X_test_1[:n_samples_test]

# SYNTHETIC DATASET 1 - XGBOOST

xgb_regressor_1 = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
)

xgb_regressor_1.fit(
    X_train_1,
    y_train_1,
)


def xgb_synthetic_1_predict(X):
    return xgb_regressor_1.predict(X)


xgb_explainer_1 = shapiq.TabularExplainer(
    model=xgb_synthetic_1_predict,
    data=X_train_1,
    index="SV",
    max_order=1,
    random_state=42,
    sample_size=shap_background_size,
)

xgb_shap_1 = [
    xgb_explainer_1.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_1
]


# Plot 1: XGBoost Synthetic Dataset 1 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    xgb_shap_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("XGBoost - Synthetic Dataset 1 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 2: XGBoost Synthetic Dataset 1 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    xgb_shap_1,
    data=X_explain_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("XGBoost - Synthetic Dataset 1 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 1 - TABPFN

tabicl_regressor_1 = make_tabicl_regressor()

tabicl_regressor_1.fit(
    X_train_1,
    y_train_1,
)

print(f"TabICL device: {tabicl_regressor_1.device_}")
print(f"TabICL ensemble estimators: {tabicl_n_estimators}")


def tabicl_synthetic_1_predict(X):
    return tabicl_regressor_1.predict(X)


tabicl_explainer_1 = shapiq.TabularExplainer(
    model=tabicl_synthetic_1_predict,
    data=X_train_1,
    index="SV",
    max_order=1,
    random_state=42,
    sample_size=shap_background_size,
)

tabicl_shap_1 = [
    tabicl_explainer_1.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_1
]


# Plot 3: TabICL Synthetic Dataset 1 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabicl_shap_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("TabICL - Synthetic Dataset 1 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 4: TabICL Synthetic Dataset 1 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabicl_shap_1,
    data=X_explain_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("TabICL - Synthetic Dataset 1 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 1 MODEL PERFORMANCE COMPARISON

xgb_predictions_1 = xgb_regressor_1.predict(X_test_1)
tabicl_predictions_1 = tabicl_regressor_1.predict(X_test_1)

xgb_r2_1 = r2_score(y_test_1, xgb_predictions_1)
tabicl_r2_1 = r2_score(y_test_1, tabicl_predictions_1)

print(f"Synthetic Dataset 1 - XGBoost R^2: {xgb_r2_1:.4f}")
print(f"Synthetic Dataset 1 - TabICL R^2: {tabicl_r2_1:.4f}")

# Plot 5: Synthetic Dataset 1 R^2 Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabICL"],
    [xgb_r2_1, tabicl_r2_1],
)

plt.bar_label(
    bars,
    fmt="%.4f",
    padding=3,
)

plt.ylim(0.0, 1.0)
plt.ylabel("R^2")
plt.title(f"{synthetic_1_target_label} - R^2 Comparison")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 2
# y_i = sin(x_i1) + x_i2^2 + x_i3 + I{x_i4 > 0}
#       + x_i5 + x_i6 + x_i5*x_i6 + epsilon_i
# epsilon_i ~ N(0, 1)
# x_i6 is binary; all other x_ij ~ N(0, 1)

rng_2 = np.random.default_rng(84)

X_synthetic_2_array = rng_2.normal(
    loc=0.0,
    scale=1.0,
    size=(n_samples, n_features),
)

# x6 is binary so the x5*x6 interaction can be inspected directly.
X_synthetic_2_array[:, 5] = rng_2.integers(
    low=0,
    high=2,
    size=n_samples,
)

epsilon_2 = rng_2.normal(
    loc=0.0,
    scale=1.0,
    size=n_samples,
)

indicator_2 = (
    X_synthetic_2_array[:, 3] > 0
).astype(float)

y_synthetic_2_array = (
    np.sin(X_synthetic_2_array[:, 0])
    + X_synthetic_2_array[:, 1] ** 2
    + X_synthetic_2_array[:, 2]
    + indicator_2
    + X_synthetic_2_array[:, 4]
    + X_synthetic_2_array[:, 5]
    + X_synthetic_2_array[:, 4] * X_synthetic_2_array[:, 5]
    + epsilon_2
)

feature_names_synthetic_2 = [
    f"x{i + 1}" for i in range(n_features)
]

X_synthetic_2 = pd.DataFrame(
    X_synthetic_2_array,
    columns=feature_names_synthetic_2,
)

y_synthetic_2 = pd.Series(
    y_synthetic_2_array,
    name="Target",
)

synthetic_2_target_label = "Synthetic Dataset 2 Target"

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_synthetic_2,
    y_synthetic_2,
    test_size=0.25,
    random_state=42,
)

X_train_2 = X_train_2.to_numpy()
X_test_2 = X_test_2.to_numpy()
y_train_2 = y_train_2.to_numpy()

X_explain_2 = X_test_2[:n_samples_test]

# SYNTHETIC DATASET 2 - XGBOOST

xgb_regressor_2 = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
)

xgb_regressor_2.fit(
    X_train_2,
    y_train_2,
)


def xgb_synthetic_2_predict(X):
    return xgb_regressor_2.predict(X)


xgb_explainer_2 = shapiq.TabularExplainer(
    model=xgb_synthetic_2_predict,
    data=X_train_2,
    index="SV",
    max_order=1,
    random_state=42,
    sample_size=shap_background_size,
)

xgb_shap_2 = [
    xgb_explainer_2.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_2
]


# Plot 6: XGBoost Synthetic Dataset 2 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    xgb_shap_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("XGBoost - Synthetic Dataset 2 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 7: XGBoost Synthetic Dataset 2 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    xgb_shap_2,
    data=X_explain_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("XGBoost - Synthetic Dataset 2 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 2 - TABPFN

tabicl_regressor_2 = make_tabicl_regressor()

tabicl_regressor_2.fit(
    X_train_2,
    y_train_2,
)


def tabicl_synthetic_2_predict(X):
    return tabicl_regressor_2.predict(X)


tabicl_explainer_2 = shapiq.TabularExplainer(
    model=tabicl_synthetic_2_predict,
    data=X_train_2,
    index="SV",
    max_order=1,
    random_state=42,
    sample_size=shap_background_size,
)

tabicl_shap_2 = [
    tabicl_explainer_2.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_2
]


# Plot 8: TabICL Synthetic Dataset 2 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabicl_shap_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("TabICL - Synthetic Dataset 2 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 9: TabICL Synthetic Dataset 2 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabicl_shap_2,
    data=X_explain_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("TabICL - Synthetic Dataset 2 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 2 MODEL PERFORMANCE COMPARISON

xgb_predictions_2 = xgb_regressor_2.predict(X_test_2)
tabicl_predictions_2 = tabicl_regressor_2.predict(X_test_2)

xgb_r2_2 = r2_score(y_test_2, xgb_predictions_2)
tabicl_r2_2 = r2_score(y_test_2, tabicl_predictions_2)

print(f"Synthetic Dataset 2 - XGBoost R^2: {xgb_r2_2:.4f}")
print(f"Synthetic Dataset 2 - TabICL R^2: {tabicl_r2_2:.4f}")

# Plot 10: Synthetic Dataset 2 R^2 Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabICL"],
    [xgb_r2_2, tabicl_r2_2],
)

plt.bar_label(
    bars,
    fmt="%.4f",
    padding=3,
)

plt.ylim(0.0, 1.0)
plt.ylabel("R^2")
plt.title(f"{synthetic_2_target_label} - R^2 Comparison")
plt.tight_layout()
plt.show()
plt.close()

# INTERPRETABILITY HELPER FUNCTIONS

def plot_loco(model_factory, X_train, y_train, X_test, y_test,
              feature_names, task, title):

    model = model_factory()
    model.fit(X_train, y_train)

    if task == "classification":
        baseline_loss = log_loss(y_test, model.predict_proba(X_test))
    else:
        baseline_loss = mean_squared_error(y_test, model.predict(X_test))

    scores = []

    for feature_id in range(X_train.shape[1]):

        X_train_reduced = np.delete(X_train, feature_id, axis=1)
        X_test_reduced = np.delete(X_test, feature_id, axis=1)

        reduced_model = model_factory()
        reduced_model.fit(X_train_reduced, y_train)

        if task == "classification":
            reduced_loss = log_loss(
                y_test,
                reduced_model.predict_proba(X_test_reduced)
            )
        else:
            reduced_loss = mean_squared_error(
                y_test,
                reduced_model.predict(X_test_reduced)
            )

        scores.append(reduced_loss - baseline_loss)

    scores = np.array(scores)
    order = np.argsort(scores)

    plt.figure(figsize=(10, 7))
    plt.barh(np.array(feature_names)[order], scores[order])
    plt.xlabel("Increase in Loss When Feature Is Removed")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()

    return int(np.argmax(scores))


def plot_ice(model, X, feature_id, feature_names, task, title,
             max_levels=10):

    if feature_id is None:
        print(f"{title} skipped: enter feature ID from LOCO plot.")
        return

    unique_values = np.unique(X[:, feature_id])

    if (
        len(unique_values) <= 2
        and set(unique_values).issubset({0.0, 1.0})
    ):
        feature_values = np.array([0.0, 1.0])
    else:
        feature_values = np.linspace(
            np.percentile(X[:, feature_id], 5),
            np.percentile(X[:, feature_id], 95),
            max_levels,
        )

    ice_predictions = []

    for value in feature_values:

        X_changed = X.copy()
        X_changed[:, feature_id] = value

        if task == "classification":
            predictions = model.predict_proba(X_changed)[:, 1]
        else:
            predictions = model.predict(X_changed)

        ice_predictions.append(predictions)

    ice_predictions = np.array(ice_predictions).T

    plt.figure(figsize=(9, 6))

    plt.plot(
        feature_values,
        ice_predictions.T,
        alpha=0.25,
        linewidth=1,
    )

    plt.plot(
        feature_values,
        ice_predictions.mean(axis=0),
        linewidth=3,
        label="Average",
    )

    plt.xlabel(feature_names[feature_id])
    plt.ylabel("Predicted Probability" if task == "classification"
               else "Predicted Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_sensitivity(model, X, feature_names, task, title,
                     perturbation=0.10):

    if task == "classification":
        baseline = model.predict_proba(X)[:, 1]
    else:
        baseline = model.predict(X)

    sensitivities = np.zeros((X.shape[0], X.shape[1]))

    for feature_id in range(X.shape[1]):

        step = perturbation * np.std(X[:, feature_id])

        if step == 0:
            continue

        X_plus = X.copy()
        X_minus = X.copy()

        X_plus[:, feature_id] += step
        X_minus[:, feature_id] -= step

        if task == "classification":
            plus = model.predict_proba(X_plus)[:, 1]
            minus = model.predict_proba(X_minus)[:, 1]
        else:
            plus = model.predict(X_plus)
            minus = model.predict(X_minus)

        sensitivities[:, feature_id] = (
            np.abs(plus - baseline)
            + np.abs(minus - baseline)
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

# Plot 11: LOCO Graph - TabICL, Synthetic Dataset 1

plot_loco(
    make_tabicl_regressor,
    X_train_1, y_train_1, X_test_1, y_test_1,
    feature_names_synthetic_1,
    "regression",
    "TabICL - Synthetic Dataset 1 - LOCO",
)

# Plot 12: LOCO Graph - TabICL, Synthetic Dataset 2

plot_loco(
    make_tabicl_regressor,
    X_train_2, y_train_2, X_test_2, y_test_2,
    feature_names_synthetic_2,
    "regression",
    "TabICL - Synthetic Dataset 2 - LOCO",
)

# Plot 13: LOCO Graph - XGBoost, Synthetic Dataset 1

plot_loco(
    lambda: clone(xgb_regressor_1),
    X_train_1, y_train_1, X_test_1, y_test_1,
    feature_names_synthetic_1,
    "regression",
    "XGBoost - Synthetic Dataset 1 - LOCO",
)

# Plot 14: LOCO Graph - XGBoost, Synthetic Dataset 2

plot_loco(
    lambda: clone(xgb_regressor_2),
    X_train_2, y_train_2, X_test_2, y_test_2,
    feature_names_synthetic_2,
    "regression",
    "XGBoost - Synthetic Dataset 2 - LOCO",
)

# ICE FEATURE DEFINITIONS

SYNTHETIC_1_ICE_FEATURES = ["x1", "x2"]
SYNTHETIC_2_ICE_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]

# ICE Graphs - TabICL, Synthetic Dataset 1

for feature_name in SYNTHETIC_1_ICE_FEATURES:
    plot_ice(
        tabicl_regressor_1,
        X_test_1,
        feature_names_synthetic_1.index(feature_name),
        feature_names_synthetic_1,
        "regression",
        f"TabICL - Synthetic Dataset 1 - ICE - {feature_name}",
    )

# ICE Graphs - TabICL, Synthetic Dataset 2

for feature_name in SYNTHETIC_2_ICE_FEATURES:
    plot_ice(
        tabicl_regressor_2,
        X_test_2,
        feature_names_synthetic_2.index(feature_name),
        feature_names_synthetic_2,
        "regression",
        f"TabICL - Synthetic Dataset 2 - ICE - {feature_name}",
    )

# ICE Graphs - XGBoost, Synthetic Dataset 1

for feature_name in SYNTHETIC_1_ICE_FEATURES:
    plot_ice(
        xgb_regressor_1,
        X_test_1,
        feature_names_synthetic_1.index(feature_name),
        feature_names_synthetic_1,
        "regression",
        f"XGBoost - Synthetic Dataset 1 - ICE - {feature_name}",
    )

# ICE Graphs - XGBoost, Synthetic Dataset 2

for feature_name in SYNTHETIC_2_ICE_FEATURES:
    plot_ice(
        xgb_regressor_2,
        X_test_2,
        feature_names_synthetic_2.index(feature_name),
        feature_names_synthetic_2,
        "regression",
        f"XGBoost - Synthetic Dataset 2 - ICE - {feature_name}",
    )

# Plot 19: Sensitivity Graph - TabICL, Synthetic Dataset 1

plot_sensitivity(
    tabicl_regressor_1,
    X_test_1,
    feature_names_synthetic_1,
    "regression",
    "TabICL - Synthetic Dataset 1 - Sensitivity",
)

# Plot 20: Sensitivity Graph - TabICL, Synthetic Dataset 2

plot_sensitivity(
    tabicl_regressor_2,
    X_test_2,
    feature_names_synthetic_2,
    "regression",
    "TabICL - Synthetic Dataset 2 - Sensitivity",
)

# Plot 21: Sensitivity Graph - XGBoost, Synthetic Dataset 1

plot_sensitivity(
    xgb_regressor_1,
    X_test_1,
    feature_names_synthetic_1,
    "regression",
    "XGBoost - Synthetic Dataset 1 - Sensitivity",
)

# Plot 22: Sensitivity Graph - XGBoost, Synthetic Dataset 2

plot_sensitivity(
    xgb_regressor_2,
    X_test_2,
    feature_names_synthetic_2,
    "regression",
    "XGBoost - Synthetic Dataset 2 - Sensitivity",
)

# BOSTON HOUSING CORRELATION EXPERIMENTS

boston_columns = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
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

boston = pd.read_csv(
    boston_url,
    sep=r"\s+",
    header=None,
    names=boston_columns,
)

X_boston = boston.drop(columns=["MEDV"]).copy()

continuous_boston_features = [
    feature for feature in X_boston.columns
    if feature != "CHAS"
]

boston_scaler = StandardScaler()

X_boston_standardized = X_boston.copy()
X_boston_standardized[continuous_boston_features] = (
    boston_scaler.fit_transform(
        X_boston[continuous_boston_features]
    )
)

boston_correlation = X_boston_standardized.corr()

# BOSTON HOUSING CORRELATION MATRIX PDF

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
    boston_correlation_display.columns,
    rotation=75,
    ha="right",
)

ax.set_yticklabels(
    boston_correlation_display.index,
)

for i in range(len(boston_correlation_display.index)):
    for j in range(len(boston_correlation_display.columns)):
        ax.text(
            j,
            i,
            f"{boston_correlation_display.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
        )

fig.colorbar(
    correlation_image,
    ax=ax,
    label="Correlation",
)

plt.title("Boston Housing Predictor Correlation Matrix")
plt.tight_layout()

plt.show()
plt.close()

def get_correlation_pairs(correlation_matrix):
    pairs = []
    feature_names = list(correlation_matrix.columns)

    for i, feature_1 in enumerate(feature_names):
        for feature_2 in feature_names[i + 1:]:
            correlation = correlation_matrix.loc[
                feature_1,
                feature_2,
            ]

            pairs.append(
                (
                    abs(correlation),
                    correlation,
                    feature_1,
                    feature_2,
                )
            )

    return pairs

boston_pairs = get_correlation_pairs(boston_correlation)

boston_high_pair = max(
    boston_pairs,
    key=lambda pair: pair[0],
)

boston_low_pair = min(
    boston_pairs,
    key=lambda pair: pair[0],
)

print("\nBoston Housing - Simulation 1 Correlation Selection")
print(
    "High-correlation pair: "
    f"{boston_high_pair[2]} and {boston_high_pair[3]} "
    f"(correlation = {boston_high_pair[1]:.4f})"
)
print(
    "Low-correlation pair: "
    f"{boston_low_pair[2]} and {boston_low_pair[3]} "
    f"(correlation = {boston_low_pair[1]:.4f})"
)

def make_boston_simulation_1(feature_pair, random_seed=142):
    selected_features = list(feature_pair)

    X = X_boston_standardized[
        selected_features
    ].copy()

    X.columns = ["x1", "x2"]

    rng = np.random.default_rng(random_seed)
    epsilon = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(X),
    )

    noise_features = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(X), 8),
    )

    for noise_id in range(8):
        X[f"x{noise_id + 3}"] = noise_features[:, noise_id]

    y = (
        2 * X["x1"].to_numpy()
        + X["x2"].to_numpy()
        + epsilon
    )

    mapping = {
        "x1": selected_features[0],
        "x2": selected_features[1],
    }

    mapping.update({
        f"x{feature_id}": "Noise"
        for feature_id in range(3, 11)
    })

    return X, pd.Series(y, name="Target"), mapping


X_boston_sim_1_high, y_boston_sim_1_high, mapping_boston_sim_1_high = (
    make_boston_simulation_1(
        (boston_high_pair[2], boston_high_pair[3])
    )
)

X_boston_sim_1_low, y_boston_sim_1_low, mapping_boston_sim_1_low = (
    make_boston_simulation_1(
        (boston_low_pair[2], boston_low_pair[3])
    )
)

print(
    "Boston Sim 1 High mapping: "
    f"x1 = {mapping_boston_sim_1_high['x1']}, "
    f"x2 = {mapping_boston_sim_1_high['x2']}"
)
print(
    "Boston Sim 1 Low mapping: "
    f"x1 = {mapping_boston_sim_1_low['x1']}, "
    f"x2 = {mapping_boston_sim_1_low['x2']}"
)


def average_absolute_correlation(feature_names):
    correlation_matrix = X_boston_standardized[
        feature_names
    ].corr()

    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1,
    )

    values = np.abs(
        correlation_matrix.to_numpy()[upper_triangle]
    )

    return float(values.mean())


boston_sim_2_candidates = [
    feature for feature in X_boston_standardized.columns
    if feature != "CHAS"
]

boston_sim_2_sets = []

for feature_combination in combinations(
    boston_sim_2_candidates,
    5,
):
    feature_set = list(feature_combination) + ["CHAS"]

    boston_sim_2_sets.append(
        (
            average_absolute_correlation(feature_set),
            feature_set,
        )
    )

boston_sim_2_high = max(
    boston_sim_2_sets,
    key=lambda result: result[0],
)

boston_sim_2_low = min(
    boston_sim_2_sets,
    key=lambda result: result[0],
)

print("\nBoston Housing - Simulation 2 Correlation Selection")
print(
    "High average absolute correlation: "
    f"{boston_sim_2_high[0]:.4f}"
)
print(
    "Features: "
    + ", ".join(boston_sim_2_high[1])
)
print(
    X_boston_standardized[
        boston_sim_2_high[1]
    ].corr().round(4)
)

print(
    "\nLow average absolute correlation: "
    f"{boston_sim_2_low[0]:.4f}"
)
print(
    "Features: "
    + ", ".join(boston_sim_2_low[1])
)
print(
    X_boston_standardized[
        boston_sim_2_low[1]
    ].corr().round(4)
)


def make_boston_simulation_2(feature_set, random_seed=184):
    selected_features = list(feature_set)

    X = X_boston_standardized[
        selected_features
    ].copy()

    X.columns = [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
    ]

    rng = np.random.default_rng(random_seed)
    epsilon = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(X),
    )

    noise_features = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(X), 4),
    )

    for noise_id in range(4):
        X[f"x{noise_id + 7}"] = noise_features[:, noise_id]

    indicator = (
        X["x4"].to_numpy() > 0
    ).astype(float)

    y = (
        np.sin(X["x1"].to_numpy())
        + X["x2"].to_numpy() ** 2
        + X["x3"].to_numpy()
        + indicator
        + X["x5"].to_numpy()
        + X["x6"].to_numpy()
        + X["x5"].to_numpy() * X["x6"].to_numpy()
        + epsilon
    )

    mapping = {
        f"x{i + 1}": feature
        for i, feature in enumerate(selected_features)
    }

    mapping.update({
        f"x{feature_id}": "Noise"
        for feature_id in range(7, 11)
    })

    return X, pd.Series(y, name="Target"), mapping


X_boston_sim_2_high, y_boston_sim_2_high, mapping_boston_sim_2_high = (
    make_boston_simulation_2(
        boston_sim_2_high[1]
    )
)

X_boston_sim_2_low, y_boston_sim_2_low, mapping_boston_sim_2_low = (
    make_boston_simulation_2(
        boston_sim_2_low[1]
    )
)

print("Boston Sim 2 High mapping:", mapping_boston_sim_2_high)
print("Boston Sim 2 Low mapping:", mapping_boston_sim_2_low)

def run_boston_regression_experiment(
    X,
    y,
    experiment_label,
    ice_features,
):
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()

    X_explain = X_test[:n_samples_test]

    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    xgb_model.fit(
        X_train,
        y_train,
    )

    def xgb_predict(X_values):
        return xgb_model.predict(X_values)

    xgb_explainer = shapiq.TabularExplainer(
        model=xgb_predict,
        data=X_train,
        index="SV",
        max_order=1,
        random_state=42,
        sample_size=shap_background_size,
    )

    xgb_shap = [
        xgb_explainer.explain(
            x=x,
            budget=shap_budget,
        )
        for x in X_explain
    ]

    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        xgb_shap,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"XGBoost - {experiment_label} - Feature Importance"
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        xgb_shap,
        data=X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"XGBoost - {experiment_label} - Beeswarm"
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    tabicl_model = make_tabicl_regressor()

    tabicl_model.fit(
        X_train,
        y_train,
    )

    def tabicl_predict(X_values):
        return tabicl_model.predict(X_values)

    tabicl_explainer = shapiq.TabularExplainer(
        model=tabicl_predict,
        data=X_train,
        index="SV",
        max_order=1,
        random_state=42,
        sample_size=shap_background_size,
    )

    tabicl_shap = [
        tabicl_explainer.explain(
            x=x,
            budget=shap_budget,
        )
        for x in X_explain
    ]

    plt.figure(figsize=(10, 6))
    shapiq.plot.bar_plot(
        tabicl_shap,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"TabICL - {experiment_label} - Feature Importance"
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    shapiq.plot.beeswarm_plot(
        tabicl_shap,
        data=X_explain,
        feature_names=feature_names,
        show=False,
    )
    plt.title(
        f"TabICL - {experiment_label} - Beeswarm"
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    xgb_predictions = xgb_model.predict(X_test)
    tabicl_predictions = tabicl_model.predict(X_test)

    xgb_r2 = r2_score(y_test, xgb_predictions)
    tabicl_r2 = r2_score(y_test, tabicl_predictions)

    print(
        f"{experiment_label} - XGBoost R^2: {xgb_r2:.4f}"
    )
    print(
        f"{experiment_label} - TabICL R^2: {tabicl_r2:.4f}"
    )

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        ["XGBoost", "TabICL"],
        [xgb_r2, tabicl_r2],
    )

    plt.bar_label(
        bars,
        fmt="%.4f",
        padding=3,
    )

    plt.ylim(0.0, 1.0)
    plt.ylabel("R^2")
    plt.title(
        f"{experiment_label} - R^2 Comparison"
    )
    plt.tight_layout()
    plt.show()
    plt.close()

    plot_loco(
        make_tabicl_regressor,
        X_train, y_train, X_test, y_test,
        feature_names,
        "regression",
        f"TabICL - {experiment_label} - LOCO",
    )

    plot_loco(
        lambda: clone(xgb_model),
        X_train, y_train, X_test, y_test,
        feature_names,
        "regression",
        f"XGBoost - {experiment_label} - LOCO",
    )

    for feature_name in ice_features:
        feature_id = feature_names.index(feature_name)

        plot_ice(
            tabicl_model,
            X_test,
            feature_id,
            feature_names,
            "regression",
            f"TabICL - {experiment_label} - ICE - {feature_name}",
        )

        plot_ice(
            xgb_model,
            X_test,
            feature_id,
            feature_names,
            "regression",
            f"XGBoost - {experiment_label} - ICE - {feature_name}",
        )

    plot_sensitivity(
        tabicl_model,
        X_test,
        feature_names,
        "regression",
        f"TabICL - {experiment_label} - Sensitivity",
    )

    plot_sensitivity(
        xgb_model,
        X_test,
        feature_names,
        "regression",
        f"XGBoost - {experiment_label} - Sensitivity",
    )


run_boston_regression_experiment(
    X_boston_sim_1_high,
    y_boston_sim_1_high,
    "Boston Simulation 1 - High Correlation",
    ["x1", "x2"],
)

run_boston_regression_experiment(
    X_boston_sim_1_low,
    y_boston_sim_1_low,
    "Boston Simulation 1 - Low Correlation",
    ["x1", "x2"],
)

run_boston_regression_experiment(
    X_boston_sim_2_high,
    y_boston_sim_2_high,
    "Boston Simulation 2 - High Correlation",
    ["x1", "x2", "x3", "x4", "x5", "x6"],
)

run_boston_regression_experiment(
    X_boston_sim_2_low,
    y_boston_sim_2_low,
    "Boston Simulation 2 - Low Correlation",
    ["x1", "x2", "x3", "x4", "x5", "x6"],
)

# NOISE VARIANCE / SHAP RANKING EXPERIMENTS

noise_variance_levels = [0.25, 0.50, 1.00, 2.00, 4.00]
rho = 0.0
n_replicates = 1

def apply_rho_to_existing_data(X, continuous_feature_ids, rho_value):
    X_correlated = X.copy()

    if rho_value == 0:
        return X_correlated

    n_continuous = len(continuous_feature_ids)

    minimum_rho = -1 / (n_continuous - 1)

    if not minimum_rho < rho_value < 1:
        raise ValueError(
            f"rho must be between {minimum_rho:.4f} and 1 "
            f"for {n_continuous} continuous features."
        )

    X_continuous = X_correlated[:, continuous_feature_ids].copy()

    X_continuous = (
        X_continuous - X_continuous.mean(axis=0)
    ) / X_continuous.std(axis=0, ddof=0)

    target_correlation = np.full(
        (n_continuous, n_continuous),
        rho_value,
    )

    np.fill_diagonal(target_correlation, 1.0)

    cholesky_factor = np.linalg.cholesky(
        target_correlation
    )

    X_correlated[:, continuous_feature_ids] = (
        X_continuous @ cholesky_factor.T
    )

    return X_correlated


def get_mean_absolute_shap_ranking(
    model,
    X_train,
    X_explain,
):
    def predict_function(X_values):
        return model.predict(X_values)

    explainer = shapiq.TabularExplainer(
        model=predict_function,
        data=X_train,
        index="SV",
        max_order=1,
        random_state=42,
        sample_size=shap_background_size,
    )

    explanations = [
        explainer.explain(
            x=x,
            budget=shap_budget,
        )
        for x in X_explain
    ]

    shap_values = np.vstack([
        explanation.get_n_order_values(1)
        for explanation in explanations
    ])

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    return np.argsort(-mean_absolute_shap)


def calculate_top_k_accuracy(
    ranking,
    true_feature_ids,
):
    k = len(true_feature_ids)

    top_k_features = set(ranking[:k])
    true_features = set(true_feature_ids)

    return (
        len(top_k_features.intersection(true_features))
        / k
    )


def calculate_rank_sum(
    ranking,
    true_feature_ids,
):
    rank_positions = np.empty(
        len(ranking),
        dtype=int,
    )

    rank_positions[ranking] = np.arange(
        1,
        len(ranking) + 1,
    )

    return float(
        sum(
            rank_positions[feature_id]
            for feature_id in true_feature_ids
        )
    )


def build_simulation_signal(
    X,
    simulation_number,
):
    if simulation_number == 1:
        return (
            2 * X[:, 0]
            + X[:, 1]
        )

    indicator = (
        X[:, 3] > 0
    ).astype(float)

    return (
        np.sin(X[:, 0])
        + X[:, 1] ** 2
        + X[:, 2]
        + indicator
        + X[:, 4]
        + X[:, 5]
        + X[:, 4] * X[:, 5]
    )


def run_noise_variance_experiment(
    X_base,
    simulation_number,
    true_feature_ids,
):
    if simulation_number == 1:
        continuous_feature_ids = list(
            range(X_base.shape[1])
        )
    else:
        continuous_feature_ids = [
            feature_id
            for feature_id in range(X_base.shape[1])
            if feature_id != 5
        ]

    X_experiment = apply_rho_to_existing_data(
        X_base,
        continuous_feature_ids,
        rho,
    )

    signal = build_simulation_signal(
        X_experiment,
        simulation_number,
    )

    sample_ids = np.arange(
        X_experiment.shape[0]
    )

    train_ids, test_ids = train_test_split(
        sample_ids,
        test_size=0.25,
        random_state=42,
    )

    X_train = X_experiment[train_ids]
    X_test = X_experiment[test_ids]
    X_explain = X_test[:n_samples_test]

    n_variance_levels = len(
        noise_variance_levels
    )

    xgb_top_k_scores = np.zeros(
        (n_replicates, n_variance_levels)
    )

    tabicl_top_k_scores = np.zeros(
        (n_replicates, n_variance_levels)
    )

    xgb_rank_sum_scores = np.zeros(
        (n_replicates, n_variance_levels)
    )

    tabicl_rank_sum_scores = np.zeros(
        (n_replicates, n_variance_levels)
    )

    for replicate_id in range(n_replicates):
        rng = np.random.default_rng(
            10000
            + 1000 * simulation_number
            + replicate_id
        )

        base_noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=X_experiment.shape[0],
        )

        for variance_id, noise_variance in enumerate(
            noise_variance_levels
        ):
            epsilon = (
                np.sqrt(noise_variance)
                * base_noise
            )

            y = signal + epsilon

            y_train = y[train_ids]

            xgb_model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )

            xgb_model.fit(
                X_train,
                y_train,
            )

            xgb_ranking = get_mean_absolute_shap_ranking(
                xgb_model,
                X_train,
                X_explain,
            )

            xgb_top_k_scores[
                replicate_id,
                variance_id,
            ] = calculate_top_k_accuracy(
                xgb_ranking,
                true_feature_ids,
            )

            xgb_rank_sum_scores[
                replicate_id,
                variance_id,
            ] = calculate_rank_sum(
                xgb_ranking,
                true_feature_ids,
            )

            tabicl_model = make_tabicl_regressor()

            tabicl_model.fit(
                X_train,
                y_train,
            )

            tabicl_ranking = get_mean_absolute_shap_ranking(
                tabicl_model,
                X_train,
                X_explain,
            )

            tabicl_top_k_scores[
                replicate_id,
                variance_id,
            ] = calculate_top_k_accuracy(
                tabicl_ranking,
                true_feature_ids,
            )

            tabicl_rank_sum_scores[
                replicate_id,
                variance_id,
            ] = calculate_rank_sum(
                tabicl_ranking,
                true_feature_ids,
            )

    return {
        "xgb_top_k": xgb_top_k_scores.mean(axis=0),
        "tabicl_top_k": tabicl_top_k_scores.mean(axis=0),
        "xgb_rank_sum": xgb_rank_sum_scores.mean(axis=0),
        "tabicl_rank_sum": tabicl_rank_sum_scores.mean(axis=0),
    }


def format_rank_sum_label(value):
    if np.isclose(value, round(value)):
        return str(int(round(value)))  # noqa: RUF046

    return f"{value:.2f}"


def add_coordinate_labels(
    x_values,
    y_values,
    metric,
    vertical_offset,
):
    for x_value, y_value in zip(
        x_values,
        y_values,
    ):
        if metric == "top_k":
            y_text = f"{y_value:.2f}"
        else:
            y_text = format_rank_sum_label(
                y_value
            )

        plt.annotate(
            f"({x_value:g}, {y_text})",
            xy=(x_value, y_value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )


def plot_variance_metric(
    results,
    simulation_number,
    metric,
):
    if metric == "top_k":
        xgb_values = results["xgb_top_k"]
        tabicl_values = results["tabicl_top_k"]

        y_label = "Top-k Accuracy (Higher = Better)"
        metric_title = "SHAP Top-k Accuracy"

    else:
        xgb_values = results["xgb_rank_sum"]
        tabicl_values = results["tabicl_rank_sum"]

        y_label = "Rank Sum (Lower = Better)"
        metric_title = "SHAP Rank Sum"

    plt.figure(figsize=(10, 6))

    plt.plot(
        noise_variance_levels,
        xgb_values,
        marker="o",
        linewidth=2,
        label="XGBoost",
        color="tab:blue",
        zorder=2,
    )

    plt.plot(
        noise_variance_levels,
        tabicl_values,
        marker="s",
        linewidth=2,
        label="TabICL",
        color="tab:orange",
        zorder=2,
    )

    add_coordinate_labels(
        noise_variance_levels,
        xgb_values,
        metric,
        vertical_offset=11,
    )

    add_coordinate_labels(
        noise_variance_levels,
        tabicl_values,
        metric,
        vertical_offset=-13,
    )

    plt.xlabel("Noise Variance")
    plt.ylabel(y_label)

    plt.title(
        f"Synthetic Dataset {simulation_number} - "
        f"{metric_title} vs Noise Variance"
    )

    plt.xticks(noise_variance_levels)

    if metric == "top_k":
        plt.ylim(-0.08, 1.12)
    else:
        plt.margins(y=0.15)

    plt.legend()

    plt.grid(
        alpha=0.25,
        zorder=0,
    )

    plt.tight_layout()
    plt.show()
    plt.close()


X_variance_sim_1 = X_synthetic_1.to_numpy(
    copy=True
)

X_variance_sim_2 = X_synthetic_2.to_numpy(
    copy=True
)

simulation_1_variance_results = (
    run_noise_variance_experiment(
        X_variance_sim_1,
        simulation_number=1,
        true_feature_ids=[0, 1],
    )
)

simulation_2_variance_results = (
    run_noise_variance_experiment(
        X_variance_sim_2,
        simulation_number=2,
        true_feature_ids=[0, 1, 2, 3, 4, 5],
    )
)

# Plot 1: Synthetic Dataset 1 - Top-k Accuracy vs Noise Variance

plot_variance_metric(
    simulation_1_variance_results,
    simulation_number=1,
    metric="top_k",
)

# Plot 2: Synthetic Dataset 2 - Top-k Accuracy vs Noise Variance

plot_variance_metric(
    simulation_2_variance_results,
    simulation_number=2,
    metric="top_k",
)

# Plot 3: Synthetic Dataset 1 - Rank Sum vs Noise Variance

plot_variance_metric(
    simulation_1_variance_results,
    simulation_number=1,
    metric="rank_sum",
)

# Plot 4: Synthetic Dataset 2 - Rank Sum vs Noise Variance

plot_variance_metric(
    simulation_2_variance_results,
    simulation_number=2,
    metric="rank_sum",
)
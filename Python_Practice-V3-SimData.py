# IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapiq
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNRegressor, set_access_token
from xgboost import XGBRegressor

# TABPFN ACCESS TOKEN

set_access_token("tabpfn_sk_fLu-IoihAjcAQRR107zlhwcONEyED2UTlDsXNezI8i4")

# AUTO SAVE ALL GRAPHS

graph_number = 1
original_show = plt.show

def save_and_show(*args, **kwargs):
    global graph_number

    filename = Path.cwd() / f"Graph_{graph_number:02d}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {filename.name}")

    graph_number += 1
    original_show(*args, **kwargs)

plt.show = save_and_show

# SETTINGS

n_samples_test = 2
shap_budget = 16

n_samples = 500
n_features = 10

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

tabpfn_regressor_1 = TabPFNRegressor()

tabpfn_regressor_1.fit(
    X_train_1,
    y_train_1,
)


def tabpfn_synthetic_1_predict(X):
    return tabpfn_regressor_1.predict(X)


tabpfn_explainer_1 = shapiq.TabularExplainer(
    model=tabpfn_synthetic_1_predict,
    data=X_train_1,
    index="SV",
    max_order=1,
    random_state=42,
)

tabpfn_shap_1 = [
    tabpfn_explainer_1.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_1
]


# Plot 3: TabPFN Synthetic Dataset 1 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabpfn_shap_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("TabPFN - Synthetic Dataset 1 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 4: TabPFN Synthetic Dataset 1 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabpfn_shap_1,
    data=X_explain_1,
    feature_names=feature_names_synthetic_1,
    show=False,
)

plt.title("TabPFN - Synthetic Dataset 1 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 1 MODEL PERFORMANCE COMPARISON

xgb_predictions_1 = xgb_regressor_1.predict(X_test_1)
tabpfn_predictions_1 = tabpfn_regressor_1.predict(X_test_1)

xgb_r2_1 = r2_score(y_test_1, xgb_predictions_1)
tabpfn_r2_1 = r2_score(y_test_1, tabpfn_predictions_1)

print(f"Synthetic Dataset 1 - XGBoost R^2: {xgb_r2_1:.4f}")
print(f"Synthetic Dataset 1 - TabPFN R^2: {tabpfn_r2_1:.4f}")

# Plot 5: Synthetic Dataset 1 R^2 Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabPFN"],
    [xgb_r2_1, tabpfn_r2_1],
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
# y_i = sin(x_i1) + x_i2^2 + x_i3 + I{x_i4 > 0} + epsilon_i
# epsilon_i ~ N(0, 1)
# All x_ij ~ N(0, 1)

rng_2 = np.random.default_rng(84)

X_synthetic_2_array = rng_2.normal(
    loc=0.0,
    scale=1.0,
    size=(n_samples, n_features),
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

tabpfn_regressor_2 = TabPFNRegressor()

tabpfn_regressor_2.fit(
    X_train_2,
    y_train_2,
)


def tabpfn_synthetic_2_predict(X):
    return tabpfn_regressor_2.predict(X)


tabpfn_explainer_2 = shapiq.TabularExplainer(
    model=tabpfn_synthetic_2_predict,
    data=X_train_2,
    index="SV",
    max_order=1,
    random_state=42,
)

tabpfn_shap_2 = [
    tabpfn_explainer_2.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_2
]


# Plot 8: TabPFN Synthetic Dataset 2 Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabpfn_shap_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("TabPFN - Synthetic Dataset 2 - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 9: TabPFN Synthetic Dataset 2 Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabpfn_shap_2,
    data=X_explain_2,
    feature_names=feature_names_synthetic_2,
    show=False,
)

plt.title("TabPFN - Synthetic Dataset 2 - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# SYNTHETIC DATASET 2 MODEL PERFORMANCE COMPARISON

xgb_predictions_2 = xgb_regressor_2.predict(X_test_2)
tabpfn_predictions_2 = tabpfn_regressor_2.predict(X_test_2)

xgb_r2_2 = r2_score(y_test_2, xgb_predictions_2)
tabpfn_r2_2 = r2_score(y_test_2, tabpfn_predictions_2)

print(f"Synthetic Dataset 2 - XGBoost R^2: {xgb_r2_2:.4f}")
print(f"Synthetic Dataset 2 - TabPFN R^2: {tabpfn_r2_2:.4f}")

# Plot 10: Synthetic Dataset 2 R^2 Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabPFN"],
    [xgb_r2_2, tabpfn_r2_2],
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

# Plot 11: LOCO Graph - TabPFN, Synthetic Dataset 1

plot_loco(
    TabPFNRegressor,
    X_train_1, y_train_1, X_test_1, y_test_1,
    feature_names_synthetic_1,
    "regression",
    "TabPFN - Synthetic Dataset 1 - LOCO",
)

# Plot 12: LOCO Graph - TabPFN, Synthetic Dataset 2

plot_loco(
    TabPFNRegressor,
    X_train_2, y_train_2, X_test_2, y_test_2,
    feature_names_synthetic_2,
    "regression",
    "TabPFN - Synthetic Dataset 2 - LOCO",
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

# ICE FEATURE ID DEFINITION

TABPFN_SYNTHETIC_1_FEATURE = feature_names_synthetic_1.index(
    "x1"
)
TABPFN_SYNTHETIC_2_FEATURE = feature_names_synthetic_2.index(
    "x2"
)
XGBOOST_SYNTHETIC_1_FEATURE = feature_names_synthetic_1.index(
    "x1"
)
XGBOOST_SYNTHETIC_2_FEATURE = feature_names_synthetic_2.index(
    "x2"
)

# Plot 15: ICE Graph - TabPFN, Synthetic Dataset 1

plot_ice(
    tabpfn_regressor_1,
    X_test_1,
    TABPFN_SYNTHETIC_1_FEATURE,
    feature_names_synthetic_1,
    "regression",
    "TabPFN - Synthetic Dataset 1 - ICE",
)

# Plot 16: ICE Graph - TabPFN, Synthetic Dataset 2

plot_ice(
    tabpfn_regressor_2,
    X_test_2,
    TABPFN_SYNTHETIC_2_FEATURE,
    feature_names_synthetic_2,
    "regression",
    "TabPFN - Synthetic Dataset 2 - ICE",
)

# Plot 17: ICE Graph - XGBoost, Synthetic Dataset 1

plot_ice(
    xgb_regressor_1,
    X_test_1,
    XGBOOST_SYNTHETIC_1_FEATURE,
    feature_names_synthetic_1,
    "regression",
    "XGBoost - Synthetic Dataset 1 - ICE",
)

# Plot 18: ICE Graph - XGBoost, Synthetic Dataset 2

plot_ice(
    xgb_regressor_2,
    X_test_2,
    XGBOOST_SYNTHETIC_2_FEATURE,
    feature_names_synthetic_2,
    "regression",
    "XGBoost - Synthetic Dataset 2 - ICE",
)

# Plot 19: Sensitivity Graph - TabPFN, Synthetic Dataset 1

plot_sensitivity(
    tabpfn_regressor_1,
    X_test_1,
    feature_names_synthetic_1,
    "regression",
    "TabPFN - Synthetic Dataset 1 - Sensitivity",
)

# Plot 20: Sensitivity Graph - TabPFN, Synthetic Dataset 2

plot_sensitivity(
    tabpfn_regressor_2,
    X_test_2,
    feature_names_synthetic_2,
    "regression",
    "TabPFN - Synthetic Dataset 2 - Sensitivity",
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
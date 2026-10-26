# IMPORTS

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapiq
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token
from xgboost import XGBClassifier, XGBRegressor

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


# PARKINSON'S DATASET

parkinsons_url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/parkinsons/parkinsons.data"
)

parkinsons = pd.read_csv(parkinsons_url)

X_parkinsons = parkinsons.drop(columns=["status", "name"])
y_parkinsons = parkinsons["status"]

classification_feature_labels = {
    "MDVP:Fo(Hz)": "Average Voice Pitch",
    "MDVP:Fhi(Hz)": "Maximum Voice Pitch",
    "MDVP:Flo(Hz)": "Minimum Voice Pitch",
    "MDVP:Jitter(%)": "Pitch Variation (%)",
    "MDVP:Jitter(Abs)": "Absolute Pitch Variation",
    "MDVP:RAP": "Short-Term Pitch Variation (RAP)",
    "MDVP:PPQ": "Pitch Perturbation Quotient",
    "Jitter:DDP": "Average Pitch Difference",
    "MDVP:Shimmer": "Amplitude Variation",
    "MDVP:Shimmer(dB)": "Amplitude Variation (dB)",
    "Shimmer:APQ3": "Short-Term Loudness Variation (3-cycle)",
    "Shimmer:APQ5": "Short-Term Loudness Variation (5-cycle)",
    "MDVP:APQ": "Amplitude Perturbation Quotient",
    "Shimmer:DDA": "Average Loudness Difference",
    "NHR": "Noise-to-Harmonics Ratio",
    "HNR": "Harmonics-to-Noise Ratio",
    "RPDE": "Voice Signal Irregularity",
    "DFA": "Long-Term Voice Pattern Scaling",
    "spread1": "Pitch Variation Spread 1",
    "spread2": "Pitch Variation Spread 2",
    "D2": "Voice Signal Complexity",
    "PPE": "Pitch Period Entropy",
}

feature_names_parkinsons = [
    classification_feature_labels[column] for column in X_parkinsons.columns
]

classification_target_label = "Parkinson's Disease Status"

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_parkinsons,
    y_parkinsons,
    test_size=0.25,
    random_state=42,
    stratify=y_parkinsons,
)

X_train_p = X_train_p.to_numpy()
X_test_p = X_test_p.to_numpy()
y_train_p = y_train_p.to_numpy()

X_explain_p = X_test_p[:n_samples_test]

# PARKINSON'S DATA - XGBOOST

xgb_classifier = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)

xgb_classifier.fit(
    X_train_p,
    y_train_p,
)


def xgb_parkinsons_predict(X):
    return xgb_classifier.predict_proba(X)[:, 1]


xgb_explainer_p = shapiq.TabularExplainer(
    model=xgb_parkinsons_predict,
    data=X_train_p,
    index="SV",
    max_order=1,
    random_state=42,
)

xgb_shap_p = [
    xgb_explainer_p.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_p
]


# Plot 1: XGBoost Parkinson's Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    xgb_shap_p,
    feature_names=feature_names_parkinsons,
    show=False,
)

plt.title("XGBoost - Parkinson's Disease - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 2: XGBoost Parkinson's Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    xgb_shap_p,
    data=X_explain_p,
    feature_names=feature_names_parkinsons,
    show=False,
)

plt.title("XGBoost - Parkinson's Disease - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# PARKINSON'S DATA - TABPFN

tabpfn_classifier = TabPFNClassifier()

tabpfn_classifier.fit(
    X_train_p,
    y_train_p,
)


def tabpfn_parkinsons_predict(X):
    return tabpfn_classifier.predict_proba(X)[:, 1]


tabpfn_explainer_p = shapiq.TabularExplainer(
    model=tabpfn_parkinsons_predict,
    data=X_train_p,
    index="SV",
    max_order=1,
    random_state=42,
)

tabpfn_shap_p = [
    tabpfn_explainer_p.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_p
]


# Plot 3: TabPFN Parkinson's Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabpfn_shap_p,
    feature_names=feature_names_parkinsons,
    show=False,
)

plt.title("TabPFN - Parkinson's Disease - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 4: TabPFN Parkinson's Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabpfn_shap_p,
    data=X_explain_p,
    feature_names=feature_names_parkinsons,
    show=False,
)

plt.title("TabPFN - Parkinson's Disease - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# PARKINSON'S MODEL PERFORMANCE COMPARISON

xgb_probabilities_p = xgb_classifier.predict_proba(X_test_p)[:, 1]
tabpfn_probabilities_p = tabpfn_classifier.predict_proba(X_test_p)[:, 1]

xgb_roc_auc = roc_auc_score(y_test_p, xgb_probabilities_p)
tabpfn_roc_auc = roc_auc_score(y_test_p, tabpfn_probabilities_p)

print(f"XGBoost ROC AUC: {xgb_roc_auc:.4f}")
print(f"TabPFN ROC AUC: {tabpfn_roc_auc:.4f}")

# Plot 5: ROC AUC Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabPFN"],
    [xgb_roc_auc, tabpfn_roc_auc],
)

plt.bar_label(
    bars,
    fmt="%.4f",
    padding=3,
)

plt.ylim(0.95, 1.05)
plt.ylabel("ROC AUC")
plt.title(f"{classification_target_label} - ROC AUC Comparison")
plt.tight_layout()
plt.show()
plt.close()

# BOSTON HOUSING DATASET

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

X_boston = boston.drop(columns=["MEDV"])
y_boston = boston["MEDV"]

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
}

feature_names_boston = [
    regression_feature_labels[column] for column in X_boston.columns
]

regression_target_label = "Median Home Value (1000s of $)"

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_boston,
    y_boston,
    test_size=0.25,
    random_state=42,
)

X_train_b = X_train_b.to_numpy()
X_test_b = X_test_b.to_numpy()
y_train_b = y_train_b.to_numpy()

X_explain_b = X_test_b[:n_samples_test]

# BOSTON DATA - XGBOOST

xgb_regressor = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
)

xgb_regressor.fit(
    X_train_b,
    y_train_b,
)


def xgb_boston_predict(X):
    return xgb_regressor.predict(X)


xgb_explainer_b = shapiq.TabularExplainer(
    model=xgb_boston_predict,
    data=X_train_b,
    index="SV",
    max_order=1,
    random_state=42,
)

xgb_shap_b = [
    xgb_explainer_b.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_b
]


# Plot 6: XGBoost Boston Housing Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    xgb_shap_b,
    feature_names=feature_names_boston,
    show=False,
)

plt.title("XGBoost - Boston Housing - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 7: XGBoost Boston Housing Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    xgb_shap_b,
    data=X_explain_b,
    feature_names=feature_names_boston,
    show=False,
)

plt.title("XGBoost - Boston Housing - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# BOSTON DATA - TABPFN

tabpfn_regressor = TabPFNRegressor()

tabpfn_regressor.fit(
    X_train_b,
    y_train_b,
)


def tabpfn_boston_predict(X):
    return tabpfn_regressor.predict(X)


tabpfn_explainer_b = shapiq.TabularExplainer(
    model=tabpfn_boston_predict,
    data=X_train_b,
    index="SV",
    max_order=1,
    random_state=42,
)

tabpfn_shap_b = [
    tabpfn_explainer_b.explain(
        x=x,
        budget=shap_budget,
    )
    for x in X_explain_b
]


# Plot 8: TabPFN Boston Housing Bar Plot

plt.figure(figsize=(10, 6))

shapiq.plot.bar_plot(
    tabpfn_shap_b,
    feature_names=feature_names_boston,
    show=False,
)

plt.title("TabPFN - Boston Housing - Feature Importance")
plt.tight_layout()
plt.show()
plt.close()


# Plot 9: TabPFN Boston Housing Beeswarm Plot

plt.figure(figsize=(10, 6))

shapiq.plot.beeswarm_plot(
    tabpfn_shap_b,
    data=X_explain_b,
    feature_names=feature_names_boston,
    show=False,
)

plt.title("TabPFN - Boston Housing - Beeswarm")
plt.tight_layout()
plt.show()
plt.close()

# BOSTON HOUSING MODEL PERFORMANCE COMPARISON

xgb_predictions_b = xgb_regressor.predict(X_test_b)
tabpfn_predictions_b = tabpfn_regressor.predict(X_test_b)

xgb_r2 = r2_score(y_test_b, xgb_predictions_b)
tabpfn_r2 = r2_score(y_test_b, tabpfn_predictions_b)

print(f"XGBoost R^2: {xgb_r2:.4f}")
print(f"TabPFN R^2: {tabpfn_r2:.4f}")

# Plot 10: R^2 Comparison

plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["XGBoost", "TabPFN"],
    [xgb_r2, tabpfn_r2],
)

plt.bar_label(
    bars,
    fmt="%.4f",
    padding=3,
)

plt.ylim(0.85, 1.0)
plt.ylabel("R^2")
plt.title(f"{regression_target_label} - R^2 Comparison")
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

# Plot 11: LOCO Graph - TabPFN, Parkinson's

plot_loco(
    TabPFNClassifier,
    X_train_p, y_train_p, X_test_p, y_test_p,
    feature_names_parkinsons,
    "classification",
    "TabPFN - Parkinson's Disease - LOCO",
)

# Plot 12: LOCO Graph - TabPFN, Boston Housing

plot_loco(
    TabPFNRegressor,
    X_train_b, y_train_b, X_test_b, y_test_b,
    feature_names_boston,
    "regression",
    "TabPFN - Boston Housing - LOCO",
)

# Plot 13: LOCO Graph - XGBoost, Parkinson's

plot_loco(
    lambda: clone(xgb_classifier),
    X_train_p, y_train_p, X_test_p, y_test_p,
    feature_names_parkinsons,
    "classification",
    "XGBoost - Parkinson's Disease - LOCO",
)

# Plot 14: LOCO Graph - XGBoost, Boston Housing

plot_loco(
    lambda: clone(xgb_regressor),
    X_train_b, y_train_b, X_test_b, y_test_b,
    feature_names_boston,
    "regression",
    "XGBoost - Boston Housing - LOCO",
)

# ICE FEATURE ID DEFINITION

TABPFN_PARKINSONS_FEATURE = feature_names_parkinsons.index(
    "Voice Signal Irregularity"
)
TABPFN_BOSTON_FEATURE = feature_names_boston.index(
    "Average Number of Rooms"
)
XGBOOST_PARKINSONS_FEATURE = feature_names_parkinsons.index(
    "Pitch Variation Spread 2"
)
XGBOOST_BOSTON_FEATURE = feature_names_boston.index(
    "Average Number of Rooms"
)

# Plot 15: ICE Graph - TabPFN, Parkinson's

plot_ice(
    tabpfn_classifier,
    X_test_p,
    TABPFN_PARKINSONS_FEATURE,
    feature_names_parkinsons,
    "classification",
    "TabPFN - Parkinson's Disease - ICE",
)

# Plot 16: ICE Graph - TabPFN, Boston Housing

plot_ice(
    tabpfn_regressor,
    X_test_b,
    TABPFN_BOSTON_FEATURE,
    feature_names_boston,
    "regression",
    "TabPFN - Boston Housing - ICE",
)

# Plot 17: ICE Graph - XGBoost, Parkinson's

plot_ice(
    xgb_classifier,
    X_test_p,
    XGBOOST_PARKINSONS_FEATURE,
    feature_names_parkinsons,
    "classification",
    "XGBoost - Parkinson's Disease - ICE",
)

# Plot 18: ICE Graph - XGBoost, Boston Housing

plot_ice(
    xgb_regressor,
    X_test_b,
    XGBOOST_BOSTON_FEATURE,
    feature_names_boston,
    "regression",
    "XGBoost - Boston Housing - ICE",
)

# Plot 19: Sensitivity Graph - TabPFN, Parkinson's

plot_sensitivity(
    tabpfn_classifier,
    X_test_p,
    feature_names_parkinsons,
    "classification",
    "TabPFN - Parkinson's Disease - Sensitivity",
)

# Plot 20: Sensitivity Graph - TabPFN, Boston Housing

plot_sensitivity(
    tabpfn_regressor,
    X_test_b,
    feature_names_boston,
    "regression",
    "TabPFN - Boston Housing - Sensitivity",
)

# Plot 21: Sensitivity Graph - XGBoost, Parkinson's

plot_sensitivity(
    xgb_classifier,
    X_test_p,
    feature_names_parkinsons,
    "classification",
    "XGBoost - Parkinson's Disease - Sensitivity",
)

# Plot 22: Sensitivity Graph - XGBoost, Boston Housing

plot_sensitivity(
    xgb_regressor,
    X_test_b,
    feature_names_boston,
    "regression",
    "XGBoost - Boston Housing - Sensitivity",
)
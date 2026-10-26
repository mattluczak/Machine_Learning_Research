# IMPORTS

from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token

import shapiq
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

# TABPFN ACCESS TOKEN

set_access_token("tabpfn_sk_fLu-IoihAjcAQRR107zlhwcONEyED2UTlDsXNezI8i4")

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

feature_names_parkinsons = list(X_parkinsons.columns)

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

feature_names_boston = list(X_boston.columns)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_boston,
    y_boston,
    test_size=0.5,
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


# Plot 5: XGBoost Boston Housing Bar Plot

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


# Plot 6: XGBoost Boston Housing Beeswarm Plot

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


# Plot 7: TabPFN Boston Housing Bar Plot

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


# Plot 8: TabPFN Boston Housing Beeswarm Plot

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
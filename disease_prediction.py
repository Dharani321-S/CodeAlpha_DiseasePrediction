"""
CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data
Dataset : Breast Cancer Wisconsin (UCI, bundled with scikit-learn)
Models  : Logistic Regression, SVM, Random Forest, XGBoost
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)
from xgboost import XGBClassifier

# 1. Load data ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # target: 0 = malignant, 1 = benign
print("Shape:", X.shape)
print("Class balance:\n", y.value_counts(), "\n")
print("Missing values:", int(X.isna().sum().sum()))

# 2. Train / test split ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Models (scaling inside pipeline avoids data leakage) --------------------
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=1000))]),
    "SVM": Pipeline([("sc", StandardScaler()),
                     ("m", SVC(probability=True, random_state=42))]),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                             eval_metric="logloss", random_state=42),
}

# 4. Train + evaluate --------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, fitted = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV Accuracy": cross_val_score(model, X_train, y_train, cv=cv).mean(),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n=== Model Comparison ===")
print(results)
results.to_csv("results.csv")

best_name = results["F1"].idxmax()
best = fitted[best_name]
print(f"\nBest model (by F1): {best_name}")
print(classification_report(y_test, best.predict(X_test),
                            target_names=data.target_names))

# 5. Plots -------------------------------------------------------------------
# Confusion matrix
cm = confusion_matrix(y_test, best.predict(X_test))
plt.figure(figsize=(4, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, model in fitted.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC Curves")
plt.tight_layout(); plt.savefig("roc_curves.png", dpi=150); plt.close()

# Feature importance (Random Forest)
imp = pd.Series(fitted["Random Forest"].feature_importances_,
                index=X.columns).sort_values().tail(10)
plt.figure(figsize=(6, 4))
imp.plot(kind="barh")
plt.title("Top 10 Features - Random Forest")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()

print("\nSaved: results.csv, confusion_matrix.png, roc_curves.png, feature_importance.png")

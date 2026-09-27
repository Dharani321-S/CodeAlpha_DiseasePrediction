# CodeAlpha_DiseasePrediction

Machine Learning Internship - **Task 4: Disease Prediction from Medical Data** (CodeAlpha)

## Objective
Predict whether a tumor is **malignant or benign** from patient medical measurements.

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (bundled in scikit-learn).
569 samples, 30 numeric features (radius, texture, perimeter, area, etc.), no missing values.

## Approach
1. Load data and check class balance / missing values
2. Stratified 80/20 train-test split
3. Train 4 models: Logistic Regression, SVM, Random Forest, XGBoost
   (scaling done inside a Pipeline to avoid data leakage)
4. Evaluate with 5-fold CV plus Accuracy, Precision, Recall, F1, ROC-AUC
5. Plot confusion matrix, ROC curves, feature importance

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| SVM | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |
| XGBoost | 0.9474 | 0.9459 | 0.9722 | 0.9589 | 0.9921 |

Best model: **Logistic Regression** (about 98% accuracy).

## Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```
Outputs: `results.csv`, `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`

## Tech
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn

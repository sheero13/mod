import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

filepath = "my_graph2.tsv"  
sep = "\t"
label_col = "target"         # name of the class label column in the TSV
variance_threshold = 0.95 
fixed_n_components = None   
test_size = 0.2
random_state = 42

df = pd.read_csv(filepath, sep=sep)
print("Dataset loaded. Shape:", df.shape)
print(df.head())

if label_col not in df.columns:
    raise ValueError(f"Label column '{label_col}' not found in data columns: {df.columns.tolist()}")

X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors='coerce')
y = df[label_col]

nan_rows = X.isnull().any(axis=1)
if nan_rows.any():
    print(f"Dropping {nan_rows.sum()} rows with non-numeric or missing features.")
    X = X[~nan_rows]
    y = y[~nan_rows]

print("Features shape after cleaning:", X.shape)

corr = X.corr()
print("\nCorrelation matrix:\n", corr.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'shrink': .8})
plt.title("Feature Correlation Matrix (heatmap)")
plt.tight_layout()
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if fixed_n_components is not None:
    n_comp = fixed_n_components
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_ratio = pca.explained_variance_ratio_.cumsum()
else:
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.searchsorted(cumvar, variance_threshold) + 1
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_ratio = np.cumsum(pca.explained_variance_ratio_)

print(f"\nPCA chosen components: {n_comp}")
print("Cumulative explained variance by chosen components:", explained_ratio[-1].round(4))

pca_full = PCA(random_state=random_state)
pca_full.fit(X_scaled)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.axhline(variance_threshold, color='red', linestyle='--', label=f'{variance_threshold*100:.0f}% variance')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=test_size, stratify=y, random_state=random_state
)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  
}

svc = SVC(probability=True, random_state=random_state)
grid = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest SVM params:", grid.best_params_)
print("Best CV ROC AUC:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SVM on PCA features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Short justification / interpretation
print("\nJustification and interpretation:")
print(f"- PCA reduced original {X.shape[1]} features to {n_comp} components explaining {explained_ratio[-1]*100:.2f}% variance.")
print("- Using PCA helps reduce dimensionality, noise, and multicollinearity before training SVM.")
print("- GridSearchCV tuned SVM hyperparameters optimizing ROC AUC on cross-validation.")
print("- Evaluate ROC AUC and classification report to decide whether model generalizes well.")
print("- If recall (sensitivity) is important, choose threshold favouring higher recall at cost of precision.")
print("\nRecommendation:")
print("- If performance is poor, try: more PCA variance (increase threshold), try other classifiers, tune more hyperparameters, or use domain-specific feature engineering.")

# Optional: show first few PCA components (feature contributions)
loadings = pca.components_.T  # shape: original_features x n_comp
pc_index = list(range(1, n_comp+1))
loading_df = pd.DataFrame(loadings, index=X.columns, columns=[f"PC{i}" for i in pc_index])
print("\nTop contributors to principal components (absolute loadings):")
print(loading_df.abs().sort_values(by=f"PC1", ascending=False).head(10))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# --- CONFIGURATION (ADAPTED FOR YOUR DATA) ---
filepath = "my_graph4.tsv"  # Use the correct file name (e.g., my_graph4.tsv)
sep = "\t"
label_col_index = 2         # CRITICAL FIX: Assuming Column 2 (index 2) holds the classification label (0 or 1)
variance_threshold = 0.95   # Retain 95% of the variance after PCA
test_size = 0.2
random_state = 42

# --- 1. DATA LOADING AND CLEANING ---

# CRITICAL FIX: Use header=None to read files without column names
try:
    df = pd.read_csv(filepath, sep=sep, header=None)
    # Assign temporary column names 0, 1, 2...
    df.columns = range(df.shape[1])
    print(f"Dataset loaded. Shape: {df.shape}")
    print("\nFirst 5 rows (Features and Target):")
    print(df.head())
except FileNotFoundError:
    print(f"Error: File not found at {filepath}. Please check the path.")
    exit()

# Separate Features (X) and Target (y)
# X contains all columns EXCEPT the label_col_index
X = df.drop(columns=[label_col_index]).apply(pd.to_numeric, errors='coerce')
y = df[label_col_index].apply(pd.to_numeric, errors='coerce')

# Handle rows where conversion to numeric failed (often due to non-1 values)
nan_rows = X.isnull().any(axis=1) | y.isnull()
if nan_rows.any():
    print(f"Dropping {nan_rows.sum()} rows with non-numeric or missing data.")
    X = X[~nan_rows]
    y = y[~nan_rows]

print(f"\nFeatures shape after cleaning: {X.shape}, Target shape: {y.shape}")

# Ensure we have data left and a classification problem (min 2 classes)
if X.empty or y.nunique() < 2:
    print("\nError: Data set is empty or does not contain at least two classes for classification.")
    exit()


# --- 2. CORRELATION MATRIX ---

# Note: Correlation may fail if columns are constant (all 1s), which is common in graph data.
try:
    corr = X.corr()
    print("\nFeature Correlation Matrix:")
    print(corr.round(3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'shrink': .8})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"\nWarning: Could not compute correlation matrix. Reason: {e}")
    print("This often happens if features have zero variance (e.g., all values are the same).")


# --- 3. PCA: SCALING AND DIMENSIONALITY REDUCTION ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of components for 95% variance threshold
pca_full = PCA(random_state=random_state)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = np.searchsorted(cumvar, variance_threshold) + 1
n_comp = max(1, min(n_comp, X.shape[1])) # Ensure n_comp is valid
    
pca = PCA(n_components=n_comp, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)
explained_ratio = np.cumsum(pca.explained_variance_ratio_)

print(f"\nPCA chosen components: {n_comp}")
print(f"Cumulative explained variance by chosen components: {explained_ratio[-1].round(4)}")

# Plot explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.axhline(variance_threshold, color='red', linestyle='--', label=f'{variance_threshold*100:.0f}% variance')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance (Elbow Plot)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 4. DATA SPLIT ---

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=test_size, stratify=y, random_state=random_state
)
print(f"\nTrain shape (PCA features): {X_train.shape}, Test shape: {X_test.shape}")


# --- 5. SVM CLASSIFIER (WITH HYPERPARAMETER TUNING) ---

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  
}

# Setup SVM and GridSearchCV
svc = SVC(probability=True, random_state=random_state)
grid = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("\nBest SVM parameters:", grid.best_params_)
print(f"Best CV ROC AUC: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# --- 6. MODEL EVALUATION ---

y_pred = best_model.predict(X_test)
# Get probability for positive class (index 1)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (SVM on PCA features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- 7. FINAL JUSTIFICATION ---
print("\n--- Justification and Interpretation ---")
print(f"- **PCA:** Reduced {X.shape[1]} original features to {n_comp} components, explaining {explained_ratio[-1]*100:.2f}% of the variance. This combats dimensionality and noise.")
print("- **SVM:** Chosen for its effectiveness in high-dimensional spaces (even after reduction) and ability to find complex decision boundaries.")
print("- **Hyperparameter Tuning:** GridSearchCV ensured the optimal `C`, `kernel`, and `gamma` were chosen, maximizing model generalization and performance (measured by ROC AUC).")
print("- **Results:** The Test ROC AUC score shows the model's overall discriminative power between the two classes in unseen data.")

#-----------------------------------------------------------------------------------------------------------------
# Q1. Naive Bayes - Gender Classification
#-----------------------------------------------------------------------------------------------------------------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# df = pd.read_csv(r"gender_classification_v7.csv")


# df.columns = [
#     'long_hair',
#     'forehead_width_cm',
#     'forehead_height_cm',
#     'nose_wide',
#     'nose_long',
#     'lips_thin',
#     'distance_nose_to_lip_long',
#     'gender'
# ]

# print(" Dataset loaded successfully!")
# print(df.head())



# X = df.drop('gender', axis=1)
# y = df['gender']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# model = GaussianNB()
# model.fit(X_train, y_train)


# y_pred = model.predict(X_test)

# print("\n Model Evaluation Results")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# new_profiles = pd.DataFrame([
#     [1, 12.0, 6.2, 0, 0, 0, 0],
#     [0, 14.5, 5.8, 1, 1, 1, 1]
# ], columns=['long_hair', 'forehead_width_cm', 'forehead_height_cm',
#             'nose_wide', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long'])

# predictions = model.predict(new_profiles)
# print("\n Predictions for New Profiles:")
# print(new_profiles)
# print("Predicted Genders:", predictions)


#--------------------------------------------------------------------------------------------------------------------------
# Q2. SVD - Credit Card Fraud Detection Classification
#--------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.decomposition import PCA

# df = pd.read_csv(r"creditcard.csv") 

# X = df.drop('Class', axis=1)
# y = df['Class']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )

# svm = SVC(kernel='rbf', random_state=42)
# svm.fit(X_train, y_train)

# y_pred = svm.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Simple Classification Plot using PCA 
# pca = PCA(n_components=2)
# X_test_pca = pca.fit_transform(X_test)

# plt.figure(figsize=(6, 5))
# sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1],
#                 hue=y_pred, style=y_test,
#                 palette='coolwarm', s=40)
# plt.title("SVM Classification (PCA Projection)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(title="Predicted / Actual", loc='upper right')
# plt.show()

#-------------------------------------------------------------------------------------------------------------------------
#Q3. Logistic Regression - Heart disease prediction
#-------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load dataset
# df = pd.read_csv(r"heart.csv")
# print("Dataset loaded successfully!")
# print(df.head())

# # Choose only two features for visualization
# X = df[["age", "chol"]]     # you can change features like ["age", "trestbps"]
# y = df["target"]

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Scale data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train logistic regression
# model = LogisticRegression()
# model.fit(X_train_scaled, y_train)

# # Predict
# y_pred = model.predict(X_test_scaled)

# # Evaluation
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # --- Decision Boundary Plot ---
# plt.figure(figsize=(7, 5))

# # Create mesh grid
# x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
# y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))

# # Predict for each point in mesh
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot decision boundary
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
# plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
# plt.xlabel("Age (scaled)")
# plt.ylabel("Cholesterol (scaled)")
# plt.title("Logistic Regression Decision Boundary")
# plt.show()


#--------------------------------------------------------------------------------------------------------------------------------
#Q4. Linear Regression - Medical Cost prediction
#--------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error, r2_score

# data = pd.read_csv(r"insurance.csv")  

# label_enc = LabelEncoder()
# data['sex'] = label_enc.fit_transform(data['sex'])
# data['smoker'] = label_enc.fit_transform(data['smoker'])
# data['region'] = label_enc.fit_transform(data['region'])

# X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
# y = data['charges']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))

# sample = [[28, 1, 33.0, 2, 0, 2]]  # [age, sex, bmi, children, smoker, region]
# print("Predicted Charge:", model.predict(sample)[0])

# plt.figure(figsize=(6, 4))
# plt.scatter(y_test, y_pred, color='blue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
# plt.xlabel("Actual Charges")
# plt.ylabel("Predicted Charges")
# plt.title("Actual vs Predicted Medical Charges")
# plt.show()

#---------------------------------------------------------------------------------------------------------------------
#Q5. Matrix - Det & Transpose
#---------------------------------------------------------------------------------------------------------------------

# import numpy as np

# rows_A = int(input("Enter number of rows for Matrix A: "))
# cols_A = int(input("Enter number of columns for Matrix A: "))

# rows_B = int(input("Enter number of rows for Matrix B: "))
# cols_B = int(input("Enter number of columns for Matrix B: "))

# print("\nEnter elements of Matrix A (space separated rows):")
# A = np.array([list(map(float, input().split())) for _ in range(rows_A)])

# print("\nEnter elements of Matrix B (space separated rows):")
# B = np.array([list(map(float, input().split())) for _ in range(rows_B)])

# # Dot product (only if inner dimensions match)
# if cols_A == rows_B:
#     C = np.dot(A, B)
# else:
#     C = None
#     print("\nDot product not possible (A columns != B rows)")

# # Addition (only if dimensions are the same)
# if A.shape == B.shape:
#     D = A + B
# else:
#     D = None
#     print("\nAddition not possible (A and B must have same dimensions)")

# # Determinant (only for square matrices)
# def determinant(mat):
#     if mat is not None and mat.shape[0] == mat.shape[1]:
#         return np.linalg.det(mat)
#     return "Not defined (matrix not square)"

# # Transpose
# C_T = C.T if C is not None else "Not available"
# D_T = D.T if D is not None else "Not available"

# print("\nMatrix A:\n", A)
# print("\nMatrix B:\n", B)

# if C is not None:
#     print("\nC = A dot B:\n", C)
#     print("Determinant of C:", determinant(C))
#     print("Transpose of C:\n", C_T)

# if D is not None:
#     print("\nD = A + B:\n", D)
#     print("Determinant of D:", determinant(D))
#     print("Transpose of D:\n", D_T)

#--------------------------------------------------------------------------------------------------------------------
#Q6. Matrix - Eigen and Diag
#--------------------------------------------------------------------------------------------------------------------

# import numpy as np

# rows_A = int(input("Enter number of rows for Matrix A: "))
# cols_A = int(input("Enter number of columns for Matrix A: "))

# rows_B = int(input("Enter number of rows for Matrix B: "))
# cols_B = int(input("Enter number of columns for Matrix B: "))

# print("\nEnter elements of Matrix A (space separated rows):")
# A = np.array([list(map(float, input().split())) for _ in range(rows_A)])

# print("\nEnter elements of Matrix B (space separated rows):")
# B = np.array([list(map(float, input().split())) for _ in range(rows_B)])

# #Dot product 
# if cols_A == rows_B:
#     C = np.dot(A, B)
# else:
#     print("\nDot product not possible! (Columns of A != Rows of B)")
#     C = None

# # Eigenvalues and Eigenvectors 
# def eigen_vectors(mat):
#     if mat.shape[0] == mat.shape[1]:
#         values, vectors = np.linalg.eig(mat)
#         return values, vectors
#     else:
#         return None, None

# eig_vals_A, eig_vecs_A = eigen_vectors(A)
# eig_vals_B, eig_vecs_B = eigen_vectors(B)

# # Addition between eigenvectors of A and B 
# if eig_vecs_A is not None and eig_vecs_B is not None and eig_vecs_A.shape == eig_vecs_B.shape:
#     eigen_sum = eig_vecs_A + eig_vecs_B
# else:
#     eigen_sum = "Not possible (dimensions differ or non-square matrices)"

# # Diagonal matrix of C 
# if C is not None:
#     diag_C = np.diag(np.diag(C))
# else:
#     diag_C = "Not available (C not computed)"

# # Display results 
# print("\nMatrix A:\n", A)
# print("\nMatrix B:\n", B)

# if C is not None:
#     print("\nC = A dot B:\n", C)
#     print("\nDiagonal matrix of C:\n", diag_C)

# if eig_vecs_A is not None:
#     print("\nEigenvalues of A:\n", eig_vals_A)
#     print("Eigenvectors of A:\n", eig_vecs_A)
# else:
#     print("\nEigen decomposition not possible for A (not square).")

# if eig_vecs_B is not None:
#     print("\nEigenvalues of B:\n", eig_vals_B)
#     print("Eigenvectors of B:\n", eig_vecs_B)
# else:
#     print("\nEigen decomposition not possible for B (not square).")

# print("\nAddition between eigenvectors of A and B:\n", eigen_sum)

#-------------------------------------------------------------------------------------------------------------------------
#Q7.SVM_LR - Grid search
#-------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# -------------------------------
# 1. Load Dataset
# -------------------------------
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(r"heart.csv", names=columns, na_values='?')

# Fill missing values and convert target to binary
df.fillna(df.median(), inplace=True)
df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 2. Logistic Regression GridSearch
# -------------------------------
pipe_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
])

param_log = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1','l2']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_log = GridSearchCV(pipe_log, param_log, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_log.fit(X_train, y_train)

best_log = grid_log.best_estimator_

# -------------------------------
# 3. SVM GridSearch
# -------------------------------
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True))
])

param_svm = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear','rbf'],
    'clf__gamma': ['scale','auto']
}

grid_svm = GridSearchCV(pipe_svm, param_svm, scoring='roc_auc', cv=cv, n_jobs=-1)
grid_svm.fit(X_train, y_train)

best_svm = grid_svm.best_estimator_

# -------------------------------
# 4. Evaluate Models
# -------------------------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return y_proba

proba_log = evaluate(best_log, X_test, y_test, "Logistic Regression")
proba_svm = evaluate(best_svm, X_test, y_test, "SVM")

# -------------------------------
# 5. Plot ROC Curves
# -------------------------------
fpr_log, tpr_log, _ = roc_curve(y_test, proba_log)
fpr_svm, tpr_svm, _ = roc_curve(y_test, proba_svm)

plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC={roc_auc_score(y_test, proba_log):.3f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={roc_auc_score(y_test, proba_svm):.3f})')
plt.plot([0,1],[0,1],'k--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 6. Justification
# -------------------------------
print("\nJustification:")
print("- GridSearchCV tuned hyperparameters optimizing ROC AUC.")
print("- Logistic Regression is simple, interpretable, performs well for linear relationships.")
print("- SVM (RBF kernel) can capture non-linear patterns, sometimes giving slightly higher ROC AUC.")
print("- ROC curves help select the threshold based on Recall vs Precision preference.")

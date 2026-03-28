import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Load Dataset ──────────────────────────────────
df = pd.read_csv("loan_data.csv")

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum())

# ── Drop ID column (not a feature) ───────────────
df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

# ── Handle Missing Values ─────────────────────────
# Categorical: fill with mode
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Numerical: fill with median
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

print("\nMissing values after fill:", df.isnull().sum().sum())

# ── Feature Engineering ───────────────────────────
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmountLog"] = np.log1p(df["LoanAmount"])
df["TotalIncomeLog"] = np.log1p(df["TotalIncome"])

# ── Encode Categorical Columns ────────────────────
le = LabelEncoder()
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Define Features ───────────────────────────────
feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
    "TotalIncomeLog", "LoanAmountLog"
]

X = df[feature_cols]

# ── Target Column ─────────────────────────────────
if "Loan_Status" in df.columns:
    le_target = LabelEncoder()
    y = le_target.fit_transform(df["Loan_Status"])  # Y=1 Approved, N=0 Rejected
    print("\nTarget distribution:\n", pd.Series(y).value_counts())
else:
    # No labeled target — use Credit_History as proxy for demonstration
    print("\n[NOTE] No 'Loan_Status' column found. Using Credit_History as proxy target.")
    y = df["Credit_History"].astype(int)

# ── Train / Test Split ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── Train Decision Tree ───────────────────────────
dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
print(f"\nDecision Tree Accuracy : {dt_acc:.4f} ({dt_acc*100:.2f}%)")

# ── Train Random Forest ───────────────────────────
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Accuracy : {rf_acc:.4f} ({rf_acc*100:.2f}%)")

# ── Pick Best Model ───────────────────────────────
best_model = rf_model if rf_acc >= dt_acc else dt_model
best_name = "Random Forest" if rf_acc >= dt_acc else "Decision Tree"
print(f"\nBest model: {best_model.__class__.__name__}")

best_preds = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, best_preds))

# ── Feature Importance ────────────────────────────
importances = pd.Series(best_model.feature_importances_, index=feature_cols)
print("\nTop Feature Importances:")
print(importances.sort_values(ascending=False).to_string())

# ── Save Model ────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved to model.pkl")
print("Accuracy:", round(max(dt_acc, rf_acc) * 100, 2), "%")
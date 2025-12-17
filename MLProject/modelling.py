import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("🚀 CI Training (Production Mode)")

dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

mlflow.set_experiment("Modelling-Advance")

# ======================
# Load Data
# ======================
train_df = pd.read_csv("telco_churn_preprocessed/train_data.csv")
test_df = pd.read_csv("telco_churn_preprocessed/test_data.csv")

X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]
X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

# ======================
# Best Params (FROM KRITERIA 2)
# ======================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# ======================
# Train
# ======================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_params({
    "n_estimators": 100,
    "max_depth": 10
})

mlflow.log_metrics({
    "accuracy": acc,
    "f1_score": f1
})

mlflow.sklearn.log_model(model, "model")

# ======================
# Artifact: Confusion Matrix
# ======================
os.makedirs("artifacts", exist_ok=True)
cm_path = "artifacts/confusion_matrix.png"

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)

print("✅ CI training & logging selesai")

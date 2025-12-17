import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import matplotlib.pyplot as plt
import joblib

# =========================
# Init DagsHub + MLflow
# =========================
dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

# =========================
# Load Dataset (punyamu)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "telco_churn_preprocessed")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]
X_test  = test_df.drop("Churn", axis=1)
y_test  = test_df["Churn"]

input_example = X_train.iloc[:5]

# =========================
# Model Training
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)
model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# =========================
# Logging (PAKAI RUN AKTIF)
# =========================
run = mlflow.active_run()
print("✅ Active Run ID:", run.info.run_id)

mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 300)
mlflow.log_param("max_depth", 20)
mlflow.log_metric("accuracy", acc)

# =========================
# Confusion Matrix (Artefak)
# =========================
os.makedirs("model", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("model/confusion_matrix.png")
plt.close()

mlflow.log_artifact("model/confusion_matrix.png")

# =========================
# Classification Report (HTML)
# =========================
report_dict = classification_report(y_test, y_pred, output_dict=True)

html_content = f"""
<html>
<head><title>Telco Churn Classification Report</title></head>
<body>
    <h2>Classification Report</h2>
    <pre>{json.dumps(report_dict, indent=2)}</pre>
</body>
</html>
"""

with open("model/classification_report.html", "w") as f:
    f.write(html_content)

mlflow.log_artifact("model/classification_report.html")

# =========================
# Metric JSON
# =========================
with open("model/metric_info.json", "w") as f:
    json.dump({"accuracy": acc}, f)

mlflow.log_artifact("model/metric_info.json")

# =========================
# Backup Model (file biasa)
# =========================
joblib.dump(model, "model/model.pkl")
mlflow.log_artifact("model/model.pkl")

# =========================
# Requirements (Artefak)
# =========================
with open("model/requirements.txt", "w") as f:
    f.write(
        "scikit-learn\n"
        "mlflow\n"
        "joblib\n"
        "matplotlib\n"
        "pandas\n"
    )

mlflow.log_artifact("model/requirements.txt")

# =========================
# LOG SEBAGAI MLFLOW MODEL (INI YANG PENTING)
# =========================
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

print(f"🎯 Training selesai | Accuracy = {acc:.4f}")
print("📦 Model + artefak sukses dicatat di MLflow")

import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# DagsHub Init
# ======================
dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

# ======================
# Load Dataset
# ======================
train_df = pd.read_csv("telco_churn_preprocessed/train_data.csv")
test_df = pd.read_csv("telco_churn_preprocessed/test_data.csv")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

# ======================
# Fixed Best Params (FROM KRITERIA 2)
# ======================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42
)

# ======================
# MLflow Tracking
# ======================
mlflow.set_experiment("Telco_Churn_CI")

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    mlflow.sklearn.log_model(model, "model")

    # ======================
    # Artifact: Confusion Matrix
    # ======================
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("artifacts/confusion_matrix.png")

    print("CI training completed")

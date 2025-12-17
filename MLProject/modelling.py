import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

def train_basic_model():

    print("🚀 Training via MLflow Project dimulai")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "telco_churn_preprocessed")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # ⬇️ AMBIL RUN DARI MLflow Project
    run = mlflow.active_run()
    print("✅ Active Run ID:", run.info.run_id)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # ✅ MODEL DISIMPAN DI RUN YANG BENAR
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="TelcoChurnModel"
    )

    # Artefak tambahan
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    print(f"🎯 Accuracy: {acc:.4f}")
    print("📦 Model & artefak sukses dicatat")

if __name__ == "__main__":
    train_basic_model()

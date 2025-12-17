import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from mlflow.tracking import MlflowClient # Tambahkan ini untuk manajemen registry

print("🚀 CI Training (Production Mode)")

# ======================
# DagsHub Init
# ======================
dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

# ======================
# Attach to EXISTING run (from MLflow Project)
# ======================
run_id = os.environ.get("MLFLOW_RUN_ID")
if run_id is None:
    raise RuntimeError("MLFLOW_RUN_ID not found. This script must be run via mlflow run.")

with mlflow.start_run(run_id=run_id):
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
    # Fixed Best Params
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

    # ======================
    # Log & Register Model
    # ======================
    # Tambahkan parameter registered_model_name agar model masuk ke Registry
    model_name = "Modelling-Advance"
    model_info = mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model",
        registered_model_name=model_name
    )

    # ======================
    # Transition to Production
    # ======================
    # Bagian ini penting agar CI Pipeline bisa menemukan model di stage 'Production'
    client = MlflowClient()
    model_version = model_info.registered_model_version
    
    print(f"Mendaftarkan model {model_name} versi {model_version} ke stage Production...")
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=True # Mengarsip versi Production lama jika ada
    )

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

    fi_path = "artifacts/feature_importance.png"
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.savefig(fi_path)
    plt.close()

    mlflow.log_artifact(fi_path)

print(f"✅ CI training & mendaftarkan model ke Production selesai!")
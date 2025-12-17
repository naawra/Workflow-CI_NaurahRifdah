import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os
import tempfile  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Inisialisasi DagsHub (Pastikan username & repo benar)
dagshub.init(repo_owner='naawra', repo_name='Submission_Eksperimen_SML_NaurahRifdah', mlflow=True)

def train_basic_model():
    print("Memulai Training Basic (Mode Clean & Safe)...")

    print("Loading Data...")
    train_df = pd.read_csv('telco_churn_preprocessed/train_data.csv')
    test_df = pd.read_csv('telco_churn_preprocessed/test_data.csv')

    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']

    mlflow.autolog(log_models=False)

    print("Training Model...")
    mlflow.set_tag("mlflow.runName", "Basic_RandomForest_Clean")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # 2. INI YANG PENTING: Simpan model dengan nama folder "model"
    print("Logging Model secara eksplisit ke MLflow...")
    mlflow.sklearn.log_model(model, artifact_path="model")
        
    # print("Uploading Model Backup (Optional)...")
    # with tempfile.TemporaryDirectory() as temp_dir:
    #         temp_model_path = os.path.join(temp_dir, "model_basic.joblib")
    #         joblib.dump(model, temp_model_path)
    #         mlflow.log_artifact(temp_model_path)
            
    print("Training Basic Selesai! Model sudah direkam.")

if __name__ == "__main__":
    train_basic_model()
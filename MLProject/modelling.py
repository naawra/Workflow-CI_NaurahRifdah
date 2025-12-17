import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os
import tempfile  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

dagshub.init(repo_owner='naawra', repo_name='Submission_Eksperimen_SML_NaurahRifdah', mlflow=True)

mlflow.set_experiment("Modelling-Advance")

def train_tuning():
    print("Memulai Training Advance (Clean Upload)...")
    
    # Load Data
    train_df = pd.read_csv('telco_churn_preprocessed/train_data.csv')
    test_df = pd.read_csv('telco_churn_preprocessed/test_data.csv')

    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']

    with mlflow.start_run(run_name="Advance_Final_Clean"):
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        
        print("⏳ Sedang Tuning...")
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Akurasi: {acc:.4f}")

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # print("Membuat Artefak di Folder Sementara...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            cm_path = os.path.join(temp_dir, "confusion_matrix.png")
            fig_cm = plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.savefig(cm_path) 
            plt.close(fig_cm)
            mlflow.log_artifact(cm_path) 
            
            fi_path = os.path.join(temp_dir, "feature_importance.png")
            fig_fi = plt.figure(figsize=(8, 6))
            importances = best_model.feature_importances_
            plt.bar(range(len(importances)), importances)
            plt.savefig(fi_path) 
            plt.close(fig_fi)
            mlflow.log_artifact(fi_path) 

            model_path = os.path.join(temp_dir, "model_telco_churn.joblib")
            joblib.dump(best_model, model_path) 
            mlflow.log_artifact(model_path) 
            
            print("Upload Selesai!")
        
if __name__ == "__main__":
    train_tuning()
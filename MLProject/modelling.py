import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

dagshub.init(
    repo_owner="naawra",
    repo_name="Submission_Eksperimen_SML_NaurahRifdah",
    mlflow=True
)

def run_advanced_tuning():
    print("Menjalankan Tuning (Model + Gambar)...")
    
    train_df = pd.read_csv("telco_churn_preprocessed/train_data.csv")
    test_df = pd.read_csv("telco_churn_preprocessed/test_data.csv")
    X_train, y_train = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]

    # Param Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10]
    }
    
    # Mulai Run
    with mlflow.start_run(run_name="Advance_Tuning_Complete"):
        
        print("Sedang mencari hyperparameter terbaik...")
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        os.makedirs("artifacts", exist_ok=True)

        plt.figure(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlGnBu")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        plt.figure(figsize=(8,5))

        feat_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("artifacts/feature_importance.png")
        plt.close()
        mlflow.log_artifact("artifacts/feature_importance.png")
        
        try:
            mlflow.sklearn.log_model(best_model, "model_folder")
        except Exception:
            pass 

        model_filename = "artifacts/best_model_tuned.pkl"
        joblib.dump(best_model, model_filename)
        mlflow.log_artifact(model_filename)
        

if __name__ == "__main__":
    run_advanced_tuning()
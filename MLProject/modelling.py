import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt # Tamabahan buat gambar
import seaborn as sns           # Tambahan buat gambar bagus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix # Tambah confusion matrix
import os

mlflow.set_tracking_uri("")
mlflow.set_experiment("Eksperimen_Lokal_Nawra")

mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)

def load_data():
    train_path = 'telco_churn_preprocessed/train_data.csv'
    test_path = 'telco_churn_preprocessed/test_data.csv'

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {train_path} !")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_test = test_df.drop('Churn', axis=1)
    y_test = test_df['Churn']

    return X_train, X_test, y_train, y_test

def train_basic_model():
    try:
        X_train, X_test, y_train, y_test = load_data()

        # Inisialisasi Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        print("Mulai training dengan Autolog + Artifak Gambar... ⏳")
        
        with mlflow.start_run(run_name="Run_Autolog_Plus_Artifacts"):
            
            # Autolog bekerja di sini saat fit()
            model.fit(X_train, y_train)

            # Prediksi manual buat bikin grafik
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")

            
            # A. Gambar 1: Confusion Matrix
            print("📸 Membuat Confusion Matrix...")
            plt.figure(figsize=(6,5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Simpan jadi file gambar 
            plt.savefig("confusion_matrix.png") 
            plt.close() 
            
            # Upload ke MLflow sebagai Artifak
            mlflow.log_artifact("confusion_matrix.png")

            # B. Gambar 2: Feature Importance
            print("📸 Membuat Feature Importance...")
            plt.figure(figsize=(8,6))
            feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            feat_importances.nlargest(10).plot(kind='barh') # Ambil top 10
            plt.title("Top 10 Feature Importance")
            
            # Simpan jadi file gambar
            plt.savefig("feature_importance.png")
            plt.close()
            
            # Upload ke MLflow
            mlflow.log_artifact("feature_importance.png")

    

    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    train_basic_model()
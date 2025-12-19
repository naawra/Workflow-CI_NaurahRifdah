import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Set URI ke folder lokal 'mlruns'
mlflow.set_tracking_uri("")

# Set Nama Eksperimen
mlflow.set_experiment("Eksperimen_Lokal_Nawra")

# --- KUNCI UTAMA: AKTIFKAN AUTOLOG DI SINI ---
# Cukup satu baris ini, dia akan menghandle logging params, metrics, dan model.
mlflow.sklearn.autolog(
    log_models=True,       # Wajib simpan modelnya
    log_input_examples=True, # Simpan contoh data (opsional tapi bagus)
    log_model_signatures=True
)

def load_data():
    train_path = 'telco_churn_preprocessed/train_data.csv'
    test_path = 'telco_churn_preprocessed/test_data.csv'

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {train_path} !")

    print("Memuat data... 📂")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = 'Churn' 
    
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test

def train_basic_model():
    try:
        X_train, X_test, y_train, y_test = load_data()

        # Inisialisasi Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        print("Mulai training dengan Autolog... ⏳")
        
        # Mulai MLflow Run
        # Kita tetap pakai start_run biar bisa kasih nama Run yang rapi
        with mlflow.start_run(run_name="Run_Autolog_RandomForest"):
            
            # --- MAGIC HAPPENS HERE ---
            # Karena autolog() sudah aktif di atas, saat .fit() jalan, 
            # MLflow otomatis nyatet semua (param, metrik, model) ke folder mlruns.
            model.fit(X_train, y_train)

            # Prediksi (Hanya untuk print terminal, tidak perlu dilog manual lagi)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"\n--- Hasil Training ---")
            print(f"Accuracy (Terminal): {acc:.4f}")
            
            print("\n✅ SUKSES! Autolog sudah bekerja.")
            print("Cek folder 'mlruns' atau buka 'mlflow ui'.")

    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    train_basic_model()
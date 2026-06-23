# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Config
st.set_page_config(page_title="ML Studio Wow", layout="wide")
st.title("ML Studio Wow  🚀")

# -------------------------
# Sidebar settings
st.sidebar.header("1. Upload Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
target_col = st.sidebar.text_input("Nama kolom target (label)", value="")
task_type = st.sidebar.selectbox("Tipe task", ["Auto detect", "Classification", "Regression"])
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("2. Model & Training")
model_choice = st.sidebar.selectbox("Pilih model", ["RandomForest", "XGBoost", "LightGBM", "NeuralNet"])
do_optuna = st.sidebar.checkbox("Hyperparameter tuning (Optuna)", value=False)
n_trials = st.sidebar.number_input("Optuna trials", min_value=10, max_value=200, value=30)
run_train = st.sidebar.button("Train Model")

# -------------------------
# Utility functions
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)

def infer_task(y_series, user_choice):
    if user_choice == "Classification":
        return "classification"
    if user_choice == "Regression":
        return "regression"
    # auto detect
    if y_series.dtype.kind in "biufc" and y_series.nunique() > 20:
        return "regression"
    else:
        return "classification"

def preprocess(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    # simple numeric-only pipeline for demo
    X_num = X.select_dtypes(include=[np.number]).copy()
    # fillna
    X_num = X_num.fillna(X_num.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    return X_scaled, y, scaler, X_num.columns.tolist()

# -------------------------
# Main flow
if uploaded is None:
    st.info("Upload dataset CSV di sidebar untuk mulai.")
    st.stop()

df = load_data(uploaded)
st.subheader("Preview Data")
st.dataframe(df.head())

if target_col == "":
    st.warning("Isi nama kolom target di sidebar untuk melanjutkan.")
    st.stop()

if target_col not in df.columns:
    st.error("Kolom target tidak ditemukan di dataset.")
    st.stop()

# determine task
task = infer_task(df[target_col], task_type)
st.write(f"**Detected task**: {task}")

# preprocess
X, y, scaler, feature_names = preprocess(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

# model factory
def build_model(name, task):
    if name == "RandomForest":
        if task == "classification":
            return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
        else:
            return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    if name == "XGBoost":
        if task == "classification":
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=random_state)
        else:
            return xgb.XGBRegressor(n_jobs=-1, random_state=random_state)
    if name == "LightGBM":
        if task == "classification":
            return lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
        else:
            return lgb.LGBMRegressor(random_state=random_state, n_jobs=-1)
    if name == "NeuralNet":
        # simple MLP using sklearn wrapper for demo
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        if task == "classification":
            return MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=random_state)
        else:
            return MLPRegressor(hidden_layer_sizes=(128,64), max_iter=500, random_state=random_state)

# Optuna objective example for RandomForest only (extendable)
def optuna_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20)
    }
    model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1) if task=="classification" else RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if task == "classification":
        score = accuracy_score(y_test, preds)
    else:
        score = -mean_squared_error(y_test, preds)
    return score

# Training
if run_train:
    st.info("Training dimulai. Ini bisa memakan waktu tergantung ukuran data dan model.")
    model = build_model(model_choice, task)

    if do_optuna:
        st.write("Running Optuna tuning...")
        study = optuna.create_study(direction="maximize" if task=="classification" else "maximize")
        study.optimize(optuna_objective, n_trials=int(n_trials))
        st.write("Best params:", study.best_params)
        # rebuild model with best params for RF only; for other models extend accordingly
        if model_choice == "RandomForest":
            model = RandomForestClassifier(**study.best_params, random_state=random_state, n_jobs=-1) if task=="classification" else RandomForestRegressor(**study.best_params, random_state=random_state, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    if task == "classification":
        acc = accuracy_score(y_test, preds)
        try:
            proba = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = None
        st.success(f"Accuracy: {acc:.4f}")
        if auc is not None:
            st.success(f"AUC: {auc:.4f}")
    else:
        mse = mean_squared_error(y_test, preds)
        st.success(f"MSE: {mse:.4f}")

    # SHAP explainability for tree models
    if model_choice in ["RandomForest", "XGBoost", "LightGBM"]:
        st.subheader("Model Explainability SHAP")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        st.write("SHAP summary plot")
        fig_shap = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches="tight")
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_choice}_{task}.joblib")
    joblib.dump({"model": model, "scaler": scaler, "features": feature_names}, model_path)
    st.write(f"Model disimpan di `{model_path}`. Download file model dari repo untuk produksi.")

# -------------------------
# Prediction UI
st.sidebar.markdown("---")
st.sidebar.header("3. Predict Single Row")
if st.sidebar.button("Load latest model"):
    # try to load last saved model
    try:
        files = sorted([f for f in os.listdir("models") if f.endswith(".joblib")])
        latest = files[-1]
        data = joblib.load(os.path.join("models", latest))
        st.sidebar.success(f"Loaded {latest}")
        loaded_model = data["model"]
        loaded_scaler = data["scaler"]
        loaded_features = data["features"]
        # build input form
        st.sidebar.write("Masukkan fitur numerik:")
        input_vals = {}
        for f in loaded_features:
            input_vals[f] = st.sidebar.number_input(f, value=0.0, format="%.4f")
        if st.sidebar.button("Predict"):
            X_in = np.array([list(input_vals.values())])
            X_in_scaled = loaded_scaler.transform(X_in)
            pred = loaded_model.predict(X_in_scaled)
            st.sidebar.write("Prediksi:", pred)
    except Exception as e:
        st.sidebar.error("Tidak ada model tersimpan atau folder models kosong.")

# Footer
st.markdown("---")
st.caption("Tip: Untuk dataset besar gunakan cloud VM dan simpan model ke object storage. Untuk produksi gunakan API terpisah.")

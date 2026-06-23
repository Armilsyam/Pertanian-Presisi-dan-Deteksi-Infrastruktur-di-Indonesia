# train.py
import argparse
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

def load_data(path):
    return pd.read_csv(path)

def preprocess(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_num = X.select_dtypes(include=[np.number]).fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    return X_scaled, y, scaler, X_num.columns.tolist()

def main(args):
    df = load_data(args.data)
    X, y, scaler, features = preprocess(df, args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    if args.task == "classification":
        model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=args.seed)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=args.seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if args.task == "classification":
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        try:
            proba = model.predict_proba(X_test)[:,1]
            print("AUC:", roc_auc_score(y_test, proba))
        except Exception:
            pass
    else:
        print("MSE:", mean_squared_error(y_test, preds))
    os.makedirs("model_store", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": features, "task": args.task}, "model_store/model.joblib")
    print("Model saved to model_store/model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["classification","regression"], default="classification")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)

import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_scaler_params(scaler, features):
    data = {
        "features": list(features),
        "mu": scaler.mean_.tolist(),
        "sigma": scaler.scale_.tolist(),
    }
    with open(base_path + "/data/split/scaler_params.txt", "w", encoding="utf-8") as f:
        json.dump(data, f)


def run_preprocessing():

    selected_features = [
        "texture3",
        "perimeter3",
        "smoothness3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]
    split_path = base_path + "/data/split/"
    train_path = pd.read_csv(split_path + "train.csv")
    valid_path = pd.read_csv(split_path + "valid.csv")

    X_train = train_path[selected_features]
    y_train = train_path["diagnosis"]

    X_valid = valid_path[selected_features]
    y_valid = valid_path["diagnosis"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_valid_scaled_df = pd.DataFrame(X_valid_scaled, columns=selected_features)

    train_scaled = pd.concat([X_train_scaled_df, y_train], axis=1)
    valid_scaled = pd.concat([X_valid_scaled_df, y_valid], axis=1)
    train_scaled.to_csv(split_path + "train_scaled.csv", index=False)
    valid_scaled.to_csv(split_path + "valid_scaled.csv", index=False)

    save_scaler_params(scaler, selected_features)


if __name__ == "__main__":
    run_preprocessing()

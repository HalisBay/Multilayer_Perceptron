import numpy as np
import pandas as pd
import json
from train import forward_propagation


def load_model(path):
    data = np.load(path, allow_pickle=True)

    W = data["weights"]
    b = data["biases"]

    W = [w for w in W]
    b = [bias for bias in b]

    return W, b


def transform_with_saved_scaler(X_df):
    with open("data/split/scaler_params.txt", "r", encoding="utf-8") as f:
        params = json.load(f)

    features = list(params["features"])
    mean = np.array(params["mean"], dtype=float)
    scale = np.array(params["scale"], dtype=float)
    scale = np.where(scale == 0, 1, scale)

    X_ordered = X_df[features].copy()
    X_scaled = (X_ordered.values - mean) / scale
    return pd.DataFrame(X_scaled, columns=features)


if __name__ == "__main__":

    selected_features = [
        "texture3",
        "perimeter3",
        "smoothness3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]

    W, b = load_model("saved_model.npz")
    data = pd.read_csv("data/raw/wdbc.csv")

    if all(col in data.columns for col in selected_features):
        X_df = data[selected_features].copy()
    else:
        if "diagnosis" in data.columns:
            X_df = data.drop("diagnosis", axis=1).copy()
        else:
            X_df = data.copy()

    X_scaled_df = transform_with_saved_scaler(X_df)
    X = X_scaled_df.values

    activations, _ = forward_propagation(X, W, b)
    probs = activations[-1]
    predictions = np.argmax(probs, axis=1)

    if "diagnosis" in data.columns:
        y_true_raw = data["diagnosis"].values
        classes = np.unique(y_true_raw)
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        y_true = np.array([class_to_idx[label] for label in y_true_raw], dtype=np.int64)
        accuracy = np.mean(predictions == y_true) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print(predictions)

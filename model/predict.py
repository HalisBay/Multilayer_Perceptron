import numpy as np
import pandas as pd
import json
from train import forward_propagation
import os


def get_scaled_params(path):
    with open(path, "r") as txt:
        params = json.load(txt)
    X = params["features"]
    mu = np.array(params["mu"])
    sigma = np.array(params["sigma"])

    return X, mu, sigma


def compute_bce_loss(y, y_probs):
    epsilon = 1e-15
    p = np.clip(y_probs, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_model(path):
    data = np.load(path, allow_pickle=True)

    W = data["weights"]
    b = data["biases"]
    layers = data["layers"]
    return W, b, layers


def scale_data(X, mu, sigma):
    return (X - mu) / sigma


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    W, b, layers = load_model(base_path + "/saved_model.npz")
    data = pd.read_csv(base_path + "/data/raw/data.csv")
    features, mu, sigma = get_scaled_params(base_path + "/data/split/scaler_params.txt")

    selected_X = data[features].values
    scaled_X = scale_data(selected_X, mu, sigma)

    activations, _ = forward_propagation(scaled_X, W, b)
    probs = activations[-1]

    yhat = np.argmax(probs, axis=1)
    y = data["diagnosis"].values
    result = pd.DataFrame({"yhat": yhat, "y": y})
    bce_value = compute_bce_loss(y, probs[:, 1])
    print(result)
    print(f"% {np.mean(y == yhat) * 100:.2f}")
    print(f"% {bce_value}")

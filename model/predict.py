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


def load_model(path):
    data = np.load(path, allow_pickle=True)

    W = data["weights"]
    b = data["biases"]
    return W, b


def scale_data(X, mu, sigma):
    return (X - mu) / sigma


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    W, b = load_model(base_path + "/saved_model.npz")
    data = pd.read_csv(base_path + "/data/raw/wdbc.csv")
    features, mu, sigma = get_scaled_params(base_path + "/data/split/scaler_params.txt")

    selected_X = data[features].values
    scaled_X = scale_data(selected_X, mu, sigma)

    activations, _ = forward_propagation(scaled_X, W, b)
    probs = activations[-1]

    yhat = np.argmax(probs, axis=1)
    y = data["diagnosis"].values
    print(f"% {np.mean(y == yhat) * 100:.2f}")

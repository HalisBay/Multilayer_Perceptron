import argparse
import pandas as pd
import numpy as np


def init_params_with_He(layer_size):
    # Relu aktivasyonu kullanacağım için Xavier yerine HeUniform initialization seçtim;
    # çünkü XavierUniform sigmoid/tanh ile , HeUniform ise ReLU ile daha optimize çalışır.
    # W ~ U(-√(6 / fan_in), +√(6 / fan_in))
    weights = []
    biasses = []
    for i in range(len(layer_size) - 1):
        limit = np.sqrt(6 / layer_size[i])

        W = np.random.uniform(-limit, limit, (layer_size[i], layer_size[i + 1]))
        b = np.zeros((1, layer_size[i + 1]))

        weights.append(W)
        biasses.append(b)
    return weights, biasses


def forward_propagation(X, W, b):
    print("Burada kaldım")


def main_process(
    layers,
    layer_size,
    epochs=84,
    loss="categoricalCrossentropy",
    batch_size=8,
    learning_rate=0.05,
):
    W, b = init_params_with_He(layer_size=layer_size)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    args = parser.parse_args()
    layers = args.layer
    epochs = args.epochs
    loss = args.loss
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    data = pd.read_csv("data/split/train_scaled.csv")
    X = data.drop("diagnosis", axis=1).values
    y = data["diagnosis"].values
    input_size = X.shape[1]
    output_size = len(np.unique(y))


    layer_size = [input_size] + layers + [output_size]
    main_process(layers, layer_size, epochs, loss, batch_size, learning_rate)

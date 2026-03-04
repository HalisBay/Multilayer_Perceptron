import argparse
import pandas as pd
import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def derivative_relu(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    # Softmax(x(i))=exp(xi) / ∑(j) exp(x(j))

    # Overflow riskinden  dolayı her satırdan max'ı çıkarıyorum
    shiftZ = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(shiftZ)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def init_params_with_He(layer_size):
    # Relu aktivasyonu kullanacağım için Xavier yerine HeUniform initialization seçtim;
    # çünkü XavierUniform sigmoid/tanh ile , HeUniform ise ReLU ile daha optimize çalışır.
    # W ~ U(-√(6 / fan_in), +√(6 / fan_in))
    weights = []
    biases = []
    for i in range(len(layer_size) - 1):
        limit = np.sqrt(6 / layer_size[i])

        W = np.random.uniform(-limit, limit, (layer_size[i], layer_size[i + 1]))
        b = np.zeros((1, layer_size[i + 1]))

        weights.append(W)
        biases.append(b)
    return weights, biases


def forward_propagation(X, W, b):
    """
    X    - input matrix
    Zh   - hidden layer weighted input
    Zo   - output layer weighted input
    H    - hidden layer activations
    yHat - output layer predictions
    """

    activations = [X]
    z_scores = []
    H = X
    for i in range(len(W) - 1):
        Zh = H @ W[i] + b[i]
        H = relu(Zh)
        z_scores.append(Zh)
        activations.append(H)

    Zo = H @ W[-1] + b[-1]
    yHat = softmax(Zo)
    z_scores.append(Zo)
    activations.append(yHat)

    return activations, z_scores


def back_propagation(y, activations, z_scores, W, b, learning_rate):

    dwh = [0] * len(W)
    dbh = [0] * len(b)
    eh = [None] * len(W)
    batch = y.shape[0]

    eh[-1] = activations[-1] - y
    for i in reversed(range(len(W) - 1)):
        eh[i] = (eh[i + 1] @ W[i + 1].T) * derivative_relu(z_scores[i])

    for i in range(len(W)):
        dwh[i] = activations[i].T @ eh[i] / batch
        dbh[i] = np.sum(eh[i], axis=0, keepdims=True) / batch

    for i in range(len(W)):
        W[i] -= dwh[i] * learning_rate
        b[i] -= dbh[i] * learning_rate
    return W, b


def main_process(
    layers,
    layer_size,
    X,
    y,
    epochs=84,
    loss="categoricalCrossentropy",
    batch_size=8,
    learning_rate=0.05,
):
    W, b = init_params_with_He(layer_size=layer_size)
    activations, z_scores = forward_propagation(X, W, b)
    back_propagation(y, activations, z_scores, W, b, learning_rate)
    # TODO: Train ve loss function fonksiyonu oluşturulacak, kaynak dosyalar ile gerekli anlatımlar yapılacak.

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
    y_raw = data["diagnosis"].values
    input_size = X.shape[1]
    output_size = len(np.unique(y_raw))

    # Categorical softmax+cross-entropy için one hot çevirdim
    y_int, _ = pd.factorize(y_raw)
    y = np.eye(output_size)[y_int]

    layer_size = [input_size] + layers + [output_size]
    main_process(layers, layer_size, X, y, epochs, loss, batch_size, learning_rate)

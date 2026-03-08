import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


def compute_loss(m, y, y_pred):
    # L=− 1/m ​∑ylog(yhat​)
    epsilon = 1e-15  # tam olarak 0 ve 1 olmaması için, aksi takdirde log(0) olursa -inf
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -1 * np.sum(y * np.log(y_pred)) / m


def batch_maker(X, y, batch_size):
    m = X.shape[0]

    idx = np.arange(m)
    np.random.shuffle(idx)
    for i in range(0, m, batch_size):
        end = i + batch_size
        batch = idx[i:end]
        yield X[batch], y[batch]


def train(layer_size,epochs,W,b,X,y,X_valid,y_valid,learning_rate,batch_size,patience,min_delta):
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    best_valid_loss = np.inf
    best_W = [weights.copy() for weights in W]
    best_b = [bias.copy() for bias in b]
    wait = 0

    for e in range(epochs):
        for X_batch, y_batch in batch_maker(X, y, batch_size):
            activations, z_scores = forward_propagation(X_batch, W, b)
            W, b = back_propagation(y_batch, activations, z_scores, W, b, learning_rate)

        train_activations, train_z_scores = forward_propagation(X, W, b)
        train_loss = compute_loss(y.shape[0], y, train_activations[-1])
        train_y_hat = np.argmax(train_activations[-1], axis=1)
        train_y = np.argmax(y, axis=1)
        train_acc = np.mean(train_y_hat == train_y)

        valid_activations, valid_z_scores = forward_propagation(X_valid, W, b)
        valid_loss = compute_loss(y_valid.shape[0], y_valid, valid_activations[-1])
        valid_y_hat = np.argmax(valid_activations[-1], axis=1)
        valid_y = np.argmax(y_valid, axis=1)
        valid_acc = np.mean(valid_y_hat == valid_y)
        print(
            f"epoch: {e +1} / {epochs} | train_loss : {train_loss} |  valid_loss {valid_loss}\n"
             + f" train accuracy {train_acc * 100 } | valid_acc: {valid_acc * 100} "
        )

        if best_valid_loss - valid_loss > min_delta:
            best_valid_loss = valid_loss
            best_W = [weights.copy() for weights in W]
            best_b = [bias.copy() for bias in b]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"Early stopping at epoch {e + 1}. Best valid_loss: {best_valid_loss}"
                )
                break

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

    weights_to_save = np.empty(len(best_W), dtype=object)
    weights_to_save[:] = best_W
    biases_to_save = np.empty(len(best_b), dtype=object)
    biases_to_save[:] = best_b
    np.savez(
        "saved_model.npz",
        weights=weights_to_save,
        biases=biases_to_save,
        layers=np.array(layer_size, dtype=np.int64),
        allow_pickle=True,
    )

    return train_losses, valid_losses, train_accs, valid_accs


def plot_datas(train_losses, valid_losses, train_accs, valid_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(valid_losses, label="Valid Loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label="Train Accuracy")
    ax2.plot(valid_accs, label="Valid Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", nargs="+", type=int, default=[24, 24])
    parser.add_argument("--epochs", type=int, default=84)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    args = parser.parse_args()
    layers = args.layer
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    patience = args.patience
    min_delta = args.min_delta

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = base_path + "/data/split/"
    data = pd.read_csv(data_path + "train_scaled.csv")
    valid_data = pd.read_csv(data_path + "valid_scaled.csv")

    X = data.drop("diagnosis", axis=1).values
    X_valid = valid_data.drop("diagnosis", axis=1).values

    y_raw = data["diagnosis"].values
    y_valid_raw = valid_data["diagnosis"].values

    input_size = X.shape[1]
    classes = np.unique(y_raw)
    output_size = len(classes)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    # Categorical softmax+cross-entropy için one hot çevirdim
    y_int = np.array([class_to_idx[label] for label in y_raw], dtype=np.int64)
    y = np.eye(output_size)[y_int]

    y_valid_int = np.array(
        [class_to_idx[label] for label in y_valid_raw], dtype=np.int64
    )
    y_valid = np.eye(output_size)[y_valid_int]

    layer_size = [input_size] + layers + [output_size]

    W, b = init_params_with_He(layer_size=layer_size)
    train_losses, valid_losses, train_accs, valid_accs = train(layer_size,epochs,W,b,X,y,X_valid,y_valid,learning_rate,batch_size,patience,min_delta)
    plot_datas(train_losses, valid_losses, train_accs, valid_accs)

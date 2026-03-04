# Multilayer Perceptron – Mathematical Foundations

## 1. Forward Propagation

A fully connected (dense) layer computes two operations.

### Step 1 — Weighted Sum

$$
Z = XW + b
$$

Where:

- $X \in \mathbb{R}^{(\text{batch\_size},\,\text{input\_size})}$
- $W \in \mathbb{R}^{(\text{input\_size},\,\text{output\_size})}$
- $b \in \mathbb{R}^{(1,\,\text{output\_size})}$

Result:

$$
Z \in \mathbb{R}^{(\text{batch\_size},\,\text{output\_size})}
$$

### Example

If:

- Batch size = 8
- Input size = 4
- Hidden layer neurons = 5

Then:

```text
X  = (8,4)
W1 = (4,5)
b1 = (1,5)
```

Forward pass:

```text
Z1 = X @ W1 + b1     → (8,5)
```

---

### Worked small example — explicit dot products (batch size 2, hidden 2)

1. Matrices (numeric):

```text
  X  = [[1, 2, 3],
      [4, 5, 6]]            # shape (2,3)

  W1 = [[0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]]          # shape (3,2)

b1 = [[0.01, 0.02]]         # shape (1,2)
```

2. Dot-product step — row 1 (first sample):

- Hidden neuron 1 (first column):

    1 * 0.1 + 2 * 0.3 + 3 * 0.5 = 0.1 + 0.6 + 1.5 = 2.2

- Hidden neuron 2 (second column):

    1 * 0.2 + 2 * 0.4 + 3 * 0.6 = 0.2 + 0.8 + 1.8 = 2.8

Add bias: [2.2, 2.8] + [0.01, 0.02] = [2.21, 2.82]

3. Dot-product step — row 2 (second sample):

- Hidden neuron 1:

    4 * 0.1 + 5 * 0.3 + 6 * 0.5 = 0.4 + 1.5 + 3.0 = 4.9

- Hidden neuron 2:

    4 * 0.2 + 5 * 0.4 + 6 * 0.6 = 0.8 + 2.0 + 3.6 = 6.4

Add bias: [4.9, 6.4] + [0.01, 0.02] = [4.91, 6.42]

4. Final pre-activation matrix Z:

```text
  Z = [[2.21, 2.82],
          [4.91, 6.42]]    # shape (2,2)
```

Apply activation element-wise (e.g., sigmoid) to get A of shape (2,2).

This explicit walkthrough helps verify dimension consistency and implementation correctness when coding the forward pass.

### Step 2 — Activation Function

$$
A = f(Z)
$$

Example with sigmoid activation:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Output:

```text
A1 = sigmoid(Z1) → (8,5)
```
Numeric sigmoid example (apply to the worked example Z):

```text
Z = [[2.21, 2.82],
    [4.91, 6.42]]

sigmoid(2.21) ≈ 0.90
sigmoid(2.82) ≈ 0.94
sigmoid(4.91) ≈ 0.99
sigmoid(6.42) ≈ 0.998

A = [[0.900, 0.944],
    [0.993, 0.998]]    # shape (2,2)
```

Note on activation outputs and classification:

- Sigmoid (and most activation functions) return continuous values, not discrete 0/1 labels. For binary classification with a single sigmoid output, convert to a class label by thresholding (e.g., `label = a >= 0.5`).
- Alternatively, use a 2-unit output with `softmax` and take `argmax` to select the class.
- When training, do not round activations before computing loss — use the continuous outputs with the appropriate loss (binary cross-entropy for sigmoid, categorical cross-entropy for softmax).

---

## 2. Sigmoid Derivative

The sigmoid derivative simplifies to:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

In implementation (we use the already computed activation value $a = \sigma(z)$):

```python
def sigmoid_derivative(a):
    return a * (1 - a)
```

Numeric example (use $A$ from the worked example):

```text
A = [[0.900, 0.944],
     [0.993, 0.998]]    # shape (2,2)

sigma'(0.900) = 0.900 * (1 - 0.900) = 0.0900
sigma'(0.944) = 0.944 * (1 - 0.944) = 0.0529
sigma'(0.993) = 0.993 * (1 - 0.993) = 0.00695
sigma'(0.998) = 0.998 * (1 - 0.998) = 0.001996

sigmoid_deriv = [[0.0900, 0.0529],
                 [0.00695, 0.001996]]
```

Using the derivative in backprop (concept): if the next-layer delta for the same shape is `delta_next`, the hidden-layer delta is computed element-wise as:

```python
# element-wise
delta_hidden = delta_next * sigmoid_deriv
```

Example (toy numbers):

```text
delta_next = [[0.1, 0.2],
              [0.05, 0.01]]

delta_hidden = [[0.1*0.0900, 0.2*0.0529],
                [0.05*0.00695, 0.01*0.001996]]

             = [[0.00900, 0.01058],
                [0.0003475, 0.00001996]]
```

This shows how the sigmoid derivative rescales the propagated error per unit.

---

## 3. Output Layer — Softmax + Cross-Entropy

### Softmax

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax transforms logits into a probability distribution over classes.

### Cross-Entropy Loss

$$
L = -\sum_i y_i \log(p_i)
$$

### Important Property

When softmax and cross-entropy are combined, the gradient of the loss with respect to the logits simplifies to:

$$
\frac{\partial L}{\partial Z} = A - Y
$$

This simplifies backpropagation for the output layer.


Numeric example (continuation from section 1/2 worked example, batch size 2, 3-class output):

We continue with the same hidden activations:

```text
A1 = [[0.900, 0.944],
    [0.993, 0.998]]      # shape (2,2)
```

Choose output-layer parameters:

```text
W2 = [[ 0.20, -0.10,  0.05],
    [ 0.30,  0.40, -0.20]]   # shape (2,3)

b2 = [[0.01, -0.02, 0.03]]     # shape (1,3)
```

1) Logits from the same forward chain:

```text
Z2 = A1 @ W2 + b2
   ≈ [[0.4732, 0.2676, -0.1138],
    [0.5080, 0.2799, -0.1210]]   # shape (2,3)
```

2) Stable softmax (per row):

```text
z1_stable = [0.0000, -0.2056, -0.5870]
exp(z1_stable) ≈ [1.0000, 0.8142, 0.5560]
p1 ≈ [0.4219, 0.3435, 0.2346]

z2_stable = [0.0000, -0.2281, -0.6290]
exp(z2_stable) ≈ [1.0000, 0.7961, 0.5332]
p2 ≈ [0.4293, 0.3418, 0.2289]

P = [[0.4219, 0.3435, 0.2346],
     [0.4293, 0.3418, 0.2289]]      # shape (2,3)
```

3) Labels and cross-entropy loss (same batch):

```text
Y = [[1, 0, 0],
     [0, 1, 0]]

L1 = -log(0.4219) ≈ 0.8630
L2 = -log(0.3418) ≈ 1.0735
L_batch_mean = (L1 + L2) / 2 ≈ 0.9683
```

4) Output gradient (softmax + cross-entropy simplification):

```text
delta2 = dZ2 = P - Y
    ≈ [[-0.5781,  0.3435, 0.2346],
       [ 0.4293, -0.6582, 0.2289]]   # shape (2,3)
```

Interpretation:

- These values are a direct continuation of the earlier `A1` matrix from the worked forward pass.
- The `delta2` computed here is reused directly in the next backprop and gradient sections.

---

## 4. Backpropagation

### Output Layer

$$
\delta^{L} = A^{L} - Y
$$

### Hidden Layers

$$
\delta^{l} = (W^{l+1})^{\top} \delta^{l+1} \odot f'(Z^{l})
$$

Where $\odot$ denotes element-wise multiplication and $f'(Z^{l})$ is the activation derivative.

---

Numeric backprop example (continuation using `delta2` from section 3):

We use the worked example values:

```text
X  = [[1, 2, 3],
      [4, 5, 6]]          # shape (2,3)

Z (pre-activation) = [[2.21, 2.82],
                      [4.91, 6.42]]  # shape (2,2)

A (sigmoid) = [[0.900, 0.944],
              [0.993, 0.998]]      # shape (2,2)

sigmoid_deriv = [[0.0900, 0.0529],
                [0.00695, 0.001996]]
```

From section 3, we already have:

```text
delta2 = [[-0.5781,  0.3435, 0.2346],
             [ 0.4293, -0.6582, 0.2289]]   # shape (2,3)

W2 = [[ 0.20, -0.10,  0.05],
        [ 0.30,  0.40, -0.20]]           # shape (2,3)
```

1) Propagate to hidden pre-activation space:

```text
delta_next = delta2 @ W2.T
             ≈ [[-0.13824, -0.08295],
                 [ 0.16313, -0.18027]]     # shape (2,2)
```

2) Hidden-layer delta (element-wise with sigmoid derivative):

```text
delta_hidden = delta_next * sigmoid_deriv
                ≈ [[-0.01244, -0.00439],
                    [ 0.00113, -0.00036]]  # shape (2,2)
```

3) Weight gradients for W1 (W1 shape = (3,2)):

```text
dW = X^T @ delta_hidden
    ≈ [[-0.00791, -0.00582],
        [-0.01921, -0.01057],
        [-0.03052, -0.01532]]             # shape (3,2)
```

4) Bias gradients (sum over batch):

```text
db = sum(delta_hidden, axis=0, keepdims=True)
    ≈ [[-0.01131, -0.00475]]          # shape (1,2)
```

If your loss is averaged over the batch, divide gradients by batch size (2):

```text
dW_mean = dW / 2 ≈ [[-0.00395, -0.00291],
                          [-0.00961, -0.00529],
                          [-0.01526, -0.00766]]

db_mean = db / 2 ≈ [[-0.00565, -0.00237]]
```

5) Parameter update (gradient descent, learning rate $\eta$):

```text
W1_new = W1 - eta * dW_mean
b1_new = b1 - eta * db_mean
```

This shows a full local backprop step in one chain: `delta2` (section 3) → `delta_next` → `delta_hidden` → gradients `dW1` and `db1`.

## 5. Gradient Computation

Given shapes:

```text
A1 = (8,5)
delta2 = (8,3)
W2 = (5,3)
```

### Weight Gradient

$$
\frac{\partial L}{\partial W^{2}} = (A^{1})^{\top} \delta^{2}
$$

Implementation:

```python
dW2 = A1.T @ delta2
```

Shape check:

```text
(5,8) @ (8,3) = (5,3)
```

Matches $W^2$ shape.

---

### Bias Gradient

$$
\frac{\partial L}{\partial b^{2}} = \sum_{i=1}^{\text{batch}} \delta^{2}_i
$$

Implementation:

```python
db2 = np.sum(delta2, axis=0, keepdims=True)
```

---

### Linked numeric example (continuation from section 3 output gradient)

Use the same `A1` and `delta2` values already computed in section 3.

Numeric values (same chain):

```text
A1 = [[0.900, 0.944],
    [0.993, 0.998]]    # shape (2,2)

delta2 = [[-0.5781,  0.3435, 0.2346],
      [ 0.4293, -0.6582, 0.2289]]  # shape (2,3)
```

Weight gradient (matrix multiply):

```text
dW2 = A1.T @ delta2

# compute entries:
# row 0: [0.900*(-0.5781) + 0.993*(0.4293),
#         0.900*(0.3435)  + 0.993*(-0.6582),
#         0.900*(0.2346)  + 0.993*(0.2289)]
#       = [-0.0940, -0.3444, 0.4384]
# row 1: [0.944*(-0.5781) + 0.998*(0.4293),
#         0.944*(0.3435)  + 0.998*(-0.6582),
#         0.944*(0.2346)  + 0.998*(0.2289)]
#       = [-0.1173, -0.3326, 0.4499]

dW2 ≈ [[-0.0940, -0.3444, 0.4384],
      [-0.1173, -0.3326, 0.4499]]   # shape (2,3)
```

Bias gradient (sum over batch):

```text
db2 = sum_rows(delta2) ≈ [-0.1488, -0.3147, 0.4635]  # shape (1,3)
```

If you average over the batch (batch size = 2):

```text
dW2_mean = dW2 / 2  # element-wise division
db2_mean = db2 / 2
```


This demonstrates the same formulas (`dW = A^T @ delta`, `db = sum(delta)`) on the exact `delta2` produced in section 3 (no disconnected toy values).


## 6. Parameter Update (Gradient Descent)

$$
W = W - \eta \frac{\partial L}{\partial W}
$$

$$
b = b - \eta \frac{\partial L}{\partial b}
$$

Where $\eta$ is the learning rate.

---

### Numeric parameter-update example (same chain, sections 3–5)

Using the same `W2` from section 3 and `dW2_mean`/`db2_mean` from section 5, apply gradient descent.

Values from above:

```text
W2 = [[ 0.20, -0.10,  0.05],
    [ 0.30,  0.40, -0.20]]          # shape (2,3)

dW2_mean = dW2 / 2 ≈ [[-0.0470, -0.1722, 0.2192],
                [-0.0586, -0.1663, 0.2250]]  # shape (2,3)

db2_mean = db2 / 2 ≈ [[-0.0744, -0.1574, 0.2318]]  # shape (1,3)
```

Choose learning rate $\eta = 0.1$. Parameter update (element-wise):

```text
W2_new = W2 - eta * dW2_mean
    ≈ [[ 0.2047, -0.0828,  0.0281],
       [ 0.3059,  0.4166, -0.2225]]

b2_new = b2 - eta * db2_mean
    ≈ [[0.0174, -0.0043, 0.0068]]
```

NumPy sketch (re-usable):

```python
# given: W2, b2, dW2_mean, db2_mean, eta
W2_new = W2 - eta * dW2_mean
b2_new = b2 - eta * db2_mean
```

For completeness in this same iteration, you would also update `W1`/`b1` using `dW_mean`/`db_mean` from section 4.

## 7. Dimension Consistency Rule

All matrix operations must satisfy:

$$
(\text{batch},\,\text{input}) \cdot (\text{input},\,\text{output}) = (\text{batch},\,\text{output})
$$

Maintaining dimensional consistency is critical to avoid silent logical errors.

### Dimension examples (continuation)

Use the worked-example matrix sizes to illustrate common multiplications:

```text
# forward: X (2,3) @ W1 (3,2) = Z1 (2,2)
X  = (2,3)
W1 = (3,2)
X @ W1 -> (2,2)

# next-layer: A1 (2,2) @ W2 (2,3) = Z2 (2,3)
A1 = (2,2)
W2 = (2,3)
A1 @ W2 -> (2,3)

# gradient: X.T (3,2) @ delta_hidden (2,2) = dW1 (3,2)
X.T -> (3,2)
delta_hidden -> (2,2)
X.T @ delta_hidden -> (3,2)
```

Concrete numeric check (from worked example sizes):

```text
X = [[1,2,3],[4,5,6]]   # (2,3)
W1 = [[0.1,0.2],[0.3,0.4],[0.5,0.6]]  # (3,2)

# forward: (2,3) @ (3,2) -> (2,2) (we computed Z = [[2.21,2.82],[4.91,6.42]])

# gradient: X.T (3,2) @ delta_hidden (2,2) -> (3,2) (we computed dW ≈ [[0.01039,0.01066],...])
```

Keeping these arithmetic checks in mind prevents shape mismatches and helps debug silent broadcasting errors.


---

## 8. Project-Specific Math Addendum (`model/train.py`)

Bu bölüm, mevcut kodda aktif kullanılan matematikleri toplar. Sigmoid bölümleri korunmuş, bu kısım sadece **ek** olarak yazılmıştır.

### 8.1 ReLU and ReLU Derivative

Hidden layers use ReLU:

$$
	ext{ReLU}(z)=\max(0,z)
$$

Derivative used in backprop:

$$
	ext{ReLU}'(z)=
\begin{cases}
1, & z>0 \\
0, & z\le 0
\end{cases}
$$

Mini example:

```text
z      = [-2.0, -0.3, 0.0, 1.2, 3.1]
ReLU   = [ 0.0,  0.0, 0.0, 1.2, 3.1]
ReLU'  = [ 0,    0,   0,   1,   1  ]
```

### 8.2 Stable Softmax Used in Code

Softmax per sample:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Numerically stable version (used in code):

$$
\hat{y}_i = \frac{e^{z_i - z_{max}}}{\sum_j e^{z_j - z_{max}}}
$$

Mini example:

```text
z = [5.0, 2.0, -1.0]
z_max = 5.0
z_shift = [0.0, -3.0, -6.0]
exp(z_shift) ≈ [1.0000, 0.0498, 0.0025]
sum ≈ 1.0523
softmax ≈ [0.9503, 0.0473, 0.0024]
```

### 8.3 Categorical Cross-Entropy (Batch Mean)

The training loss in `compute_loss()`:

$$
L = -\frac{1}{m}\sum_{k=1}^{m}\sum_{c=1}^{C} y_{k,c}\log(\hat{y}_{k,c})
$$

with clipping:

$$
\hat{y}_{k,c} \leftarrow \text{clip}(\hat{y}_{k,c}, \epsilon, 1-\epsilon)
$$

to avoid $\log(0)$.

### 8.4 Softmax + Cross-Entropy Gradient Simplification

For output layer logits:

$$
\delta^L = \hat{Y} - Y
$$

This is exactly implemented as:

```python
eh[-1] = activations[-1] - y
```

### 8.5 Hidden-Layer Error Propagation in This Project

For hidden layer $l$:

$$
\delta^l = (\delta^{l+1}(W^{l+1})^T)\odot \text{ReLU}'(Z^l)
$$

matching code:

```python
eh[i] = (eh[i + 1] @ W[i + 1].T) * derivative_relu(z_scores[i])
```

### 8.6 Mini-Batch Gradients and Update Rule

For each layer:

$$
\nabla W^l = \frac{(A^{l-1})^T\delta^l}{m_b},
\qquad
\nabla b^l = \frac{\sum\delta^l}{m_b}
$$

Update:

$$
W^l \leftarrow W^l - \eta\nabla W^l,
\qquad
b^l \leftarrow b^l - \eta\nabla b^l
$$

where $m_b$ is batch size and $\eta$ is learning rate.

### 8.7 He Uniform Initialization (Used)

For each layer with `fan_in`:

$$
W \sim U\left(-\sqrt{\frac{6}{fan_{in}}}, +\sqrt{\frac{6}{fan_{in}}}\right),
\qquad b=0
$$

This matches ReLU-based hidden layers and stabilizes initial signal flow.



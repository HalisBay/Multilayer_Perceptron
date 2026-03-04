# Forward Propagation (İleri Yayılım)

## Neden Önemli?

Forward propagation, bir sinir ağının **tahmin üretme** adımıdır. Modelin ne öğrendiğini görmek için önce forward çalışır, sonra loss hesaplanır.

- Girdi verisini katman katman işler
- Her katmanda lineer + nonlineer dönüşüm yapar
- Çıkışta sınıf olasılıkları veya tahmin değeri üretir

---

## Temel Denklem

Bir katman için:

$$
Z = XW + b
$$

$$
A = f(Z)
$$

Burada:

- $X$: giriş matrisi
- $W$: ağırlık matrisi
- $b$: bias
- $f$: aktivasyon fonksiyonu (bu projede hidden için ReLU)

---

## Bu Projede Forward Akışı (`model/train.py`)

1. `activations = [X]` ile giriş saklanır
2. Hidden katmanlarda:
   - `Zh = H @ W[i] + b[i]`
   - `H = relu(Zh)`
3. Çıkış katmanında:
   - `Zo = H @ W[-1] + b[-1]`
   - `yHat = softmax(Zo)`
4. `activations` ve `z_scores` geri döndürülür (backprop için)

---

## Boyut (Shape) Mantığı

Örnek:

```text
X       : (batch, input_dim)
W1      : (input_dim, h1)
b1      : (1, h1)
Z1, A1  : (batch, h1)
W2      : (h1, output_dim)
Z2, yHat: (batch, output_dim)
```

Temel kural:

$$
(m, n) \cdot (n, k) = (m, k)
$$

---

## Küçük Sayısal Örnek

```text
X  = [[1, 2]]
W1 = [[0.1, 0.2],
      [0.3, 0.4]]
b1 = [[0.0, 0.1]]

Z1 = X @ W1 + b1
   = [[0.7, 1.1]]

A1 = ReLU(Z1)
   = [[0.7, 1.1]]
```

Adım adım açılım:

```text
Z1[0,0] = 1*0.1 + 2*0.3 + 0.0 = 0.1 + 0.6 = 0.7
Z1[0,1] = 1*0.2 + 2*0.4 + 0.1 = 0.2 + 0.8 + 0.1 = 1.1
```

ReLU uygulaması:

```text
ReLU(0.7) = 0.7
ReLU(1.1) = 1.1
```

Çıkış katmanını da ekleyelim (2 sınıf):

```text
W2 = [[0.5, -0.3],
      [0.2,  0.4]]
b2 = [[0.0, 0.0]]

Zo = A1 @ W2 + b2
   = [[0.7, 1.1]] @ [[0.5, -0.3],
                     [0.2,  0.4]]
   = [[0.7*0.5 + 1.1*0.2,
       0.7*(-0.3) + 1.1*0.4]]
   = [[0.57, 0.23]]
```

Softmax adımı:

```text
exp(0.57) ≈ 1.768
exp(0.23) ≈ 1.259
toplam     ≈ 3.027

yHat = [1.768/3.027, 1.259/3.027]
     ≈ [0.584, 0.416]
```

Yani model bu örnekte ilk sınıfa yaklaşık %58.4 olasılık veriyor.

---


## Kaynaklar

- https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html#larger-network

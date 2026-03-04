# Backpropagation (Geri Yayılım)

## Neden Önemli?

Backpropagation, modelin hatadan öğrenmesini sağlar.

- Tahmin ile gerçek etiket farkını (loss) ölçer
- Zincir kuralı ile her ağırlığın hataya etkisini hesaplar
- Gradient descent ile ağırlıkları günceller

Özet: **Öğrenme mekanizmasının kalbi backpropagation’dır.**

---

## Temel Mantık

1. Forward propagation ile tahmin al
2. Loss hesapla
3. Çıkış katmanından başlayarak hata terimlerini geriye yay
4. Her katmanın gradyanını hesapla
5. Ağırlıkları güncelle

---

## Zincir Kuralı (Chain Rule)

Backpropagation tamamen chain rule uygulamasıdır.

$$
\frac{\partial L}{\partial W^l} = \frac{\partial L}{\partial Z^l}\frac{\partial Z^l}{\partial W^l}
$$

Katman bazlı standart form:

$$
\delta^L = \hat{Y}-Y
$$

$$
\delta^l = (\delta^{l+1}(W^{l+1})^T)\odot f'(Z^l)
$$

$$
\nabla W^l = (A^{l-1})^T\delta^l, \quad \nabla b^l = \sum \delta^l
$$

---

## Bu Projedeki Backprop Akışı (`model/train.py`)

- Çıkış hata terimi:

```python
eh[-1] = activations[-1] - y
```

- Hidden hata terimi:

```python
eh[i] = (eh[i + 1] @ W[i + 1].T) * derivative_relu(z_scores[i])
```

- Gradyanlar:

```python
dwh[i] = activations[i].T @ eh[i] / batch
dbh[i] = np.sum(eh[i], axis=0, keepdims=True) / batch
```

- Güncelleme:

```python
W[i] -= dwh[i] * learning_rate
b[i] -= dbh[i] * learning_rate
```

---

## Kaynaklar

- https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
- https://medium.com/@omerkaanvural/backpropagation-nedir-f2cd01d9ec1d

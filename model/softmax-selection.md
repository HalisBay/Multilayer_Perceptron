# Softmax (Çok Sınıflı Çıkış Aktivasyonu)

## Neden Önemli?

Softmax, modelin ürettiği ham skorları (logit) **olasılık dağılımına** dönüştürür.

- Çıktılar 0 ile 1 arasına gelir
- Tüm sınıf olasılıkları toplamı 1 olur
- Çok sınıflı sınıflandırmada standart çıkış aktivasyonudur

---

## Softmax Formülü

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### Neden üs alma var?

- Tüm değerleri pozitif yapar
- Büyük logitleri daha baskın hale getirir
- Olasılık yorumunu mümkün kılar

---

## Sayısal Kararlılık (Very Important)

Doğrudan `exp(z)` overflow yapabilir. Bu nedenle stabil versiyon:

$$
\text{softmax}(z_i)=\frac{e^{z_i-z_{max}}}{\sum_j e^{z_j-z_{max}}}
$$

Bu projedeki kod da bunu yapıyor:

```python
shiftZ = Z - np.max(Z, axis=1, keepdims=True)
expZ = np.exp(shiftZ)
yHat = expZ / np.sum(expZ, axis=1, keepdims=True)
```

---

## Cross-Entropy ile Birlikte Kullanım

Çok sınıflı problemler için yaygın ikili:

- Çıkış aktivasyonu: Softmax
- Loss: Categorical Cross-Entropy

Önemli sadeleşme:

$$
\frac{\partial L}{\partial Z} = \hat{Y} - Y
$$

Bu yüzden backprop daha sade ve hızlı olur.

---

## Bu Projedeki Kullanımı

- Hidden katmanlar: ReLU
- Son katman: Softmax
- Etiketler: One-hot encoding
- Tahmin sınıfı: `argmax(yHat)`

---

## Softmax vs Sigmoid

| Özellik | Softmax | Sigmoid |
|---|---|---|
| Kullanım | Çok sınıflı (mutually exclusive) | İkili / çok etiketli |
| Çıktı toplamı | 1 | 1 olmak zorunda değil |
| Sınıf rekabeti | Var | Yok |

---

## Kaynaklar

- https://www.ultralytics.com/tr/glossary/softmax
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html
- https://medium.com/deeper-deep-learning-tr/softmax-bir-aktivasyon-fonksiyonu-da8382d8a281

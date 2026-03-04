# ReLU (Rectified Linear Unit)

## Neden Önemli?

ReLU, modern ağlarda hidden katmanlar için en yaygın aktivasyonlardan biridir.

- Hesaplaması çok hızlıdır
- Derin ağlarda öğrenmeyi kolaylaştırır
- Sigmoid/tanh’a göre vanishing gradient riskini azaltır

---

## ReLU Formülü

$$
\text{ReLU}(x)=\max(0,x)
$$

Türev:

$$
\text{ReLU}'(x)=
\begin{cases}
1, & x>0 \\
0, & x\le 0
\end{cases}
$$

Bu projede:

```python
def relu(Z):
    return np.maximum(0, Z)

def derivative_relu(Z):
    return (Z > 0).astype(float)
```

---

## ReLU Avantajları

- **Simplicity:** Uygulaması çok basit
- **Efficiency:** Hesaplama maliyeti düşük
- **Gradient Flow:** Pozitif bölgede türev sabit 1

---

## ReLU Sorunları

### Dying ReLU

Eğer nöron sürekli negatif bölgede kalırsa çıktı hep 0 olur:

- İleri yayılımda katkısı yok
- Geri yayılımda gradient 0
- Nöron öğrenmeyi bırakır

Çözüm alternatifleri:

- Leaky ReLU
- PReLU
- ELU
- Uygun initialization (He)

---

## Bu Projede Neden ReLU + He?
WDBC veri kümesi 30 özellik içerir ve genelde lineer olarak ayrılabilir değildir; bu nedenle ağımıza doğrusal olmayanlık (non-linearity) eklemek gerekir. Sigmoid veya tanh gibi doygunlaşan aktivasyonlar derin katmanlarda küçük gradyanlara yol açarak öğrenmeyi yavaşlatabilir. ReLU ise pozitif bölgede sabit türevi sayesinde daha hızlı ve daha stabil öğrenme sağlar ve pratikte genelde daha yüksek doğruluk verir.

Bu sebeple gizli katmanlarda ReLU kullandım ve ReLU ile uyumlu olduğu için He (Uniform/Normal) başlangıç yöntemini tercih ettim. He başlangıcı, ReLU aktivasyonlu katmanlarda ağırlıkların uygun ölçeğe sahip olmasını sağlayarak gradyan akışını korumaya yardımcı olur.

Ayrıca ön-işleme ve feature selection uygulayarak özellik sayısını 30'dan 6'ya düşürdüm; bu, modelin hesaplama verimliliğini artırır ve aşırı uyumu (overfitting) azaltmaya katkı sağlar.
---

## Sayısal Mini Örnek

```text
Z = [-2.0, -0.1, 0.0, 1.5, 3.0]
ReLU(Z) = [0.0, 0.0, 0.0, 1.5, 3.0]
ReLU'(Z)= [0,   0,   0,   1,   1]
```

---

## Kaynak

- https://www.datacamp.com/blog/rectified-linear-unit-relu

# Weight Initialization Techniques (Ağırlık Başlatma Teknikleri)

## Neden Önemli?

Ağırlık ilkleştirmesi, sinir ağlarının öğrenme hızı ve doğruluğunu doğrudan etkiler. Uygun olmayan başlatma, ciddi sorunlara yol açabilir:

- **Vanishing Gradient Problem:** Gradientler çok küçülür, training yavaşlar veya tamamen durur
- **Exploding Gradient Problem:** Gradientler kontrol dışı büyür, instability ve model çökmesi yaşanır
- **Slow Convergence:** Ağ optimal çözüme ulaşmak için aşırı uzun süre gerektirir

---

## Gradient Problemleri Detaylı Açıklama

### Vanishing Gradient (Kaybolan Gradyan)

Backpropagation sırasında gradientler katmanlar arasında çarpılarak geriye aktarılır. Eğer bu gradientler çok küçükse (0 ile 1 arasında), çarpım işlemi onları daha da küçültür:

```
gradient_katman_1 = gradient_katman_n * 0.5 * 0.5 * 0.5 * ...
```

Derinliklere gidildikçe gradient neredeyse sıfıra yaklaşır:
- **Ön katmanlar neredeyse hiç güncellenmez**
- **Training durur, model platoya ulaşır**
- **Derinliklerin ön layers'ı asla öğrenemez**

**Hangi aktivasyon fonksiyonları etkilenir?**
- **Sigmoid:** Türevi max 0.25, vanishing gradient riski çok yüksek
- **Tanh:** Sigmoid'den biraz daha iyi ama yine risk var
- **ReLU:** Non-saturating, vanishing gradient problemi az

### Exploding Gradient (Patlayan Gradyan)

Tam tersi durumda, gradientler çok büyükse çarpımlar onları daha da büyütür:

```
gradient_katman_1 = gradient_katman_n * 2.5 * 2.5 * 2.5 * ...
```

Sonuç:
- **Gradientler sonsuzluğa doğru gider**
- **Ağırlıklar kontrolsüz şekilde güncellenir**
- **Ağ diverge olur, NaN/Inf değerlere ulaşır**
- **Tüm eğitim şartsızır (crash)**

---

## 1. Zero Initialization (Sıfır Başlatma)

**Yöntem:** Tüm ağırlıklar sıfırdan başlatılır.

```python
W = 0, b = 0
```

**Sorun:**
- Tüm nöronlar **simetrik** çalışır → aynı çıktıyı üretir
- Her nöron kendini **birbirinden farklılaştıramaz**
- Çıkış: Ağ tam sıfır gradyan problemi ile karşılaşır
- **RESULT: Model hiçbir şey öğrenemez!** ❌

**Ne Zaman Kullanılır?** Hiçbir zaman gerçek projekte → Sadece referans/karşılaştırma için

---

## 2. Random Initialization (Rastgele Başlatma)

**Yöntem:** Ağırlıklar rastgele bir dağılımdan seçilir. Nöronların simetrisi kırılır.

### 2.1 Random Normal (Normal Dağılım)
```
W ~ N(0, 1)  // Standart Normal Dağılım
```

### 2.2 Random Uniform (Düzgün Dağılım)
```
W ~ U(-a, +a)  // [-a, +a] aralığında eşit olasılık
```

**Sorun Nedir?**
- **Çok küçük değerler:** Sigmoid/Tanh ile → Vanishing gradient → Training çok yavaş ⚠️
- **Çok büyük değerler:** → Exploding gradient → Instability ⚠️
- Katmanlar arasında consistency yok → Activation değerleri dengesiz dağılır

**Ne Zaman Kullanılır?** Genellikle kullanılmaz; gerçek projelerde Xavier/He tercih edilir

---

## 3. Xavier / Glorot Initialization

**Amaç:** Activation'ların varyansını tüm katmanlar arasında **sabit tutmak**

### 3.1 Xavier Uniform
```
Limit = √(6 / (fan_in + fan_out))
W ~ U(-Limit, +Limit)
```

### 3.2 Xavier Normal
```
σ = √(2 / (fan_in + fan_out))
W ~ N(0, σ²)
```

**Formülü Neden Bu?**
- **Fan_in:** Giriş nöronlarının sayısı
- **Fan_out:** Çıkış nöronlarının sayısı
- Büyük katmanlar → Küçük ağırlıklar
- Küçük katmanlar → Büyük ağırlıklar
- **Sonuç:** Dengeli activation dağılması sağlanır

**Uygun Aktivasyon:** **Sigmoid, Tanh** ✅

**Avantaj:**
- Sigmoid/Tanh için vanishing gradient riskini minimize eder
- Katmanlar arasında dengeli gradient akışı
- Konvergens hızıdır

**Dezavantaj:**
- ReLU ile iyi çalışmaz (ReLU'un doğası göz önüne alınmamış)
- ReLU yapılarında dead neurons artabilir

---

## 4. He Initialization

**Amaç:** ReLU aktivasyonunun **doğası** göz önüne alarak ağırlık başlatma

### 4.1 He Uniform
```
Limit = √(6 / fan_in)
W ~ U(-Limit, +Limit)
```

### 4.2 He Normal
```
σ = √(2 / fan_in)
W ~ N(0, σ²)
```

**Neden Sadece fan_in?**
- ReLU, negatif değerleri **sıfıra basıyor** → Non-saturating
- Xavier'in `fan_out`'u göz önüne alması ReLU için gereksiz
- **Çıkış katmanının boyutu, ağırlık başlatmasına teknik etki etmez**

**Uygun Aktivasyon:** **ReLU, LeakyReLU, ELU, GELU** ✅

**Avantaj:**
- ReLU ile optimal convergence sağlar
- Exploding gradient riskini minimize eder
- Dead neuron problemi çok az

**ReLU'da Ölü Nöron (Dead Neuron) Problemi:**
```
ReLU: f(x) = max(0, x)

Eğer w ve b öyle initialize edilirse ki z = w*x + b her zaman negatif:
  → f(z) = 0 (nöron "ölü" kalır)
  → Gradient = 0 (nöron hiçbir şey öğrenmez)
  → tüm epochlar boyunca aynı kalır
```

**He initialization bunu önler çünkü ağırlıkları uygun aralıkta başlatır** ✅

---

## Karşılaştırma Tablosu

| Yöntem | Sigmoid | Tanh | ReLU | Açıklama |
|--------|---------|------|------|----------|
| **Zero** | ❌ | ❌ | ❌ | Simetri problemi, model öğrenemez |
| **Random** | ⚠️ | ⚠️ | ⚠️ | Unstable, gradient sorunları |
| **Xavier** | ✅✅ | ✅✅ | ⚠️ | Sigmoid/Tanh için optimal |
| **He** | ⚠️ | ⚠️ | ✅✅ | ReLU ve varyantları için optimal |

---


## Kaynaklar
- https://medium.com/@piyushkashyap045/understanding-weight-initialization-techniques-in-neural-networks-582e80a1e839
- https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/

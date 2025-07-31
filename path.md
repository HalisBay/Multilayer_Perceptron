MLP Projesi (Wisconsin Breast Cancer Dataset)
│
├── 1. Veriyi indir ve Keşfet ✅
│   ├── 1.1 Veri setini indir ✅
│   │   └── UCI üzerinden veri çek veya hazır CSV’yi kullan
│   ├── 1.2 Veri setini incele ✅
│   │   ├── Satır/sütun sayısı, eksik değer kontrolü (NaN)
│   │   ├── Sınıf dağılımı (M vs B sayısı, oranlar)
│   │   └── Özelliklerin açıklamaları (istatistiksel analiz, describe())
│   ├── 1.3 Etiketleri dönüştür ✅
│   │   └── 'M' → 1 (kötü huylu), 'B' → 0 (iyi huylu)
│   └── 1.4 Özellikleri normalize et ✅
│       ├── MinMaxScaler (0-1 aralığı) veya StandartScaler (z-score)
│       └── Tüm özellikler aynı ölçekte olmalı (gradient descent için önemli)
│
├── 2. Proje Altyapısı ve Yapılandırma
│   ├── 2.1 Klasör yapısını oluştur ✅
│   │   ├── /data, /model, /plots, /utils gibi dizinler
│   └── 2.2 Train/Validation ayrımı ✅
│       ├── %80 train / %20 validation
│       └── Ayrımı tekrarlanabilir yapmak için `random_state` sabitlenir
│
├── 3. Sinir Ağı Mimarisi (MLP)
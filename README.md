# Multilayer Perceptron

## Introduce
This project aims to implement a multilayer perceptron to predict whether a cancer diagnosis is malignant or benign, using a dataset of breast cancer cases from Wisconsin.

## Dataset

**Dataset: Wisconsin Diagnostic Breast Cancer [(WDBC)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)** 
This dataset contains 569 samples for breast cancer diagnosis.

- There are 32 columns: 1 ID, 1 diagnosis label, and 30 numerical features.
- Diagnosis labels:
  - `'M'` — Malignant (cancerous)
  - `'B'` — Benign (non-cancerous)
- Features are calculated from cell nuclei images obtained via fine needle aspiration biopsy of breast masses, including measurements such as radius, texture, area, perimeter, smoothness, compactness, and symmetry.
- There are no missing values in the dataset.
- Class distribution: 357 benign, 212 malignant samples.
- The dataset is split as 80% for training and 20% for validation.


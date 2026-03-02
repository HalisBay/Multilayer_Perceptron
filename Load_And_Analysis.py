from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataPipeline:
    def __init__(self, raw_path="data/raw/wdbc.csv", split_data_path="data/split"):
        self.raw_path = raw_path
        self.split_data_path = split_data_path
        self.train_path = os.path.join(split_data_path, "train.csv")
        self.valid_path = os.path.join(split_data_path, "valid.csv")

    def download_and_split(self):
        # Veri dosyası yoksa indir ve kaydet
        if not os.path.exists(self.raw_path):
            os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
            dataset = fetch_ucirepo(id=17)
            df = dataset.data.features.copy()
            # Target sütununu (diagnosis) sayısal olarak kodla: M=1, B=0
            labels = dataset.data.targets.replace({"M": 1, "B": 0})
            df["diagnosis"] = labels
            df.to_csv(self.raw_path, index=False)
            print(f"Dataset downloaded and saved to {self.raw_path}")
        else:
            pass

        # Veriyi yükle x ve y etiketlerine göre ayır
        df = pd.read_csv(self.raw_path)
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Save processed data
        os.makedirs(self.split_data_path, exist_ok=True)
        train_df = pd.concat([X_train, y_train], axis=1)
        train_df.to_csv(self.train_path, index=False)
        valid_df = pd.concat([X_valid, y_valid], axis=1)
        valid_df.to_csv(self.valid_path, index=False)

    def analyze(self):
        train_df = pd.read_csv(self.train_path)
        X_train = train_df.drop(columns=["diagnosis"])
        y_train = train_df["diagnosis"]

        print("\nTrain features shape:", X_train.shape)
        print("Train targets shape:", y_train.shape)
        print("First 5 train targets:\n", X_train.head())
        # print("First 5 train targets:\n", y_train.head())

        print("Missing values in train features:", X_train.isnull().sum().sum())
        print("Missing values in train targets:", y_train.isnull().sum().sum())
        # Sınıf dağılımı ve oranları
        print("\nTrain class distribution:")
        print(y_train.value_counts())
        print("Class ratios:")
        print(y_train.value_counts(normalize=True))

        # Özelliklerin istatistiksel özetini yazdır
        print("\nTrain feature statistics:")
        print(X_train.describe())


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.download_and_split()
    pipeline.analyze()

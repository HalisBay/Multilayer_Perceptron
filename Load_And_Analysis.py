from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class DataPipeline:
    def __init__(self, raw_path="data/raw/wdbc.csv", processed_dir="data/processed"):
        self.raw_path = raw_path
        self.processed_dir = processed_dir
        self.train_path = os.path.join(processed_dir, "train.csv")
        self.valid_path = os.path.join(processed_dir, "valid.csv")

    def download_and_preprocess(self):
        """
        Downloads the UCI Breast Cancer Wisconsin (Diagnostic) dataset if not already present,
        preprocesses the data by scaling features and encoding labels, splits it into training
        and validation sets, and saves the processed data to CSV files.
        Steps performed:
        - Checks if the raw dataset exists locally; if not, downloads and saves it.
        - Loads the dataset and encodes the 'diagnosis' column ('M' as 1, 'B' as 0).
        - Scales feature values to the [0, 1] range using MinMaxScaler.
        - Splits the data into training and validation sets with stratification.
        - Saves the processed training and validation sets to CSV files.
        No parameters are required. The method relies on instance attributes for file paths.
        """
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
            print(f"Dataset already exists at {self.raw_path}")

        # Veriyi yükle x ve y etiketlerine göre ayır
        df = pd.read_csv(self.raw_path)
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]

        # Özellikleri 0- 1 aralığına ölçekle
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_scaled_df, y, test_size=0.2, stratify=y, random_state=42
        )

        # İşlenmiş verileri kaydet
        os.makedirs(self.processed_dir, exist_ok=True)
        X_train.assign(label=y_train.values).to_csv(self.train_path, index=False)
        X_valid.assign(label=y_valid.values).to_csv(self.valid_path, index=False)
        print(f"Processed train/valid data saved to {self.processed_dir}")

    def analyze(self):
        """
        Loads the training dataset from the specified path and performs exploratory data analysis.
        - Reads the training data CSV file.
        - Separates features and target labels.
        - Prints the shape of features and targets.
        - Displays the first 5 target labels.
        - Reports the number of missing values in features and targets.
        - Shows the class distribution and class ratios of the target labels.
        - Provides descriptive statistics for the feature columns.
        """
        # Eğitim verisini oku
        train_df = pd.read_csv(self.train_path)
        X_train = train_df.drop(columns=["label"])
        y_train = train_df["label"]

        print("\nTrain features shape:", X_train.shape)
        print("Train targets shape:", y_train.shape)
        print("First 5 train targets:\n", y_train.head())

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
    pipeline.download_and_preprocess()
    pipeline.analyze()

import pandas as pd
from sklearn.model_selection import train_test_split
import os

base_path = os.path.dirname(os.path.abspath(__file__))


class DataPipeline:
    def __init__(
        self,
        raw_path=base_path + "/data/raw/data.csv",
        split_data_path=base_path + "/data/split",
    ):
        self.raw_path = raw_path
        self.split_data_path = split_data_path
        self.train_path = os.path.join(split_data_path, "train.csv")
        self.valid_path = os.path.join(split_data_path, "valid.csv")

    def load_and_split(self):
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"File not found: {self.raw_path}")

        features = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points",
            "symmetry",
            "fractal_dimension",
        ]
        feature_columns = [f"{f}{i}" for i in (1, 2, 3) for f in features]
        target_columns = feature_columns + ["diagnosis"]
        data_columns = ["id", "diagnosis"] + feature_columns

        header_df = pd.read_csv(self.raw_path)
        if list(header_df.columns) == target_columns:
            df = header_df.copy()
        else:
            raw_df = pd.read_csv(self.raw_path, header=None)
            if raw_df.shape[1] == len(data_columns):
                raw_df.columns = data_columns
                raw_df = raw_df.drop(columns=["id"])
                df = raw_df[target_columns]
            elif raw_df.shape[1] == len(target_columns):
                raw_df.columns = target_columns
                df = raw_df.copy()
            else:
                raise ValueError("Unexpected column please check data")

        df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

        df.to_csv(self.raw_path, index=False)

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

        print("Train features shape:", X_train.shape)
        print("Train targets shape:", y_train.shape)
        print("First 5 train data :\n", X_train.head())
        # print("First 5 train targets:\n", y_train.head())

        print("Null count in train features:", X_train.isnull().sum().sum())
        print("Null count in train targets:", y_train.isnull().sum().sum())
        # Sınıf dağılımı ve oranları
        print("\nTrain class distribution:")
        print(y_train.value_counts())

        print(X_train.describe())

    def corr_with_target(self):
        train_df = pd.read_csv(self.train_path)
        X_train = train_df.drop(columns=["diagnosis"])
        y_train = train_df["diagnosis"]

        features = []
        for base_name in [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points",
            "symmetry",
            "fractal_dimension",
        ]:
            corr_target = X_train.corrwith(y_train).abs()
            candidates = [f"{base_name}1", f"{base_name}2", f"{base_name}3"]
            # idxmax selects the largest value
            selected_feature = corr_target[candidates].idxmax()
            features.append(selected_feature)

        print("Selected features:", features)

    def corr_with_selected_features(self):
        train_df = pd.read_csv(self.train_path)
        X_train = train_df[["radius3", "perimeter3", "area3"]]
        # X_train = train_df[[ "concavity1", "concave_points3", "compactness1"]]
        y_train = train_df["diagnosis"]

        corr_target = X_train.corrwith(y_train).abs()
        selected_feature = corr_target.idxmax()
        print("Selected feature:", selected_feature)


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.load_and_split()
    pipeline.analyze()
    pipeline.corr_with_target()
    pipeline.corr_with_selected_features()

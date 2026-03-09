import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt


def rain_cloud(df, features):
    
    tmp = df.copy()
    tmp["diagnosis"] = tmp["diagnosis"].astype(str)
    for f in features:
        plt.figure(figsize=(6, 2.8))
        pt.RainCloud(
            x="diagnosis",
            y=f,
            data=tmp,
            hue="diagnosis",
            palette="bright",
            bw=0.3,
            width_viol=0.7,
            orient="h",
        )
        plt.title(f)
        plt.tight_layout()
        plt.show()


def heatmap(feats):

    mask = np.triu(np.ones_like(feats.corr(), dtype=bool))

    plt.figure(figsize=(10, 6))

    sns.heatmap(feats.corr(), mask=mask, annot=True, square=True)
    plt.title("Correlation matrix ")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = base_path + "/data/split/train.csv"
    df = pd.read_csv(data_path)
    selected_features = [
        "texture3",
        "perimeter3",
        "smoothness3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]

    # feats = [
    #     "radius3",
    #     "texture3",
    #     "perimeter3",
    #     "area3",
    #     "smoothness3",
    #     "compactness1",
    #     "concavity1",
    #     "concave_points3",
    #     "symmetry3",
    #     "fractal_dimension3",
    # ]
    # concavity1, concave_points3, compactness1 = +0.85 Selected feature: concave_points3
    # radius3, perimeter3, area3 = +98 Selected feature: perimeter3

    heatmap(df[selected_features])
    rain_cloud(df, selected_features)

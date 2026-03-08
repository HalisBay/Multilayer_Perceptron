import ptitprince as pt
import pandas as pd
import matplotlib.pyplot as plt
import os

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

features = [c for c in df.columns if c in selected_features]

df["diagnosis"] = df["diagnosis"].astype(str)


for f in features:
    plt.figure(figsize=(6, 2.8))
    pt.RainCloud(
        x="diagnosis",
        y=f,
        data=df,
        hue="diagnosis",
        palette="bright",
        bw=0.3,
        width_viol=0.7,
        orient="h",
    )
    plt.title(f)
    plt.tight_layout()
    plt.show()

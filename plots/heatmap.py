import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

base = os.path.dirname(__file__)

data_path = os.path.join(base, "..", "data", "split", "train.csv")
data = pd.read_csv(data_path)


# feats = data[
#     [
#         "radius3",
#         "texture3",
#         "perimeter3",
#         "area3",
#         "smoothness3",
#         "compactness1",
#         "concavity1",
#         "concave_points3",
#         "symmetry3",
#         "fractal_dimension3",
#     ]
# ]

# concavity1, concave_points3, compactness1 = +0.85 Selected feature: concave_points3
# radius3, perimeter3, area3 = +98 Selected feature: perimeter3

feats = data[
    [
        "texture3",
        "perimeter3",
        "smoothness3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]
]

mask = np.triu(np.ones_like(feats.corr(), dtype=bool))

plt.figure(figsize=(10, 6))

sns.heatmap(feats.corr(), mask=mask, annot=True, square=True)
plt.title("Correlation matrix ")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
features = list(iris.feature_names)
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

for k in range(2, 7):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print("K =", k, "Silhouette Score =", score)

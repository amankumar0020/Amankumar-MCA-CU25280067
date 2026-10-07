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

kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
hier_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)

print("K-Means Silhouette:", silhouette_score(X_scaled, kmeans_labels))
print("Hierarchical Silhouette:", silhouette_score(X_scaled, hier_labels))

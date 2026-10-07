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

before_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
after_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca)

print("Silhouette Before PCA:", silhouette_score(X_scaled, before_labels))
print("Silhouette After PCA:", silhouette_score(X_pca, after_labels))

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

before = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
after = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
print("Before Scaling:", before[:10])
print("After Scaling:", after[:10])

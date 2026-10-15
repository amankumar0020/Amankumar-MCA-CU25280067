import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Sample/illustrative customer dataset. Replace with real data for real business use.
data = {
    "Age": [19,21,20,23,31,35,40,42,46,48,22,25,28,33,37,45,50,52,27,30,41,44,26,29,34,39,47,49,24,36],
    "Annual_Income": [15,16,17,18,40,42,45,46,48,50,18,20,22,38,41,47,52,55,23,35,44,49,21,32,39,43,51,54,19,40],
    "Spending_Score": [39,81,6,77,40,60,13,79,20,42,76,65,35,55,70,15,25,10,73,48,18,30,68,52,63,22,33,12,80,58],
}
df = pd.DataFrame(data)
df.to_csv("customer_segmentation_dataset.csv", index=False)

print("First five rows:")
print(df.head())
print("\\nMissing values:")
print(df.isnull().sum())

# Preprocessing
df = df.drop_duplicates().copy()
df = df.fillna(df.median(numeric_only=True))
features = ["Age", "Annual_Income", "Spending_Score"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find a suitable K using silhouette scores.
scores = {}
for k in range(2, min(7, len(df))):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
print("\\nSilhouette scores by K:", {k: round(v, 3) for k, v in scores.items()})
print("Selected number of clusters:", best_k)

# Fit K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# Compare with Hierarchical/Agglomerative clustering using the same cluster count.
hierarchical = AgglomerativeClustering(n_clusters=best_k)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X_scaled)

print("\\nK-Means silhouette score:",
      round(silhouette_score(X_scaled, df["KMeans_Cluster"]), 3))
print("Hierarchical silhouette score:",
      round(silhouette_score(X_scaled, df["Hierarchical_Cluster"]), 3))

# PCA reduces the 3 features to 2 dimensions for visualization.
pca = PCA(n_components=2, random_state=42)
points_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1],
                      c=df["KMeans_Cluster"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using K-Means")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=150)
plt.show()

print("\\nCluster averages (original feature units):")
print(df.groupby("KMeans_Cluster")[features].mean().round(2))

joblib.dump(
    {"model": kmeans, "scaler": scaler, "features": features, "pca": pca},
    "customer_segmentation_model.pkl"
)
print("\\nSaved model: customer_segmentation_model.pkl")

# Example prediction for a new customer
new_customer = pd.DataFrame([{
    "Age": 30,
    "Annual_Income": 40,
    "Spending_Score": 65
}])
new_scaled = scaler.transform(new_customer[features])
print("New customer cluster:", int(kmeans.predict(new_scaled)[0]))

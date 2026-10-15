import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Small illustrative news corpus. For real use, replace with a larger news dataset.
articles = [
    ("Sports", "The football team won the championship after a dramatic final match."),
    ("Sports", "The cricket team secured victory in the international tournament."),
    ("Sports", "A tennis player reached the final after winning three matches."),
    ("Sports", "The coach announced the squad for the upcoming football season."),
    ("Technology", "A technology company launched a new artificial intelligence model."),
    ("Technology", "Researchers developed a machine learning system for medical images."),
    ("Technology", "The new smartphone includes a faster processor and improved camera."),
    ("Technology", "Cybersecurity experts warned about a new software vulnerability."),
    ("Business", "The stock market rose as investors reacted to corporate earnings."),
    ("Business", "A major company reported increased revenue and quarterly profit."),
    ("Business", "The central bank discussed inflation, interest rates, and economic growth."),
    ("Business", "A startup raised funding to expand its financial services platform."),
    ("Health", "Doctors recommend regular exercise and a balanced healthy diet."),
    ("Health", "Researchers studied a new treatment in a clinical trial."),
    ("Health", "Hospitals are adopting digital systems to improve patient care."),
    ("Health", "Public health experts discussed vaccination and disease prevention."),
    ("Politics", "Parliament debated a bill during the national budget session."),
    ("Politics", "The election commission announced the dates for the upcoming election."),
    ("Politics", "Government ministers discussed a new public policy proposal."),
    ("Politics", "Leaders held talks on international relations and regional security."),
]
df = pd.DataFrame(articles, columns=["Category", "Article"])
df.to_csv("news_articles_dataset.csv", index=False)

print("First five articles:")
print(df.head())
print("\\nMissing values:", df.isnull().sum().to_dict())

# Clean and preprocess text.
df = df.dropna(subset=["Article"]).drop_duplicates(subset=["Article"]).copy()
df["Article"] = df["Article"].astype(str).str.strip()
df = df[df["Article"] != ""]

# TF-IDF turns text into numeric features.
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X = vectorizer.fit_transform(df["Article"])

# Compare K-Means and hierarchical clustering at a chosen k.
# For this small demo corpus, use 5 clusters; test alternatives with silhouette scores.
candidate_scores = {}
for k in range(2, min(8, X.shape[0])):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    candidate_scores[k] = silhouette_score(X, labels, metric="cosine")

best_k = max(candidate_scores, key=candidate_scores.get)
print("\\nSilhouette scores by K:", {k: round(v, 3) for k, v in candidate_scores.items()})
print("Selected number of clusters:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

# Agglomerative clustering needs dense data; okay for this small sample.
hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="average", metric="cosine")
df["Hierarchical_Cluster"] = hierarchical.fit_predict(X.toarray())

print("\\nK-Means silhouette score:",
      round(silhouette_score(X, df["KMeans_Cluster"], metric="cosine"), 3))
print("Hierarchical silhouette score:",
      round(silhouette_score(X, df["Hierarchical_Cluster"], metric="cosine"), 3))

# Display articles grouped by K-Means cluster.
print("\\nArticles grouped by K-Means cluster:")
for cluster_id in sorted(df["KMeans_Cluster"].unique()):
    print(f"\\nCluster {cluster_id}:")
    for article in df.loc[df["KMeans_Cluster"] == cluster_id, "Article"]:
        print("-", article)

# Visualize TF-IDF vectors in 2D using TruncatedSVD.
n_components = min(2, X.shape[1] - 1)
if n_components >= 2:
    reduced = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=df["KMeans_Cluster"])
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("News Article Clustering using TF-IDF and K-Means")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("news_article_clusters.png", dpi=150)
    plt.show()

# Save vectorizer and model so future articles can be transformed and clustered.
joblib.dump(
    {"vectorizer": vectorizer, "model": kmeans, "num_clusters": best_k},
    "news_clustering_model.pkl"
)
print("\\nSaved model: news_clustering_model.pkl")

new_articles = [
    "The football club signed a striker before the new season.",
    "Scientists released a new artificial intelligence research model."
]
new_vectors = vectorizer.transform(new_articles)
predicted_clusters = kmeans.predict(new_vectors)
print("\\nNew article predictions:")
for article, cluster in zip(new_articles, predicted_clusters):
    print(f"Cluster {cluster}: {article}")

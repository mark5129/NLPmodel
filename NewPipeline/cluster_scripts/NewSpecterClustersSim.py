import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Filbaner ===
cluster_path = "NewPipeline/clustering_outputs/Specter2_clustering.csv"
embedding_path = "NewPipeline/clustering_outputs/Specter2_embeddings.csv"

# === Last inn data ===
clusters = pd.read_csv(cluster_path)
embeddings = pd.read_csv(embedding_path)

# === Legg til index for å matche ===
clusters["index"] = clusters.index
embeddings["index"] = embeddings.index
merged = pd.merge(clusters, embeddings, on="index")

# === Lag unik ID for cluster per source ===
merged["cluster_id"] = merged["cluster"].astype(str) + "_" + merged["Source"]

# === Beregn cosine similarity for hver (cluster, source)-gruppe ===
results = []
for cluster_id in sorted(merged["cluster_id"].unique()):
    sub = merged[merged["cluster_id"] == cluster_id]
    if len(sub) < 2:
        continue
    vecs = sub.select_dtypes(include=[np.number]).drop(columns=["index", "x", "y"]).values
    cosine_matrix = cosine_similarity(vecs)
    upper = np.triu_indices_from(cosine_matrix, k=1)
    mean_sim = cosine_matrix[upper].mean()

    cluster_label, source_label = cluster_id.split("_")
    results.append({
        "Cluster": int(cluster_label),
        "Source": source_label,
        "Size": len(sub),
        "Intra-cluster Cosine Similarity": round(mean_sim, 4)
    })

# === Print resultater ===
df_results = pd.DataFrame(results)
print("\n=== COSINE SIMILARITY PER (SOURCE, CLUSTER) I SPECTER2 ===")
print(df_results.sort_values(by=["Source", "Cluster"]))

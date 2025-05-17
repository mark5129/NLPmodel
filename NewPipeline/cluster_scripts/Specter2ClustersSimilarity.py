import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === FILBANER ===
EMBED_PATH = "NewPipeline/clustering_outputs/Specter2_merged_embeddings.csv"
CLUSTER_PATH = "NewPipeline/clustering_outputs/Specter2_merged_clustering.csv"

# === Last inn data ===
embeddings = pd.read_csv(EMBED_PATH)
clustering = pd.read_csv(CLUSTER_PATH)

# === Legg til indeks og slå sammen
embeddings["index"] = embeddings.index
clustering["index"] = clustering.index
merged = pd.merge(clustering[["index", "cluster"]], embeddings, on="index")

# === Resultater per cluster
results = []

for cluster_id in sorted(merged["cluster"].unique()):
    cluster_df = merged[merged["cluster"] == cluster_id]
    if len(cluster_df) < 2:
        continue

    vecs = cluster_df.select_dtypes(include=[np.number]).drop(columns=["index"]).values
    cosine_matrix = cosine_similarity(vecs)
    upper = np.triu_indices_from(cosine_matrix, k=1)
    avg_sim = cosine_matrix[upper].mean()

    results.append({
        "Cluster": cluster_id,
        "Size": len(cluster_df),
        "Intra-cluster Cosine Similarity": round(avg_sim, 4)
    })

# === Print resultat
print("\n=== COSINE SIMILARITY INNENFOR HVER CLUSTER (Specter2) ===\n")
for r in results:
    print(f"Cluster {r['Cluster']} | Size: {r['Size']} | Cosine similarity: {r['Intra-cluster Cosine Similarity']}")

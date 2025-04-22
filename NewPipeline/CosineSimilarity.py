import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Last inn data ===
embeddings = pd.read_csv("NewPipeline/clustering_outputs/Specter2_merged_embeddings.csv")
clustering = pd.read_csv("NewPipeline/clustering_outputs/Specter2_merged_clustering.csv")

# === Kombiner data ===
df = clustering.copy()
df["cluster_id"] = df["cluster"]  # kun cluster-nummer, ikke kilde
df["index"] = df.index
embeddings["index"] = embeddings.index
merged = pd.merge(df, embeddings, on="index")

# === Beregn baseline cosine similarity for alle tekster ===
embedding_vectors = merged.select_dtypes(include=[np.number]).values
cosine_all = cosine_similarity(embedding_vectors)
upper = np.triu_indices_from(cosine_all, k=1)
overall_mean = cosine_all[upper].mean()

# === Beregn cosine similarity for hver cluster (uavhengig av Source) ===
results = []
for cluster_id in merged["cluster_id"].unique():
    sub = merged[merged["cluster_id"] == cluster_id]
    if len(sub) < 2:
        continue
    vecs = sub.select_dtypes(include=[np.number]).values
    cosines = cosine_similarity(vecs)
    upper = np.triu_indices_from(cosines, k=1)
    mean_sim = cosines[upper].mean()
    results.append({
        "Cluster": cluster_id,
        "Size": len(sub),
        "Intra-cluster Cosine Similarity": round(mean_sim, 4),
        "Baseline (All Texts)": round(overall_mean, 4)
    })

# === Lag dataframe og print ===
results_df = pd.DataFrame(results)
print(results_df.sort_values("Intra-cluster Cosine Similarity", ascending=False))

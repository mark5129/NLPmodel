from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def recommend_k(X, k_range=range(2, 16), min_k=6):
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        inertias.append(kmeans.inertia_)

    k_range = list(k_range)

    # Normaliser scorene for sammenligning
    sil_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
    ch_norm = (ch_scores - np.min(ch_scores)) / (np.max(ch_scores) - np.min(ch_scores))
    db_norm = 1 - (db_scores - np.min(db_scores)) / (np.max(db_scores) - np.min(db_scores))  # lavere = bedre
    inertia_diff = np.gradient(inertias)
    inertia_norm = 1 - (inertia_diff - np.min(inertia_diff)) / (np.max(inertia_diff) - np.min(inertia_diff))

    # Summer og finn k med høyest total score (vektet sum)
    total_score = sil_norm + ch_norm + db_norm + inertia_norm
    for i, k in enumerate(k_range):
        if k >= min_k:
            print(f"k={k}: Score={total_score[i]:.3f} (Sil={silhouette_scores[i]:.3f}, CH={ch_scores[i]:.1f}, DB={db_scores[i]:.2f})")

    # Returner beste k ≥ min_k
    best_k = k_range[np.argmax([s if k >= min_k else -1 for k, s in zip(k_range, total_score)])]
    print(f"\n✅ Anbefalt k: {best_k}")
    return best_k
import pandas as pd

X = pd.read_csv("modelling/outputs/Specter2Actually/2572276933_merged_embeddings_Specter2Actually_embeddings.csv").values
recommend_k(X)

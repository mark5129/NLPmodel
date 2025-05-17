import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def find_optimal_clusters(embedding_path, k_range=range(2, 16)):
    df = pd.read_csv(embedding_path)
    X = df.values

    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(sil)
        calinski_scores.append(ch)
        davies_scores.append(db)
        inertias.append(inertia)

        print(f"k={k}: Silhouette={sil:.4f}, Calinski-Harabasz={ch:.2f}, Davies-Bouldin={db:.4f}, Inertia={inertia:.2f}")

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(k_range, silhouette_scores, marker='o')
    axs[0, 0].set_title("Silhouette Score")
    axs[0, 0].set_xlabel("k")
    axs[0, 0].set_ylabel("Score")

    axs[0, 1].plot(k_range, calinski_scores, marker='o', color='green')
    axs[0, 1].set_title("Calinski-Harabasz Score")
    axs[0, 1].set_xlabel("k")
    axs[0, 1].set_ylabel("Score")

    axs[1, 0].plot(k_range, davies_scores, marker='o', color='red')
    axs[1, 0].set_title("Davies-Bouldin Score")
    axs[1, 0].set_xlabel("k")
    axs[1, 0].set_ylabel("Score (lower is better)")

    axs[1, 1].plot(k_range, inertias, marker='o', color='purple')
    axs[1, 1].set_title("Elbow Method (Inertia)")
    axs[1, 1].set_xlabel("k")
    axs[1, 1].set_ylabel("Inertia")

    plt.tight_layout()
    plt.show()

# Eksempel på bruk:
find_optimal_clusters("modelling/outputs/Specter2Actually/4552557450_merged_embeddings_Specter2Actually_embeddings.csv")
#Specter2=7
#XLM_Roberta=6


import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def evaluate_silhouette_score(embedding_path, label_path):
    # Load embeddings
    embeddings = pd.read_csv(embedding_path)
    embeddings_array = embeddings.values  # Convert to numpy array

    # Load cluster labels
    labels_df = pd.read_csv(label_path)
    if 'cluster' in labels_df.columns:
        labels = labels_df['cluster'].values
    elif 'label' in labels_df.columns:
        labels = labels_df['label'].values
    else:
        raise ValueError("No 'cluster' or 'label' column found in KMeans file")

    # Compute silhouette score
    score = silhouette_score(embeddings_array, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score


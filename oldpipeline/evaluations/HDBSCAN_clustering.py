import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import hdbscan
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clustering_with_umap_hdbscan(df_file, embeddings, run_id, doc, model):
    """
    Run clustering using UMAP (for clustering) followed by HDBSCAN, then UMAP (for visualization).

    Parameters:
    - df_file: DataFrame with metadata like 'Source'.
    - embeddings: High-dimensional embeddings (array-like or DataFrame).
    - run_id: Unique identifier for this run.
    - doc: Document descriptor (e.g., dataset name).
    - model: Name of the embedding model used.
    """

    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # UMAP for clustering (10D)
    umap_cluster = UMAP(n_neighbors=30, min_dist=0.0, n_components=10, metric='cosine', random_state=42)
    cluster_space = umap_cluster.fit_transform(embeddings_scaled)

    # HDBSCAN clustering
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1, metric='euclidean')
    cluster_labels = hdbscan_model.fit_predict(cluster_space)

    # UMAP for 2D visualization
    umap_vis = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, metric='cosine', random_state=42)
    vis_2d = umap_vis.fit_transform(embeddings_scaled)

    # Prepare result DataFrame
    if 'Source' in df_file.columns:
        result_df = df_file[['Source']].copy()
    else:
        result_df = pd.DataFrame({'Source': [doc] * len(df_file)})

    result_df['cluster'] = cluster_labels
    result_df['x'] = vis_2d[:, 0]
    result_df['y'] = vis_2d[:, 1]

    # Filter out noise points (cluster = -1)
    filtered_df = result_df[result_df['cluster'] != -1]
    df_file = df_file[result_df['cluster'] != -1]
    

    # Save results
    output_dir = 'evaluations/outputs/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_clusters_HDBSCAN.csv')
    filtered_df.to_csv(output_file, index=False)
    print(f"Clustering completed. Results saved to '{os.path.basename(output_file)}'.")

    return filtered_df, df_file

import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import hdbscan
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clustering_with_umap_hdbscan(df_file, embeddings, run_id, doc, model):
    """
    Run clustering using UMAP dimensionality reduction followed by HDBSCAN.

    Parameters:
    - df_file: DataFrame containing metadata like 'Source' (for merged) or not (for individual).
    - embeddings: Embeddings (array-like or DataFrame) used for clustering.
    - run_id: Identifier for the run (e.g., 'manualrun').
    - doc: Description of the document type or source name for individual files.
    - model: Model name (e.g., 'Specter2').

    Steps:
    1. Standardize the embeddings.
    2. Reduce dimensions using UMAP.
    3. Cluster using HDBSCAN.
    4. Save clustering results and plots.
    """

    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Reduce dimensionality
    n_components = 2  # Set the desired dimensionality
    umap_model = UMAP(n_neighbors=30, min_dist=0.1, n_components=n_components, random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings_scaled)

    # Cluster using HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)

    # Prepare result DataFrame
    if 'Source' in df_file.columns:
        result_df = df_file[['Source']].copy()
    else:
        # If 'Source' is missing, use doc as the source name
        result_df = pd.DataFrame({'Source': [doc] * len(df_file)})

    result_df['cluster'] = cluster_labels

    # Add UMAP x and y coordinates if n_components is 2
    if n_components == 2:
        result_df['x'] = reduced_embeddings[:, 0]
        result_df['y'] = reduced_embeddings[:, 1]

    # Remove rows with cluster = -1
    result_df = result_df[result_df['cluster'] != -1]

    # Create output directory
    output_dir = 'evaluations/outputs/'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_clusters_HDBSCAN{"_2D" if n_components == 2 else ""}.csv')
    result_df.to_csv(output_file, index=False)
    print(f"Clustering completed. Results saved to '{os.path.basename(output_file)}'.")

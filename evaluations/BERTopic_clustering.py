import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.preprocessing import StandardScaler
import hdbscan
import os
import matplotlib.pyplot as plt
import seaborn as sns

def bertopic_clustering(df_file, embeddings, run_id, doc, model):
    """
    Run topic modeling using BERTopic with proper UMAP+HDBSCAN pipeline.

    Steps:
    1. Standardize embeddings.
    2. Apply UMAP to reduce to ~10D for clustering.
    3. Run BERTopic (HDBSCAN over UMAP-reduced embeddings).
    4. Use a separate UMAP projection to 2D for plotting.
    5. Save results.
    """

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # UMAP for clustering (10D)
    umap_cluster = UMAP(n_neighbors=30, min_dist=0.0, n_components=10, metric='cosine', random_state=42)
    reduced_embeddings_for_clustering = umap_cluster.fit_transform(embeddings_scaled)

    # HDBSCAN and BERTopic
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1, metric='euclidean')
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=None)
    topics, probs = topic_model.fit_transform(df_file['Content'], embeddings=reduced_embeddings_for_clustering)

    # UMAP for visualization (2D)
    umap_visual = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, metric='cosine', random_state=42)
    reduced_embeddings_2d = umap_visual.fit_transform(embeddings_scaled)

    # Prepare result dataframe
    if 'Source' in df_file.columns:
        result_df = df_file[['Source']].copy()
    else:
        result_df = pd.DataFrame({'Source': [doc] * len(df_file)})

    result_df['cluster'] = topics
    result_df['x'] = reduced_embeddings_2d[:, 0]
    result_df['y'] = reduced_embeddings_2d[:, 1]

    # Filter out noise topics (optional)
    filtered_df = result_df[result_df['cluster'] != -1]
    df_file = df_file[result_df['cluster'] != -1]

    # Save results
    output_dir = 'evaluations/outputs/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_clusters_BERTopic.csv')
    filtered_df.to_csv(output_file, index=False)
    print(f"Clustering for {model} is complete. Results saved in '{os.path.basename(output_file)}'.")

    return filtered_df, df_file

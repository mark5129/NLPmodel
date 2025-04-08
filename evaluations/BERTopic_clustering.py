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
    Run topic modeling using BERTopic with UMAP dimensionality reduction and HDBSCAN clustering.

    Parameters:
    - embeddings: DataFrame containing the embeddings.
    - df_file: DataFrame containing the textual data. Must include at least the 'Content' column.
    - run_id: String identifier for the run (e.g., 'manualrun').
    - doc: String representing the document type (e.g., 'merged' or 'source name').
    - model: String representing the model name (e.g., 'XLM_Roberta').

    Steps:
    1. Standardize embeddings.
    2. Apply UMAP for dimensionality reduction.
    3. Use HDBSCAN within BERTopic for clustering.
    4. Map clusters to topics.
    5. Save results to CSV.
    """

    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Perform dimensionality reduction using UMAP
    n_components = 2  # Set the desired dimensionality
    umap_model = UMAP(n_neighbors=30, min_dist=0.1, n_components=n_components, random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings_scaled)

    # Define the HDBSCAN model
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    
    # Apply BERTopic
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=None)
    topics, probs = topic_model.fit_transform(df_file['Content'], embeddings=reduced_embeddings)

    # Handle Source column for merged vs individual files
    if 'Source' in df_file.columns:
        result_df = df_file[['Source']].copy()
    else:
        result_df = pd.DataFrame({'Source': [doc] * len(df_file)})
    
    result_df['cluster'] = topics

    # Add UMAP x and y coordinates if n_components is 2
    if n_components == 2:
        result_df['x'] = reduced_embeddings[:, 0]
        result_df['y'] = reduced_embeddings[:, 1]

    # Create output directory
    output_dir = 'evaluations/outputs/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_clusters_BERTopic{"_2D" if n_components == 2 else ""}.csv')
    result_df.to_csv(output_file, index=False)
    
    print(f"Clustering for {model} is complete. Results saved in '{os.path.basename(output_file)}'.")

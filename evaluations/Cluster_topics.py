import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.preprocessing import StandardScaler
import hdbscan
import os

def clustering_and_naming(embeddings, df_file, run_id, doc, model):
    """
    Run topic modeling using BERTopic with UMAP dimensionality reduction and HDBSCAN clustering.

    Parameters:
    - embeddings: DataFrame containing the embeddings.
    - df_file: DataFrame containing the textual data. Must include at least the 'Content' and 'Source' columns.
    - run_id: String identifier for the run (e.g., 'manualrun').
    - doc: String representing the document type (e.g., 'merged' or 'merged_embeddings').
    - model: String representing the model name (e.g., 'XLM_Roberta').

    The function performs the following steps:
    1. Standardizes the embeddings.
    2. Applies UMAP for dimensionality reduction.
    3. Uses HDBSCAN within BERTopic to perform clustering.
    4. Maps the resulting clusters to topic names.
    5. Saves the final results to a CSV file in the 'evaluations/outputs/' directory.
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Perform dimensionality reduction using UMAP (reducing to 2 dimensions)
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    reduced_embeddings = umap_model.fit_transform(embeddings_scaled)
    
    # Define the HDBSCAN model with custom parameters
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    
    # Apply BERTopic with the custom HDBSCAN model (skip the embedding model since embeddings are provided)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=None)
    topics, probs = topic_model.fit_transform(df_file['Content'], embeddings=reduced_embeddings)
    
    # Get topic information including topic names
    topic_info = topic_model.get_topic_info()
    
    # Create a DataFrame for the results, keeping only the 'Source' column from df_file
    result_df = df_file[['Source']].copy()
    result_df['topic_int'] = topics  # Add the cluster/topic id
    
    # Map each topic id to its name; use 'Noise' for points labeled as -1
    result_df['labels_layer'] = result_df['topic_int'].map(
        lambda x: topic_info[topic_info['Topic'] == x]['Name'].values[0] if x != -1 else 'Noise'
    )
    
    # Replace underscores with commas in 'labels_layer'
    result_df['labels_layer'] = result_df['labels_layer'].str.replace('_', ', ')

    result_df['topic_names'] = result_df['topic_int'].map(
        lambda x: topic_info[topic_info['Topic'] == x]['Name'].values[0] if x != -1 else 'Noise'
    )
    
    # Create the output directory if it does not exist
    output_dir = 'evaluations/outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the result DataFrame to a CSV file
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_output_clusters.csv')
    result_df.to_csv(output_file, index=False)
    
    print(f"Clustering for {model} is complete. Results saved in '{os.path.basename(output_file)}'.")


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
    6. Generates and saves UMAP scatter plots with topics and sources.
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Perform dimensionality reduction using UMAP 
    umap_model = UMAP(n_neighbors=30, min_dist=0.1, n_components=2)
    reduced_embeddings = umap_model.fit_transform(embeddings_scaled)
    
    # Define the HDBSCAN model with custom parameters
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)
    
    # Apply BERTopic with the custom HDBSCAN model (skip the embedding model since embeddings are provided)
    topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=None)
    topics, probs = topic_model.fit_transform(df_file['Content'], embeddings=reduced_embeddings)
    
    # Create a DataFrame for the results, keeping only the 'Source' column from df_file
    result_df = df_file[['Source']].copy()
    result_df['topic_int'] = topics  # Add the cluster/topic id
    
    # Create the output directory if it does not exist
    output_dir = 'evaluations/outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the result DataFrame to a CSV file
    output_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_output_clusters.csv')
    result_df.to_csv(output_file, index=False)
    
    print(f"Clustering for {model} is complete. Results saved in '{os.path.basename(output_file)}'.")

    # Plot the UMAP embeddings with topics
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=topics,
        palette="tab10",
        legend="full",
        s=50
    )
    plt.title(f"UMAP Embeddings Colored by Topics ({model} - {doc})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_umap_plot.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"UMAP scatter plot saved in '{os.path.basename(plot_file)}'.")

    palette = {
        "Sci Media": "#1f77b4",  # Blue
        "Pro Media": "#ff7f0e",  # Orange
        "Reg Media": "#2ca02c"   # Green
    }
    

    # Plot the UMAP embeddings with sources
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=df_file['Source'],
        palette=palette,
        legend="full",
        s=50
    )
    
    plt.title(f"UMAP Embeddings Colored by Source ({model} - {doc})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Source", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    source_plot_file = os.path.join(output_dir, f'{run_id}_{model}_{doc}_umap_source_plot.png')
    plt.savefig(source_plot_file)
    plt.close()
    print(f"UMAP scatter plot by source saved in '{os.path.basename(source_plot_file)}'.")

    # print distinct values in the topic_int column
    unique_topics = sorted(result_df['topic_int'].unique())
    print(f"Distinct topics found: {unique_topics}")

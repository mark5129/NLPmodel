import pandas as pd
import matplotlib.pyplot as plt
import os

def individual_embeddings_plot(embedding, cluster_model, source):
    """
    Load and visualize the individual embeddings from BERTopic clustering.
    
    This function loads the results of the BERTopic clustering and visualizes the clusters
    using scatter plots. Two plots are created: one colored by cluster and another by source.
    The plots are saved as PNG files.
    
    Returns:
        None
    """
    # Load the results from CSV
    df = pd.read_csv(f"evaluations/outputs/manualrun_{embedding}_{source}_clusters_{cluster_model}_2D.csv")
            
    # Ensure the output directory exists
    output_dir = 'Visualizations/Plots/' 
    os.makedirs(output_dir, exist_ok=True)

    # Create a scatter plot colored by cluster
    plt.figure(figsize=(10, 6))
    unique_clusters = df['cluster'].unique()
    for cluster in unique_clusters:
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster}', alpha=0.5)
    plt.title(f'{embedding} - {cluster_model} Clustering (Colored by Cluster)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the cluster-colored plot
    output_file = os.path.join(output_dir, f'manualrun_{source}_{embedding}_{cluster_model}_cluster_plot.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")

    # Create a scatter plot colored by source
    plt.figure(figsize=(10, 6))
    unique_sources = df['Source'].unique()
    for source in unique_sources:
        source_data = df[df['Source'] == source]
        plt.scatter(source_data['x'], source_data['y'], label=source, alpha=0.5)
    plt.title(f'{embedding} - {cluster_model} - {source} Clustering (Colored by Source)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the source-colored plot
    output_file = os.path.join(output_dir, f'manualrun_{source}_{embedding}_{cluster_model}_source_plot.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")
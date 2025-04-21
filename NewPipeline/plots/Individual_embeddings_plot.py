import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull  # Add this import

def individual_embeddings_plot(embedding, source, model):
    """
    Load and visualize the individual embeddings from BERTopic clustering.
    
    This function loads the results of the BERTopic clustering and visualizes the clusters
    using scatter plots. Two plots are created: one colored by cluster and another by source.
    The plots are saved as PNG files.
    
    Returns:
        None
    """
    # Load the results from CSV
    df = embedding

    # Create a scatter plot colored by cluster
    plt.figure(figsize=(10, 6))
    unique_clusters = df['cluster'].unique()
    for cluster in unique_clusters:
        cluster_data = df[df['cluster'] == cluster]
        cluster_points = cluster_data[['x', 'y']].values
        
        # Compute and plot convex hull for the cluster
        if len(cluster_points) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            plt.plot(
                np.append(hull_points[:, 0], hull_points[0, 0]),
                np.append(hull_points[:, 1], hull_points[0, 1]),
                linestyle='--', linewidth=1.5, label=f'Cluster {cluster} Outline'
            )
        
        # Plot points for the cluster
        plt.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster}', alpha=0.8, edgecolors='face', linewidth=0.5, marker='o')
    plt.title(f'{source}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the cluster-colored plot
    plt.savefig(f'NewPipeline/plots/{model}_{source}_cluster_plot.png')
    plt.close()
    print(f"Cluster plot saved for '{source}'")
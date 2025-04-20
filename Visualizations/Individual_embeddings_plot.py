import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull  # Add this import

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
    df = pd.read_csv(f"evaluations/outputs/manualrun_{embedding}_{source}_clusters_{cluster_model}.csv")
            
    # Ensure the output directory exists
    output_dir = 'Visualizations/Plots/' 
    os.makedirs(output_dir, exist_ok=True)

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
    plt.title(f'{embedding} - {cluster_model} - {source}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # Save the cluster-colored plot
    output_file = os.path.join(output_dir, f'manualrun_{embedding}_{cluster_model}_{source}_cluster_plot.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as '{output_file}'")
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull



def cluster_plot50(k_means, embeddings, df, current_id, doc_type, model_name, global_x_min, global_x_max, global_y_min, global_y_max):

    # Convert embeddings to NumPy array
    embeddings = np.array(embeddings)

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=min(30, (len(embeddings) - 1) // 3), random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Store axis limits globally
    x_min, x_max = data_map[:, 0].min(), data_map[:, 0].max()
    y_min, y_max = data_map[:, 1].min(), data_map[:, 1].max()

    # Update global axis limits if this is the first plot or new extremes are found
    if global_x_min is None or x_min < global_x_min:
        global_x_min = x_min
    if global_x_max is None or x_max > global_x_max:
        global_x_max = x_max
    if global_y_min is None or y_min < global_y_min:
        global_y_min = y_min
    if global_y_max is None or y_max > global_y_max:
        global_y_max = y_max

    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame(data_map, columns=["x", "y"])
    df_plot["title"] = df["Title"].values
    df_plot["cluster_int"] = k_means['topic_int']
    df_plot["cluster_name"] = k_means['main_topic_name']
    df_plot["BERTopic_name"] = k_means['topic_names_from_bertopic']
    df_plot["source"] = df["Source"].values  # Ensure DataFrame contains "Source" column

    # Define a fixed color palette for the sources
    palette = {
        "Sci Media": "#1f77b4",  # Blue
        "Pro Media": "#ff7f0e",  # Orange
        "Reg Media": "#2ca02c"   # Green
    }

    # Plot points and cluster outlines
    plt.figure(figsize=(10, 8))
    for cluster_id in sorted(df_plot["cluster_int"].unique()):

        cluster_points = df_plot[df_plot["cluster_int"] == cluster_id][["x", "y"]].values
        
        # Compute centroid of the cluster
        centroid_x = cluster_points[:, 0].mean()
        centroid_y = cluster_points[:, 1].mean()
        distances = np.sqrt((cluster_points[:, 0] - centroid_x) ** 2 + (cluster_points[:, 1] - centroid_y) ** 2)
        
        # Sort points by distance to centroid and split into two halves
        sorted_indices = np.argsort(distances)
        closest_half = sorted_indices[:len(sorted_indices) // 2]
        farthest_half = sorted_indices[len(sorted_indices) // 2:]
        
        # Assign cluster name based on the most occurring "BERTopic_name" in closest_half
        closest_half_indices = df_plot[df_plot["cluster_int"] == cluster_id].iloc[closest_half].index
        most_common_name = df_plot.loc[closest_half_indices, "BERTopic_name"].mode()[0]
        
        # Compute and plot convex hull for the closest half of the cluster
        if len(closest_half) >= 3:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_points[closest_half])
            hull_points = cluster_points[closest_half][hull.vertices]
            plt.plot(
            np.append(hull_points[:, 0], hull_points[0, 0]),
            np.append(hull_points[:, 1], hull_points[0, 1]),
            linestyle='--', linewidth=1.5
            )
        
        # Plot closest 50% points with cluster color
        plt.scatter(cluster_points[closest_half, 0], cluster_points[closest_half, 1], label=f"Cluster {cluster_id}", s=10)
        
        # Plot farthest 50% points in grey
        plt.scatter(cluster_points[farthest_half, 0], cluster_points[farthest_half, 1], color='grey', s=10)

        # Add cluster name as text at the centroid of the cluster
        plt.text(centroid_x, centroid_y, most_common_name, fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5))


    plt.title(f"50% Cluster Visualization for {model_name}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.xlim(global_x_min, global_x_max)
    plt.ylim(global_y_min, global_y_max)
    plt.grid(True)

    # Save Visualization to CSV
    output_dir = 'Visualizations/outputs/'

    plt.savefig(f"{output_dir}{current_id}_{doc_type}_{model_name}_clusterplot50.png")
    print(f"{model_name} with {doc_type}: 50% Cluster plot saved successfully for {current_id}")




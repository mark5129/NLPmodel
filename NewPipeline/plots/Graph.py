import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances

# Function to create a weighted similarity graph from the matrix
def create_weighted_graph(upper_tri_matrix, threshold):
    n = upper_tri_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if upper_tri_matrix[i, j] >= threshold:
                # Add weighted edges
                G.add_edge(i, j, weight=upper_tri_matrix[i, j])
    
    return G

# DBSCAN clustering using distance (inverse similarity)
def apply_dbscan_weighted(G, eps=0.5, min_samples=3):
    # Convert the weighted graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G, weight='weight')
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - adjacency_matrix

    # Apply DBSCAN on the distance matrix
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(distance_matrix)  # Inverse similarity for distance matrix
    
    return labels

# Visualize clusters
def visualize_clusters(G, labels):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        # Nodes with the same label will be colored similarly
        node_list = [i for i, lbl in enumerate(labels) if lbl == label]
        nx.draw_networkx_nodes(G, pos, node_list, node_size=200, node_color=plt.cm.jet(label / len(unique_labels)))
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    plt.title('DBSCAN Clustering on Weighted Document Similarity Graph')
    plt.show()

models = ['Specter2']

threshold = 4

for model in models:

    upper_tri_sim = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_naming_score.csv')

    # Normalize the upper triangular matrix to the range [0, 1]
    upper_tri_matrix = upper_tri_sim.to_numpy()
    upper_tri_matrix = upper_tri_matrix / upper_tri_matrix.max()
    
    # Create the graph (you can tweak the threshold)
    G = create_weighted_graph(upper_tri_matrix, threshold=0.1)
    
    # Adjust eps to fix clustering
    labels = apply_dbscan_weighted(G, eps=0.1, min_samples=2)

    print(f"DBSCAN labels for model {model}: {labels}")
    
    # Visualize the result
    visualize_clusters(G, labels)
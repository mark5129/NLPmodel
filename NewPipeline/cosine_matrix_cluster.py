import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation

def cluster_embeddings_with_affinity_propagation(embeddings):
    """
    Clusters embeddings using cosine similarity and Affinity Propagation.

    Args:
        embeddings (numpy.ndarray): A 2D array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: A 1D array of cluster IDs with the same length as the number of embeddings.
    """
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Set the preference (diagonal values)
    # Use the median of off-diagonal similarities as a neutral preference
    n = similarity_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    preference = np.median(similarity_matrix[mask])
    np.fill_diagonal(similarity_matrix, preference)

    # Apply Affinity Propagation
    affinity_propagation = AffinityPropagation(affinity='precomputed', random_state=42)
    affinity_propagation.fit(similarity_matrix)

    # Get the cluster IDs
    cluster_id = affinity_propagation.labels_

    return cluster_id

def merged_naming_table(merged_naming):
    """
    Simplifies the merged_naming DataFrame by grouping and summarizing data.

    Args:
        merged_naming (pd.DataFrame): The merged naming DataFrame from the mainscript.

    Returns:
        pd.DataFrame: A simplified DataFrame with cluster information, source counts, and topic details.
    """

    # First, pivot the document counts per source
    doc_counts = merged_naming.groupby(['cluster', 'Source'])['count'].sum().unstack(fill_value=0)

    # Now group and extract first values for the other columns
    meta_info = merged_naming.groupby('cluster').agg({
        'TF_IDF_topic_name': 'first',
        'percentage_of_documents': 'first',
        'percentage_limit': 'first',
        'Topic_terms': 'first'
    })

    # Merge both DataFrames on cluster index
    result = meta_info.join(doc_counts).reset_index()


    # Sort by cluster
    result = result.sort_values(by='cluster').reset_index(drop=True)

    return result



# models = ['MiniLm12', 'Specter2', 'XLM_Roberta']

# for model in models:

#     merged_naming = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_merged_naming.csv')

#     merged_naming_df = merged_naming_table(merged_naming)

#     # Reorder the columns in the desired order
#     column_order = ['cluster', 'pro', 'reg', 'sci', 'TF_IDF_topic_name', 'percentage_of_documents', 'percentage_limit', 'Topic_terms']
#     merged_naming_df = merged_naming_df[column_order]

#     merged_naming = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_merged_naming.csv')

# merged_naming = pd.read_csv(f'NewPipeline/clustering_outputs/MiniLm12_merged_naming_clusters.csv')

# # Create a new DataFrame with the desired columns
# new_df = merged_naming[['cluster', 'pro', 'reg', 'sci', 'Topic_terms']]

# new_df.to_csv(f'NewPipeline/clustering_outputs/MiniLm12_cluster_evaluation.csv', index=False)

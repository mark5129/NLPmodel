import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation


def cluster_embeddings_with_affinity_propagation(embeddings):
    """
    Clusters embeddings using cosine similarity and Affinity Propagation.
    The preference value is scaled internally to reduce the number of clusters.

    Args:
        embeddings (numpy.ndarray): A 2D array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: A 1D array of cluster IDs with the same length as the number of embeddings.
    """
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Calculate base preference from off-diagonal similarities
    n = similarity_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    base_preference = np.median(similarity_matrix[mask])

    # Scale the preference to control cluster count (lower = fewer clusters)
    preference_scale = 0.25  # You can adjust this to 0.5, 0.1, etc. if needed
    scaled_preference = base_preference * preference_scale

    # Set diagonal to the scaled preference
    np.fill_diagonal(similarity_matrix, scaled_preference)

    # Apply Affinity Propagation with precomputed similarity
    affinity_propagation = AffinityPropagation(affinity='precomputed', random_state=42)
    affinity_propagation.fit(similarity_matrix)

    return affinity_propagation.labels_


def merged_naming_table(merged_naming):
    """
    Simplifies the merged_naming DataFrame by grouping and summarizing data.

    Args:
        merged_naming (pd.DataFrame): The merged naming DataFrame from the mainscript.

    Returns:
        pd.DataFrame: A simplified DataFrame with cluster information, source counts, and topic details.
    """
    # Pivot the document counts per source
    doc_counts = merged_naming.groupby(['cluster', 'Source'])['count'].sum().unstack(fill_value=0)

    # Group and extract first values for metadata columns
    meta_info = merged_naming.groupby('cluster').agg({
        'TF_IDF_topic_name': 'first',
        'percentage_of_documents': 'first',
        'percentage_limit': 'first',
        'Topic_terms': 'first'
    })

    # Merge metadata and counts
    result = meta_info.join(doc_counts).reset_index()

    # Sort by cluster ID
    result = result.sort_values(by='cluster').reset_index(drop=True)

    return result


# Example usage loop (uncomment and adapt as needed)
# models = ['MiniLm12', 'Specter2', 'XLM_Roberta']
# for model in models:
#     merged_naming = pd.read_csv(f'NewPipeline/clustering_outputs/hdbscan_{model}_merged_naming.csv')
#     merged_naming_df = merged_naming_table(merged_naming)
#     column_order = ['cluster', 'pro', 'reg', 'sci', 'TF_IDF_topic_name', 'percentage_of_documents', 'percentage_limit', 'Topic_terms']
#     merged_naming_df = merged_naming_df[column_order]
#     merged_naming_df.to_csv(f'NewPipeline/clustering_outputs/hdbscan_{model}_merged_naming_cluster.csv')

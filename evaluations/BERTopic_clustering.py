from bertopic import BERTopic
import os

def bertopic_clustering(df_file, embeddings, current_id, doc_type, model_name):
    """
    Run BERTopic clustering on precomputed embeddings and return cluster labels.

    Parameters:
    texts (List[str]): The original texts to cluster.
    embeddings_df (pd.DataFrame): DataFrame with each row as a high-dimensional embedding.
4
    Returns:
    List[int]: Cluster labels assigned to each text.
    """
    # Convert the DataFrame to a list of lists (or a 2D numpy array)
    embeddings = embeddings.values.tolist()
    
    # Initialize BERTopic without an embedding model
    topic_model = BERTopic()
    
    # Fit model using the provided embeddings
    topics, _ = topic_model.fit_transform(df_file, embeddings)

    # Save embeddings to CSV
    output_dir = 'evaluations/outputs/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    topics.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_{model_name}_BERTopic.csv'), index=False)

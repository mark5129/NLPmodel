import pandas as pd
from umap import UMAP
from bertopic import BERTopic
from sklearn.preprocessing import StandardScaler
import hdbscan
import os

# Load the data
Latest_run_id = '4552557450'
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# Which embeddings?
doc_type = ['merged', 'merged_embeddings']

for doc in doc_type:
    # Iterate over each model
    for model in which_model:
        # Load embeddings
        embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_{doc}_{model}_embeddings.csv')
        
        # Perform dimensionality reduction using UMAP
        # First, standardize the embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # Reduce dimensions to 5D (or any other desired dimensions)
        umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        reduced_embeddings = umap.fit_transform(embeddings_scaled)
        
        # Define the HDBSCAN model with custom parameters
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=1)  # Adjust min_cluster_size to control cluster quantity

        # Apply BERTopic with the custom HDBSCAN model and skip the text input
        topic_model = BERTopic(hdbscan_model=hdbscan_model, embedding_model=None)
        topics, probs = topic_model.fit_transform(df_file['Content'], embeddings=reduced_embeddings)
        
        # Get the topic names for each cluster
        topic_info = topic_model.get_topic_info()

        # Create a DataFrame for storing the results, including topic names
        result_df = df_file[['Source']].copy()  # Only keep the 'Source' column
        result_df['topic_int'] = topics  # Add the topic_int column (formerly cluster_id)

        # Filter out noise (topic_int = -1)
        # result_df = result_df[result_df['topic_int'] != -1]  # Uncomment this line if you want to remove noise points

        # Add topic names to the result dataframe
        result_df['labels_layer'] = result_df['topic_int'].map(lambda x: topic_info[topic_info['Topic'] == x]['Name'].values[0] if x != -1 else 'Noise')
        result_df['topic_names'] = result_df['topic_int'].map(lambda x: topic_info[topic_info['Topic'] == x]['Name'].values[0] if x != -1 else 'Noise')
        
        # Save embeddings to CSV
        output_dir = 'evaluations/outputs/'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_df.to_csv(os.path.join(output_dir, f'manualrun_{model}_{doc}_output_clusters.csv'), index=False)

        print(f"Clustering for {model} is complete. Results saved in 'manualrun_{model}_{doc}_output_clusters.csv'.")

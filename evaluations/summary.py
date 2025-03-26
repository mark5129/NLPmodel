import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
from transformers import pipeline

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


def summarize_cluster_texts(cluster_texts, max_length=130, min_length=30):
    """
    Summarize a list of texts into a single consolidated summary.
    Uses a Hugging Face summarization pipeline as an example.
    Adjust parameters as needed.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Join all texts. You may choose more sophisticated approaches 
    # (e.g., chunking if texts are very large).
    combined_text = " ".join(cluster_texts)

    # Generate summary
    summary = summarizer(
        combined_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return summary[0]['summary_text']


def Summaries_clustering(embeddings, df, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]

    # Adjust perplexity based on the number of samples
    perplexity = min(30, (n_samples - 1) // 3)

    # Step 1: Create a data map using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Step 2: Perform hierarchical clustering
    n_clusters_list = [parameters['num_summaries']]  # Adjust these numbers for your desired hierarchy levels
    labels_layers = []
    summary_names = pd.DataFrame(columns=['summary_int', 'summary_names'])

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=parameters['random_state'])
        labels_int = kmeans.fit_predict(data_map)

        used_summaries = set()

        # Generate summaries for each cluster
        label_summary_map = {}
        for label in range(n_clusters):
            indices = np.where(labels_int == label)[0]
            if len(indices) == 0:
                label_summary_map[label] = f"{label}: No data"
                continue
            cluster_texts = df['Content'].iloc[indices].astype(str).tolist()
            summary = summarize_cluster_texts(cluster_texts)  # Get the summary for the cluster

            label_summary_map[label] = f"{label}: {summary}"  # Use summary as the label name

            # Save summaries for each cluster
            summary_names.loc[label, 'summary_int'] = label
            summary_names.loc[label, 'summary_names'] = summary

        # Convert integer labels to summaries
        labels_summary = np.array([label_summary_map.get(label, f"{label}: Unknown") for label in labels_int])
        labels_layers.append(labels_summary)

    # Save embeddings to CSV
    output_dir = 'evaluations/outputs/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_labels_int = pd.DataFrame(labels_int)
    df_labels_int.columns = ['summary_int']

    df_labels_int = df_labels_int.merge(summary_names, on='summary_int', how='left')

    sources = df['Source']

    df_labels_int['Source'] = sources
    
    df_labels_int.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_{model_name}_summary.csv'), index=False)
    print(f'{model_name} k-means clustering saved for ID {current_id}')



#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '5173858928'

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# Run clustering for merged document embeddings
for model in which_model:
    
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_{model}_embeddings.csv')
    Summaries_clustering(embeddings, df_file, 'manualrun', 'merged', model)
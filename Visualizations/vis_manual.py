import pandas as pd

from Merged_embeddings_plot import merged_embeddings_plot

embeddings = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']
cluster_models = ['HDBSCAN', 'BERTopic']

for embedding in embeddings:
    for cluster_model in cluster_models:
        merged_embeddings_plot(embedding, cluster_model)
import pandas as pd

from BERTopic_cluster_topics import K_means_clustering

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Load the merged media file to get texts
file = ['modelling/outputs/XLM_Roberta/6751423855_merged_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2Actually/6751423855_merged_Specter2Actually_embeddings.csv',
        'modelling/outputs/MiniLm12/6751423855_merged_MiniLm12_embeddings.csv']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for i in range(len(which_model)):
    embeddings = pd.read_csv(file[i])
    model = which_model[i]
    K_means_clustering(embeddings, df_file, 'manualrun', 'merged', model)

file = ['modelling/outputs/XLM_Roberta/6751423855_merged_embeddings_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2Actually/6751423855_merged_embeddings_Specter2Actually_embeddings.csv',
        'modelling/outputs/MiniLm12/6751423855_merged_embeddings_MiniLm12_embeddings.csv']

for i in range(len(which_model)):
    embeddings = pd.read_csv(file[i])
    model = which_model[i]
    K_means_clustering(embeddings, df_file, 'manualrun', 'merged_embeddings', model)

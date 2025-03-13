import pandas as pd

from K_means_clustering import K_means_clustering

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2', 'MiniLm12', 'BERTopic']

# Load the merged media file to get texts
file = ['modelling/outputs/XLM_Roberta/2418438043_merged_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2/2418438043_merged_Specter2_embeddings.csv',
        'modelling/outputs/MiniLm12/2418438043_merged_MiniLm12_embeddings.csv',
        'modelling/outputs/BERTopic/2418438043_merged_BERTopic_embeddings.csv']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for i in range(len(which_model)):
    embeddings = pd.read_csv(file[i])
    model = which_model[i]
    K_means_clustering(embeddings, df_file, 'manualrun', 'merged', model)

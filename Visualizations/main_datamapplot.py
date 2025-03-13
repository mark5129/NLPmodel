import pandas as pd

from datamapplot_with_naming import data_mapplot_with_naming
from topic_source_plot import topic_source_plot

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2', 'MiniLm12', 'BERTopic']

# Load the merged media file to get texts
file = ['modelling/outputs/XLM_Roberta/2418438043_merged_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2/2418438043_merged_Specter2_embeddings.csv',
        'modelling/outputs/MiniLm12/2418438043_merged_MiniLm12_embeddings.csv',
        'modelling/outputs/BERTopic/2418438043_merged_BERTopic_embeddings.csv',
        'evaluations/outputs/manualrun_merged_XLM_Roberta_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_Specter2_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_MiniLm12_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_BERTopic_Kmeans.csv']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for i in range(len(which_model)):
    embeddings = pd.read_csv(file[i])
    kmeans = pd.read_csv(file[i+len(which_model)])
    model = which_model[i]
    data_mapplot_with_naming(embeddings, df_file, 'manualrun', 'merged', model)
    topic_source_plot(kmeans, 'manualrun', 'merged', model)

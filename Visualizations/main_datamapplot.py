import pandas as pd

from datamapplot_with_naming import data_mapplot_with_naming
from topic_source_plot import topic_source_plot

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

merged_file = ['modelling/outputs/XLM_Roberta/6751423855_merged_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2Actually/6751423855_merged_Specter2Actually_embeddings.csv',
        'modelling/outputs/MiniLm12/6751423855_merged_MiniLm12_embeddings.csv',
        'evaluations/outputs/manualrun_merged_XLM_Roberta_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_Specter2Actually_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_MiniLm12_Kmeans.csv']

embed_merged_file = ['modelling/outputs/XLM_Roberta/6751423855_merged_embeddings_XLM_Roberta_embeddings.csv',
        'modelling/outputs/Specter2Actually/6751423855_merged_embeddings_Specter2Actually_embeddings.csv',
        'modelling/outputs/MiniLm12/6751423855_merged_embeddings_MiniLm12_embeddings.csv',
        'evaluations/outputs/manualrun_merged_embeddings_XLM_Roberta_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_embeddings_Specter2Actually_Kmeans.csv',
        'evaluations/outputs/manualrun_merged_embeddings_MiniLm12_Kmeans.csv']


df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for i in range(len(which_model)):
    embeddings = pd.read_csv(merged_file[i])
    kmeans = pd.read_csv(merged_file[i+len(which_model)])
    model = which_model[i]
    data_mapplot_with_naming(embeddings, df_file, 'manualrun', 'merged', model)
    topic_source_plot(kmeans, 'manualrun', 'merged', model)


for i in range(len(which_model)):
    embeddings = pd.read_csv(embed_merged_file[i])
    kmeans = pd.read_csv(embed_merged_file[i+len(which_model)])
    model = which_model[i]
    data_mapplot_with_naming(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    topic_source_plot(kmeans, 'manualrun', 'merged_embeddings', model)

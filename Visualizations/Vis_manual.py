import pandas as pd

from datamapplot_with_naming import data_mapplot_with_naming
from topic_source_plot import topic_source_plot
from Bokeh import create_bokeh_plot
from ClusterSource_outlines import outline_plot
from Cluster_plot import cluster_plot
from Cluster_plot50 import cluster_plot50

# This script only runs on embeddings from various models.

# Determine the latest run ID to know which embeddings to use
# Important that this number is the same as the one used in the manual eval script also
Latest_run_id = '1303156299'

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_{model}_embeddings.csv')
    #kmeans = pd.read_csv(f'evaluations/outputs/manualrun_{model}_merged_output_clusters.csv')
    kmeans = pd.read_csv(f'evaluations/outputs/manualrun_merged_{model}_Kmeans.csv')

    data_mapplot_with_naming(kmeans, embeddings, df_file, 'manualrun', 'merged', model)
    topic_source_plot(kmeans, 'manualrun', 'merged', model)
    create_bokeh_plot(kmeans, embeddings, df_file, 'manualrun', 'merged', model)
    outline_plot(kmeans, embeddings, df_file, 'manualrun', 'merged', model)
    cluster_plot(kmeans, embeddings, df_file, 'manualrun', 'merged', model)
    cluster_plot50(kmeans, embeddings, df_file, 'manualrun', 'merged', model)

for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')
    #kmeans = pd.read_csv(f'evaluations/outputs/manualrun_{model}_merged_embeddings_output_clusters.csv')
    kmeans = pd.read_csv(f'evaluations/outputs/manualrun_merged_embeddings_{model}_Kmeans.csv')

    data_mapplot_with_naming(kmeans, embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    topic_source_plot(kmeans, 'manualrun', 'merged_embeddings', model)
    create_bokeh_plot(kmeans, embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    outline_plot(kmeans, embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    cluster_plot(kmeans, embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    cluster_plot50(kmeans, embeddings, df_file, 'manualrun', 'merged_embeddings', model)

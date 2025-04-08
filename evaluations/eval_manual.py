import pandas as pd

from UMAP_dimensionality_reduction import UMAP_reduction
from HDBSCAN_clustering import run_hdbscan_from_data_map
from BERTopic_clustering import bertopic_clustering
from hdbscan_cluster import clustering_with_umap_hdbscan

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']
sources = ['pro', 'reg', 'sci']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '1303156299'

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# Run clustering for merged embeddings
for model in which_model:

    if parameters['embeddings'] == 'Merged':

        embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')

        #data_map = UMAP_reduction(embeddings, 'manualrun', 'merged_embeddings', model)
        #clustered_df, cluster_labels = run_hdbscan_from_data_map(data_map, 'manualrun', 'merged_embeddings', model)
        clustering_with_umap_hdbscan(df_file, embeddings, 'manualrun', 'merged_embeddings', model)
        bertopic_clustering(df_file, embeddings, 'manualrun', 'merged_embeddings', model)
    
    elif parameters['embeddings'] == 'Individual':

        for source in sources:

            df_file = pd.read_csv(f'data/{source}_media_stemmed_eng.csv')
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_{source}_{model}_embeddings.csv')
            
            clustering_with_umap_hdbscan(df_file, embeddings, 'manualrun', source, model)
            bertopic_clustering(df_file, embeddings, 'manualrun', source, model)

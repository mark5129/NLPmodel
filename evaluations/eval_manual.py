import pandas as pd

from BERTopic_clustering import bertopic_clustering
from HDBSCAN_clustering import clustering_with_umap_hdbscan
from TFIDF_cluster_Naming import TFIDF_cluster_Naming
from BERTopic_cluster_Naming import BERTopic_cluster_Naming

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

#Load embeddings from embeddings file
which_model = ['Specter2Actually']
sources = ['pro', 'reg', 'sci']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '3886251367'

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
            
            df_file1 = pd.read_csv(f'data/{source}_media_cleaned_eng.csv')
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_{source}_{model}_embeddings.csv')
            
            result_df, df_file = clustering_with_umap_hdbscan(df_file1, embeddings, 'manualrun', source, model)

            result_df, df_file = TFIDF_cluster_Naming(df_file, result_df, 'HDBSCAN', source, model)
            BERTopic_cluster_Naming(df_file, result_df, 'HDBSCAN', source, model)

            result_df, df_file = bertopic_clustering(df_file1, embeddings, 'manualrun', source, model)

            result_df, df_file = TFIDF_cluster_Naming(df_file, result_df, 'BERTopic', source, model)
            BERTopic_cluster_Naming(df_file, result_df, 'BERTopic', source, model)

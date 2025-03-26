import pandas as pd

from BERTopic_cluster_topics import Bertopic_clustering_naming
from TFIDF_cluster_topics import TFIDF_clustering
# This script only runs on embeddings from various models.

# Which cluster namning technique to use?
# TFIDF or Bertopic
#namning_technique = 'Bertopic'
namning_technique = 'TFIDF'

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '4552557450'

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# Run clustering for merged document embeddings
for model in which_model:
    
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_{model}_embeddings.csv')

    if namning_technique == 'TFIDF':
        TFIDF_clustering(embeddings, df_file, 'manualrun', 'merged', model)
    elif namning_technique == 'Bertopic':
        Bertopic_clustering_naming(embeddings, df_file, 'manualrun', 'merged', model)
    else:
        print('No clustering method selected')
        break

# Run clustering for merged embeddings
for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')

    if namning_technique == 'TFIDF':
        TFIDF_clustering(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    elif namning_technique == 'Bertopic':
        Bertopic_clustering_naming(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    else:
        print('No clustering method selected')
        break
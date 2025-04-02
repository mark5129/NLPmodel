import pandas as pd

from BERTopic_cluster_topics import Bertopic_clustering_naming
from TFIDF_cluster_topics import TFIDF_clustering
from Cluster_topics import clustering_and_naming
from addTopic_names import addBERTopic_names
from FindOptimalTopic import assign_main_topic_name
# This script only runs on embeddings from various models.

# Which cluster namning technique to use?
# TFIDF or Bertopic
#namning_technique = 'Bertopic'
namning_technique = 'TFIDF'
#namning_technique = 'BERT'

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '1303156299'

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# # Run clustering for merged document embeddings
# for model in which_model:
    
#     embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_{model}_embeddings.csv')

#     if namning_technique == 'TFIDF':
#         TFIDF_clustering(embeddings, df_file, 'manualrun', 'merged', model)
#         clustering_and_naming(embeddings, df_file, 'manualrun', 'merged', model)
#     elif namning_technique == 'Bertopic':
#         Bertopic_clustering_naming(embeddings, df_file, 'manualrun', 'merged', model)
#     else:
#         print('No clustering method selected')
#         break
    
#     addBERTopic_names('merged',model)
#     assign_main_topic_name('merged',model)

# Run clustering for merged embeddings
for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')

    if namning_technique == 'TFIDF':
        TFIDF_clustering(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
        clustering_and_naming(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    elif namning_technique == 'Bertopic':
        Bertopic_clustering_naming(embeddings, df_file, 'manualrun', 'merged_embeddings', model)
    else:
        print('No clustering method selected')
        break

    addBERTopic_names('merged_embeddings',model)
    assign_main_topic_name('merged_embeddings',model)
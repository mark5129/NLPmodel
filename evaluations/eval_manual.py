import pandas as pd

from UMAP_dimensionality_reduction import UMAP_reduction

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Determine the latest run ID to know which embeddings to use
Latest_run_id = '1303156299'

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

# Run clustering for merged embeddings
for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')

    UMAP_reduction(embeddings, 'manualrun', 'merged_embeddings', model)

import pandas as pd

from Visualizations.datamapplot_with_naming import data_mapplot_with_naming

# This script only runs on embeddings from various models.

# Load the merged media file to get texts
df = pd.read_csv('data/merged_media_stemmed_eng.csv')

#Load embeddings from embeddings file
which_model = ['MiniLm12', 'Specter2', 'XLM_Roberta']


embeddings = pd.read_csv(f'outputs/{model}_op/{current_id}_merged_{model}_embeddings.csv')
data_mapplot_with_naming(embeddings, df, 'Manual run', 'merged', model)


data_mapplot_with_naming(embeddings, df, current_id, doc_type, model_name)
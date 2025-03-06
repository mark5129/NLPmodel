import pandas as pd

from datamapplot_with_naming import data_mapplot_with_naming

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2', 'MiniLm12']

# Load the merged media file to get texts
file = ['outputs/XLM_Roberta/5694260457_merged_XLM_Roberta_embeddings.csv',
        'outputs/Specter2/5694260457_merged_Specter2_embeddings.csv',
        'outputs/MiniLm12/5694260457_merged_MiniLm12_embeddings.csv']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for i in range(len(which_model)):
    embeddings = pd.read_csv(file[i])
    model = which_model[i]
    data_mapplot_with_naming(embeddings, df_file, 'manualrun', 'merged', model)

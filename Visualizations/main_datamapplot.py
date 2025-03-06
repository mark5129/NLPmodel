import pandas as pd

from datamapplot_with_naming import data_mapplot_with_naming

# This script only runs on embeddings from various models.


#Load embeddings from embeddings file
which_model = ['MiniLm12', 'Specter2', 'XLM_Roberta']

# Load the merged media file to get texts
file = 'outputs/MiniLm12/9076229774_merged_MiniLm12_embeddings.csv'
df_file = 'data/merged_media_stemmed_eng.csv'

#file = 'outputs/Specter2/4575903620_Specter2_embeddings.csv'
#df_file = 'data/pro_media_stemmed_eng.csv'



df = pd.read_csv(df_file)
embeddings = pd.read_csv(file)
data_mapplot_with_naming(embeddings, df, 'Manual run', 'Merged', 'MiniLm12')

import pandas as pd

data = pd.read_csv('data/merged_media_stemmed_eng.csv') # Your csv file name


data.reset_index(inplace=True)
data.rename(columns={'index': 'id'}, inplace=True)
data.to_csv('data/merged_media_stemmed_eng.csv', index=False)
import pandas as pd
# i have a csv file with columns: Source,cluster,count,TF_IDF_topic_name,percentage_of_documents,percentage_limit
# pro,0,80,"offshore wind, 0.0140, 64, 401 | wind turbines, 0.0098, 55, 167 | energy island, 0.0097, 75, 531 | north sea, 0.0093, 56, 240 | data center, 0.0086, 5, 62",0.7350000000000001,1.0

models = ['Specter2'] # , 'XLM_Roberta', 'MiniLm12'

for model in models:
    naming_table = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_naming.csv')
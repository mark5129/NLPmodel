import pandas as pd

from Merged_embeddings_plot import merged_embeddings_plot
from Individual_embeddings_plot import individual_embeddings_plot
from namingtables import naming_tableIndividual

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

embeddings = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']
cluster_models = ['HDBSCAN', 'BERTopic']
sources = ['pro', 'reg', 'sci']

for embedding in embeddings:
    for cluster_model in cluster_models:

        if parameters['visualization'] == 'Merged':
            merged_embeddings_plot(embedding, cluster_model)

        elif parameters['visualization'] == 'Individual':
            for source in sources:
                individual_embeddings_plot(embedding, cluster_model, source)
            
            naming_tableIndividual(embedding, cluster_model)


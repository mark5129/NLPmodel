import pandas as pd

from plots.Individual_embeddings_plot import individual_embeddings_plot
from plots.Merged_embeddings_plot import merged_embeddings_plot
from set_plot_style import set_style

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

set_style()

models = ['Specter2'] # , 'XLM_Roberta', 'MiniLm12'
sources = ['pro', 'reg', 'sci']

for model in models:
    clustering = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_clustering.csv')

    merged_embeddings_plot(clustering, model)

    for source in sources:

        clusterings = clustering[clustering['Source'] == source]
        
        individual_embeddings_plot(clusterings, source, model)


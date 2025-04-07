import numpy as np
from umap import UMAP
import os
import pandas as pd

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def UMAP_reduction(embeddings, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]
    
    umapmodel = UMAP(n_neighbors=15, min_dist=0.1, n_components=parameters['umap_dimensions'])
    data_map = umapmodel.fit_transform(embeddings)

    # Save embeddings to CSV
    output_dir = 'evaluations/outputs/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_data_map = pd.DataFrame(data_map)
    #df_data_map.columns = ['X', 'Y']

    df_data_map.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_{model_name}_UMAP.csv'), index=False)
    print(f'{model_name} Umap dimensionality reduction saved for ID {current_id}')

    return data_map
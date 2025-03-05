# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import os
from sentence_transformers import SentenceTransformer
import pandas as pd

def MiniLM12(text_data, current_id, doc_type):
    
    #model = SentenceTransformer('all-MiniLM-L6-v2') ## English model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') ## Multilingual model
    # Step 3: Generate embeddings

    embeddings = model.encode(text_data, show_progress_bar=True, batch_size=32)

    # Save embeddings to CSV
    output_dir = 'outputs/MiniLm12'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_embeddings = pd.DataFrame(embeddings)
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_MiniLm12_embeddings.csv'), index=False)
    print(f'MiniLm12 embeddings saved for ID {current_id}')

    return embeddings
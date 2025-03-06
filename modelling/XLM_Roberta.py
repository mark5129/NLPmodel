# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import pandas as pd
import os


# Implement XML-Roberta topic modelling

from sentence_transformers import SentenceTransformer
#from sklearn.decomposition import LatentDirichletAllocation

def XLM_Roberta_model(text_column, current_id, doc_type):

    # Load the XLM-Roberta model
    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    # Generate embeddings
    embeddings = model.encode(text_column, show_progress_bar=True, batch_size=32)

    # Convert the embeddings to a DataFrame
    embeddings_df = pd.DataFrame(embeddings)

    # Save the topics to a CSV file
    output_dir = 'outputs/XLM_Roberta_topics'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    embeddings_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_XLM_Roberta_embeddings.csv'), index=False)
    print('XLM-Roberta Topics saved')


    return embeddings_df


reg_media = pd.read_csv(parameters['reg_media_stemmed_dir'])
pro_media = pd.read_csv(parameters['pro_media_stemmed_dir'])

reg_text_column = reg_media['Content']
pro_text_column = pro_media['Full text']

XLM_Roberta_model(reg_text_column, '101', 'reg')
XLM_Roberta_model(pro_text_column, '101', 'pro')
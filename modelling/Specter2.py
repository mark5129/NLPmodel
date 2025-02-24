# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)
    
# modelling/specter2.py

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import csv

def Specter2Model(text_column, current_id):
    """
    Generates embeddings using Specter2 for scientific literature comparison.

    Parameters:
    text_column (pd.Series): Text data (scientific/media articles).
    current_id (str): Unique ID for saving outputs.

    Returns:
    embeddings (torch.Tensor): Specter2 embeddings for all documents.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")


    embeddings_list = []

    # Generate embeddings for each document
    for text in text_column:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings_list.append(embeddings.flatten())

    # Save embeddings to CSV
    output_dir = 'outputs/Specter2_op'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_Specter2_embeddings.csv'), index=False)
    print(f'Specter2 embeddings saved for ID {current_id}')

    return df_embeddings


#if parameters['train_specter2'] == True:

    #Specter2Model(reg_text_column, current_id)
    #Specter2Model(pro_text_column, current_id)
    #print('Specter2 embeddings are generated.')

#else:
    #print('Model training is turned off in parameters.yaml')
    

if __name__ == "__main__":
    # Dummy example data for testing
    import pandas as pd

    # Example test data
    sample_texts = pd.Series([
        "This is an example of a scientific article discussing renewable energy.",
        "Another article exploring the challenges and opportunities of energy islands."
    ])

    # Provide a test ID for saving outputs
    test_current_id = "test_run"

    # Call the function
    Specter2Model(sample_texts, test_current_id)
    print("Specter2 model ran successfully with test data.")

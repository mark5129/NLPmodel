import yaml
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os


def General_Specter2(text_column, current_id, doc_type):
    """
    Generates embeddings using a model designed for general English text.

    Parameters:
    text_column (pd.Series): Text data (articles, news, etc.).
    current_id (str): Unique ID for saving outputs.
    doc_type (str): Type of document being processed.

    Returns:
    pd.DataFrame: Embeddings for all documents.
    """
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    embeddings_list = []

    # Generate embeddings for each document
    for text in text_column:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings_list.append(embeddings.flatten())

    # Save embeddings to CSV
    output_dir = 'modelling/outputs/GeneralText'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_GeneralText_embeddings.csv'), index=False)
    print(f'GeneralText embeddings saved for ID {current_id}')

    return df_embeddings
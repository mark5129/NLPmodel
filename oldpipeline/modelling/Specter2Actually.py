from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import numpy as np

def Specter2ActuallyModel(text_column, current_id, doc_type):
    """
    Generates embeddings using Specter2 for scientific literature comparison.

    Parameters:
    text_column (pd.Series): Text data (scientific/media articles).
    current_id (str): Unique ID for saving outputs.

    Returns:
    embeddings (np.ndarray): Specter2 embeddings for all documents.
    """

    # ✅ Load the correct Specter2 model using SentenceTransformer
    model = SentenceTransformer("allenai/specter2_base")


    # Generate embeddings
    embeddings_list = model.encode(text_column.tolist(), show_progress_bar=True)

    # Save embeddings to CSV
    output_dir = 'modelling/outputs/Specter2Actually'
    os.makedirs(output_dir, exist_ok=True)

    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_Specter2Actually_embeddings.csv'), index=False)
    print(f'Specter2 embeddings saved for ID {current_id}')

    return df_embeddings

import pandas as pd
import os

def merge_embeddings(model_name, current_id, output_dir="modelling/outputs"):
    """
    Merges the embeddings from different text sources (Pro Media, Reg Media, Sci Media)
    into a single file for a given model.

    Parameters:
    - model_name (str): Name of the model (e.g., 'MiniLM12', 'Specter2Actually', 'XLM_Roberta').
    - current_id (str): Unique identifier for the run.
    - output_dir (str): Directory where the embeddings are stored.
    
    Returns:
    - merged_df (pd.DataFrame): Merged dataframe containing all embeddings.
    """

    # Define file paths for each source
    pro_file = os.path.join(output_dir, model_name, f"{current_id}_pro_{model_name}_embeddings.csv")
    reg_file = os.path.join(output_dir, model_name, f"{current_id}_reg_{model_name}_embeddings.csv")
    sci_file = os.path.join(output_dir, model_name, f"{current_id}_sci_{model_name}_embeddings.csv")

    pro_media_df = pd.read_csv(pro_file)
    reg_media_df = pd.read_csv(reg_file)
    sci_media_df = pd.read_csv(sci_file)

    merged_df = pd.concat([pro_media_df, reg_media_df, sci_media_df])

    # Save merged embeddings
    merged_output_path = os.path.join(output_dir, model_name, f"{current_id}_merged_embeddings_{model_name}_embeddings.csv")
    merged_df.to_csv(merged_output_path, index=False)


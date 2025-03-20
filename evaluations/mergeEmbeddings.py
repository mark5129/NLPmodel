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

    # Load embeddings if they exist
    dfs = []
    for file, source in zip([pro_file, reg_file, sci_file], ["Pro Media", "Reg Media", "Sci Media"]):
        if os.path.exists(file):
            df = pd.read_csv(file)
            df["Source"] = source  # Add a column to track the source
            dfs.append(df)
        else:
            print(f"Warning: {file} not found. Skipping.")

    # Merge all available embeddings
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged embeddings
        merged_output_path = os.path.join(output_dir, model_name, f"{current_id}_merged_{model_name}_embeddings.csv")
        merged_df.to_csv(merged_output_path, index=False)
        
        print(f"Merged embeddings saved: {merged_output_path}")
        return merged_df
    else:
        print(f"No embeddings found for model {model_name}. Merging skipped.")
        return None

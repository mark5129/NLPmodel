import pandas as pd 
import matplotlib.pyplot as plt
import hdbscan
import os
def run_hdbscan_from_data_map(data_map, current_id, doc_type, model_name, min_cluster_size=10):
    # Lag DataFrame fra UMAP
    df = pd.DataFrame(data_map)
    dim_cols = [f'UMAP{i+1}' for i in range(data_map.shape[1])]
    df.columns = dim_cols

    # Kjør HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(data_map)
    df['cluster'] = labels

    # Lagre til CSV automatisk
    output_dir = "evaluations/outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{current_id}_{doc_type}_{model_name}_UMAP.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Lagret clustering-resultat til: {output_path}")


    return df, labels

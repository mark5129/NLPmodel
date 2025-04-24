import pandas as pd

# Load the BERTopic library
from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from naming_scripts.TFIDF_cluster_Naming import TFIDF_cluster_Naming
from naming_scripts.namingtables import naming_tableIndividual
from cosine_matrix_cluster import cluster_embeddings_with_affinity_propagation
from cosine_matrix_cluster import merged_naming_table

models = ['MiniLm12', 'Specter2', 'XLM_Roberta']

clusterings = ['AP', 'hdbscan']

sources = ['pro', 'reg', 'sci']

for clustering in clusterings:
    for model in models:

        # Load a different sentence transformer model
        if model == 'XLM_Roberta':
            embedding_model = SentenceTransformer("xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
        elif model == 'Specter2':
            embedding_model = SentenceTransformer("allenai/specter2_base")
        elif model == 'MiniLm12':
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

        # Define dimensions for umap
        umap_model = umap.UMAP(
            n_neighbors=30, 
            min_dist=0.0, 
            n_components=5, 
            metric='cosine', 
            random_state=42
        )
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=15, 
            min_samples=1, 
            metric='euclidean',
            prediction_data=True  # Enable prediction data
        )

        if True:

            # list of dataframes to store topic information
            cluster_info = {
                'pro': pd.DataFrame(),
                'reg': pd.DataFrame(),
                'sci': pd.DataFrame()
            }

            naming_info = {
                'pro': pd.DataFrame(),
                'reg': pd.DataFrame(),
                'sci': pd.DataFrame()
            }

            embedding_info = {
                'pro': pd.DataFrame(),
                'reg': pd.DataFrame(),
                'sci': pd.DataFrame()
            }

            for source in sources:

                df_file1 = pd.read_csv(f'data/{source}_media_cleaned_eng.csv')

                df_text_column = df_file1['Content']

                embeddings = embedding_model.encode(df_text_column.tolist(), show_progress_bar=True)
                # Convert embeddings to a DataFrame
                embeddings_df = pd.DataFrame(embeddings)
            
                reduced_embeddings = umap_model.fit_transform(embeddings, show_progress_bar=True)

                norm_data = normalize(reduced_embeddings, norm='l2')

                cluster_id = cluster_embeddings_with_affinity_propagation(embeddings)

                cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)

                # UMAP for 2D visualization
                umap_vis = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, metric='cosine', random_state=42)
                vis_2d = umap_vis.fit_transform(embeddings)

                topic_info = pd.DataFrame({
                    'cluster': cluster_id,
                    'Source': source,
                    'x': vis_2d[:, 0],
                    'y': vis_2d[:, 1]
                })

                # Filter out noise points (cluster = -1)
                filtered_df = topic_info[topic_info['cluster'] != -1]
                embeddings_df = embeddings_df[topic_info['cluster'] != -1]
                df_file = df_file1[topic_info['cluster'] != -1]
                
                cluster_info[source] = filtered_df
                embedding_info[source] = embeddings_df

                result_df = TFIDF_cluster_Naming(df_file, filtered_df, threshold = 0)
                naming_info[source] = result_df

            # Merge the DataFrames
            merged_naming = pd.concat([naming_info['pro'], naming_info['reg'], naming_info['sci']], ignore_index=True)
            merged_clustering = pd.concat([cluster_info['pro'], cluster_info['reg'], cluster_info['sci']], ignore_index=True)
            merged_embedding = pd.concat([embedding_info['pro'], embedding_info['reg'], embedding_info['sci']], ignore_index=True)

            merged_clustering.to_csv(f'NewPipeline/clustering_outputs/indiv/{model}_clustering.csv', index=False)

            print(f"\nBERTopic clustering for {model} saved.\n")

            # Save the embeddings to a CSV file
            merged_embedding.to_csv(f'NewPipeline/clustering_outputs/indiv/{model}_embeddings.csv', index=False)

            print(f"\nBERTopic embedding for {model} saved.\n")

            merged_naming.drop(columns=['x', 'y'], inplace=True)

            merged_naming = naming_tableIndividual(merged_naming)
            merged_naming.to_csv(f'NewPipeline/clustering_outputs/indiv/{model}_naming.csv', index=False)

            print(f"\nNaming table for {model} saved.\n")

        if False:
            text_info = {
                'pro': pd.DataFrame(),
                'reg': pd.DataFrame(),
                'sci': pd.DataFrame()
            }

            for source in sources:

                df_file1 = pd.read_csv(f'data/{source}_media_cleaned_eng.csv')

                df_file1['Source'] = source

                text_info[source] = df_file1
            
            merged_text = pd.concat([text_info['pro'], text_info['reg'], text_info['sci']], ignore_index=True)

            df_text_column = merged_text['Content']

            embeddings = embedding_model.encode(df_text_column.tolist(), show_progress_bar=True)

            # Convert embeddings to a DataFrame
            embeddings_df = pd.DataFrame(embeddings)

            reduced_embeddings = umap_model.fit_transform(embeddings, show_progress_bar=True)

            norm_data = normalize(reduced_embeddings, norm='l2')

            if clustering == 'AP':
                # Affinity Propagation clustering
                cluster_id = cluster_embeddings_with_affinity_propagation(embeddings)
                
            elif clustering == 'hdbscan':
                # HDBSCAN clustering
                cluster_id = hdbscan_model.fit_predict(reduced_embeddings)


            # UMAP for 2D visualization
            umap_vis = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, metric='cosine', random_state=42)
            vis_2d = umap_vis.fit_transform(embeddings)

            cluster_info = pd.DataFrame({
                'cluster': cluster_id,
                'Source': merged_text['Source'],
                'x': vis_2d[:, 0],
                'y': vis_2d[:, 1]
            })

            # Filter out noise points (cluster = -1)
            cluster_df = cluster_info[cluster_info['cluster'] != -1]
            embeddings_df = embeddings_df[cluster_info['cluster'] != -1].reset_index(drop=True)
            merged_text = merged_text[cluster_info['cluster'] != -1].reset_index(drop=True)

            # Ensure df_file is defined correctly
            df_file = merged_text.copy()

            # Generate the result DataFrame
            result_df = TFIDF_cluster_Naming(df_file, cluster_df, threshold=0)

            # Save clustering results
            cluster_df.to_csv(f'NewPipeline/clustering_outputs/{clustering}_{model}_merged_clustering.csv', index=False)
            print(f"\nBERTopic clustering for {model} saved.\n")

            # Save the embeddings to a CSV file
            embeddings_df.to_csv(f'NewPipeline/clustering_outputs/{clustering}_{model}_merged_embeddings.csv', index=False)
            print(f"\nBERTopic embedding for {model} saved.\n")

            # Generate and save the naming table
            merged_naming = naming_tableIndividual(result_df)

            merged_naming.to_csv(f'NewPipeline/clustering_outputs/{clustering}_{model}_merged_naming.csv', index=False)
            print(f"\nNaming table for {model} saved.\n")

            merged_naming_df = merged_naming_table(merged_naming)

            # Reorder the columns in the desired order
            column_order = ['cluster', 'pro', 'reg', 'sci', 'TF_IDF_topic_name', 'percentage_of_documents', 'percentage_limit', 'Topic_terms']
            merged_naming_df = merged_naming_df[column_order]

            merged_naming_df.to_csv(f'NewPipeline/clustering_outputs/{clustering}_{model}_merged_naming_clusters.csv', index=False)


            
            



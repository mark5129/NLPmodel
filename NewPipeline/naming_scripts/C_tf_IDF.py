import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def compute_top_c_tf_idf_terms(df_texts, df_clusters, content_col="Content", cluster_col="cluster", top_n=5):

    # Merge and group by cluster
    df = pd.concat([df_texts.reset_index(drop=True), df_clusters.reset_index(drop=True)], axis=1)
    docs_per_cluster = df.groupby(cluster_col)[content_col].apply(lambda texts: ' '.join(texts)).reset_index()

    # Fit vectorizer
    tf_vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(2, 3)
        )
    tfidf_matrix = tf_vectorizer.fit_transform(docs_per_cluster[content_col])
    tf = tfidf_matrix.toarray()

    # Compute IDF and c-TF-IDF
    df_counts = np.sum(tf > 0, axis=0)
    n_clusters = tf.shape[0]
    idf = np.log(n_clusters / (df_counts + 1e-10))
    c_tf_idf = tf * idf
    terms = tf_vectorizer.get_feature_names_out()

    # Extract top n terms per cluster
    top_terms = {}
    for i, cluster_id in enumerate(docs_per_cluster[cluster_col]):
        top_indices = np.argsort(c_tf_idf[i])[::-1][:top_n]
        top_words = []
        for idx in top_indices:
            term = terms[idx]
            score = round(c_tf_idf[i][idx], 4)
            count = sum(term in doc.lower() for doc in df[df[cluster_col] == cluster_id][content_col])
            doc_count = np.sum(tf[:, idx] > 0)  # Number of documents containing the term
            top_words.append((term, score, doc_count, count))
        top_terms[cluster_id] = top_words

    # Convert to DataFrame for better readability
    result_df = pd.DataFrame([
        {"cluster": cluster, **{f"term_{i+1}": f'{term};{idf};{d_count};{count}' for i, (term, idf, d_count, count) in enumerate(terms_with_scores)}}
        for cluster, terms_with_scores in top_terms.items()
    ])

    return result_df

# Example usage:
# top_terms_df = compute_top_c_tf_idf_terms(df_texts, df_clusters)
# print(top_terms_df)


models = ['Specter2']

sources = ['pro', 'reg', 'sci']

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

for model in models:


    df_clusters = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_clustering.csv')

    df_texts = merged_text

    # Combine 'Source' and 'cluster' to create unique cluster identifiers
    df_clusters['unique_cluster'] = df_clusters['Source'] + '_' + df_clusters['cluster'].astype(str)

    # Map unique cluster identifiers to new integers
    unique_cluster_mapping = {cluster: idx for idx, cluster in enumerate(df_clusters['unique_cluster'].unique())}
    df_clusters['new_cluster'] = df_clusters['unique_cluster'].map(unique_cluster_mapping)

    # Update the cluster column to use the new cluster integers
    df_clusters['cluster'] = df_clusters['new_cluster']
    df_clusters.drop(columns=['unique_cluster', 'new_cluster'], inplace=True)

    ctfidf_df = compute_top_c_tf_idf_terms(df_texts, df_clusters)
    
    # Save the DataFrame to a CSV file
    ctfidf_df.to_csv(f'NewPipeline/clustering_outputs/{model}_ctfidf.csv', index=False)

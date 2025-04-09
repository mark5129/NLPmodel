# load parameters from yaml file.
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def TFIDF_cluster_topic(cluster_texts, other_texts, language='english', n_terms=5):
    # Define stop words for different languages
    stop_words = {
        'english': stopwords.words('english')
    }
    cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
    other_texts = [text for text in other_texts if isinstance(text, str) and text.strip()]
    if not cluster_texts or not other_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=stop_words[language],
        max_features=1000,
        ngram_range=(2, 3)  # Use bigrams and trigrams
    )
    try:
        # Combine cluster texts and other texts
        all_texts = cluster_texts + other_texts
        X = vectorizer.fit_transform(all_texts)
        if X.shape[1] == 0:
            return []

        # Split the TF-IDF matrix into cluster and other documents
        cluster_matrix = X[:len(cluster_texts)]
        other_matrix = X[len(cluster_texts):]

        # Compute mean TF-IDF scores for cluster and other documents
        cluster_tf_idf = cluster_matrix.mean(axis=0).A1
        other_tf_idf = other_matrix.mean(axis=0).A1

        # Compute the difference to find terms distinguishing the cluster
        tf_idf_diff = cluster_tf_idf - other_tf_idf
        terms = vectorizer.get_feature_names_out()

        top_indices = tf_idf_diff.argsort()[::-1][:n_terms]
        top_terms = [terms[i] for i in top_indices if tf_idf_diff[i] > 0]

        return top_terms
    except Exception as e:
        print(f"Error in get_cluster_topic: {e}")
    return []

def TFIDF_cluster_Naming(df_file, result_df, clustering_method, source, model):
    # df_file: dataframe containing texts in column 'Content'
    # result_df: dataframe containing cluster integers in column 'cluster' 

    labels_int = result_df['cluster'].astype(int).values
    n_clusters = len(np.unique(labels_int))

    # Ensure indices alignment before assigning topic names
    result_df = result_df.reset_index(drop=True)

    # Generate topic names for each cluster
    for label in range(n_clusters):
        indices = np.where(labels_int == label)[0]
        other_indices = np.where(labels_int != label)[0]

        cluster_texts = df_file['Content'].iloc[indices].astype(str).tolist()
        other_texts = df_file['Content'].iloc[other_indices].astype(str).tolist()
        top_terms = TFIDF_cluster_topic(cluster_texts, other_texts, language='english', n_terms=5)

        # Select the 3 most occurring terms as the topic name
        How_many_terms = 3
        topic_name = None
        term_counts = pd.Series(top_terms).value_counts()
        top_occurring_terms = term_counts.index[:How_many_terms]
        
        if len(top_occurring_terms) > 0:
            topic_name = " ".join(top_occurring_terms)

        if topic_name is None:
            topic_name = f"No words found for cluster {label}"

        # Assign topic names using aligned indices
        result_df.loc[result_df.index[indices], 'topic_name'] = topic_name
    
    # Save results
    output_file = f'evaluations/outputs/manualrun_{model}_{source}_clusters_{clustering_method}.csv'

    result_df.to_csv(output_file, index=False)
    print(f"TFIDF cluster namning for {source} and {model} saved for {clustering_method}")
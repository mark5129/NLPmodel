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
from sklearn.preprocessing import normalize

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def TFIDF_cluster_topic(cluster_texts, other_texts,threshold, language='english', n_terms=20):
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

        # Normalize the matrices
        cluster_matrix = normalize(cluster_matrix, norm='l1', axis=1)
        other_matrix = normalize(other_matrix, norm='l1', axis=1)

        # Compute mean TF-IDF scores for cluster and other documents
        cluster_tf_idf = cluster_matrix.mean(axis=0).A1
        other_tf_idf = other_matrix.mean(axis=0).A1
        
        # Compute the difference to find terms distinguishing the cluster
        tf_idf_diff = cluster_tf_idf - other_tf_idf
        terms = vectorizer.get_feature_names_out()

        # Get top terms with positive differences
        top_indices = tf_idf_diff.argsort()[::-1][:n_terms]

        # Get top terms with positive differences above the threshold
        top_terms = [terms[i] for i in top_indices if tf_idf_diff[i] > threshold]
        top_values = [tf_idf_diff[i] for i in top_indices if tf_idf_diff[i] > threshold]

        # Get the distinct occurrence count of top_terms in cluster texts (case insensitive)
        term_counts = {term: sum(text.lower().count(term.lower()) for text in cluster_texts) for term in top_terms}

        # Get the distinct occurrence count of top_terms in cluster texts (case insensitive)
        document_counts = {term: sum(term.lower() in text.lower() for text in cluster_texts) for term in top_terms}

        # Concatenate top_terms, top_values, and term_counts into a single string
        top_terms_info = [
            f"{terms[i]}, {tf_idf_diff[i]:.4f}, {document_counts[terms[i]]}, {term_counts[terms[i]]}" 
            for i in top_indices if tf_idf_diff[i] > threshold
        ]
        top_terms = top_terms_info

        return top_terms
    except Exception as e:
        print(f"Error in TFIDF_cluster_topic: {e}")
    return []

def TFIDF_cluster_Naming(df_file, result_df, threshold):
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
        top_terms = TFIDF_cluster_topic(cluster_texts, other_texts, threshold)

        topic_name = None
        # Select the 5 most defining terms
        How_many_terms = 5
        
        if len(top_terms) > 0:
            try:
                top_terms = top_terms[:How_many_terms]
                topic_name = " | ".join(top_terms)
            except:
                topic_name = " | ".join(top_terms)

        if topic_name is None:
            topic_name = f"No words found for cluster {label}"

        # Assign topic names using aligned indices
        result_df.loc[result_df.index[indices], 'TF_IDF_topic_name'] = topic_name

    return result_df
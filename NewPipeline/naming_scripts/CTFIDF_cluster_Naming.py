# load parameters from yaml file.
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def CTFIDF_cluster_topic(cluster_texts, other_texts, threshold, language='english', n_terms=20):
    # Define stop words for different languages
    stop_words = {
        'english': stopwords.words('english')
    }
    cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
    other_texts = [text for text in other_texts if isinstance(text, str) and text.strip()]
    if not cluster_texts or not other_texts:
        return []

    vectorizer = CountVectorizer(
        stop_words=stop_words[language],
        max_features=1000,
        ngram_range=(2, 3)  # Use bigrams and trigrams
    )
    try:
        # Combine cluster texts into a single "document" for each cluster
        cluster_document = " ".join(cluster_texts)
        other_document = " ".join(other_texts)
        all_documents = [cluster_document, other_document]

        # Compute term frequencies
        X = vectorizer.fit_transform(all_documents)
        if X.shape[1] == 0:
            return []

        # Compute class-based TF-IDF
        term_frequencies = X.toarray()
        document_frequencies = np.sum(term_frequencies > 0, axis=0)
        idf = np.log((1 + len(all_documents)) / (1 + document_frequencies)) + 1
        c_tf_idf = term_frequencies[0] * idf  # Use the first document (cluster)

        terms = vectorizer.get_feature_names_out()

        # Get top terms with positive differences
        top_indices = c_tf_idf.argsort()[::-1][:n_terms]

        # Get top terms with positive differences above the threshold
        top_terms = [terms[i] for i in top_indices if c_tf_idf[i] > threshold]
        top_values = [c_tf_idf[i] for i in top_indices if c_tf_idf[i] > threshold]
        top_terms1 = top_terms

        # Get the distinct occurrence count of top_terms in cluster texts (case insensitive)
        term_counts = {term: sum(text.lower().count(term.lower()) for text in cluster_texts) for term in top_terms}

        # Get the distinct occurrence count of top_terms in cluster texts (case insensitive)
        document_counts = {term: sum(term.lower() in text.lower() for text in cluster_texts) for term in top_terms}

        # Concatenate top_terms, top_values, and term_counts into a single string
        top_terms_info = [
            f"{terms[i]}, {c_tf_idf[i]:.4f}, {document_counts[terms[i]]}, {term_counts[terms[i]]}" 
            for i in top_indices if c_tf_idf[i] > threshold
        ]
        top_terms = top_terms_info

        return top_terms, document_counts, top_terms1
    except Exception as e:
        print(f"Error in CTFIDF_cluster_topic: {e}")
    return []

def CTFIDF_cluster_Naming(df_file, result_df, threshold):
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
        top_terms, document_counts, top_terms1 = CTFIDF_cluster_topic(cluster_texts, other_texts, threshold)

        topic_name = None
        # Select the 5 most defining terms
        How_many_terms = 5
        
        if len(top_terms) > 0:
            try:
                top_terms = top_terms[:How_many_terms]
                topic_name = " ; ".join(top_terms)
                top_terms1 = top_terms1[:How_many_terms]
                topic_name1 = " ; ".join(top_terms1)
            except:
                topic_name = " ; ".join(top_terms)
                topic_name1 = " ; ".join(top_terms1)

        if topic_name is None:
            topic_name = f"No words found for cluster {label}"

        # Assign topic names using aligned indices
        result_df.loc[result_df.index[indices], 'CTF_IDF_topic_name'] = topic_name

        
        counts = list(document_counts.values())[:How_many_terms]
        weights = np.linspace(1, 1 / len(counts), len(counts))
        weighted_average = np.average(counts, weights=weights)

        percentage_of_documents = (weighted_average / len(indices))
        result_df.loc[result_df.index[indices], 'percentage_of_documents'] = round(percentage_of_documents, 4)
        result_df.loc[result_df.index[indices], 'percentage_limit'] = [int(1) if percentage_of_documents >= 0.2 else int(0)] * len(indices)

        # Assign topic names using aligned indices
        result_df.loc[result_df.index[indices], 'Topic_terms'] = topic_name1

    return result_df
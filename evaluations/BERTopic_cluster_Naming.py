# load parameters from yaml file.
import yaml
import numpy as np
import os
import pandas as pd
from nltk.corpus import stopwords
from bertopic import BERTopic

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def BERTopic_cluster_topic(cluster_texts, other_texts, n_terms=5):
    """
    Use BERTopic to extract representative terms for a group of cluster_texts.
    This does not compare to 'other_texts' but instead relies on topic modeling output.
    """
    cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
    if not cluster_texts:
        return []

    try:
        topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=False)
        topics, _ = topic_model.fit_transform(cluster_texts)

        # Get topic words for the most common topic in the sample
        topic_counts = pd.Series(topics).value_counts()
        dominant_topic = topic_counts.index[0]

        words = topic_model.get_topic(dominant_topic)
        top_terms = [term for term, _ in words[:n_terms]]

        return top_terms
    except Exception as e:
        print(f"Error in BERTopic_cluster_topic: {e}")
        return []

def BERTopic_cluster_Naming(df_file, result_df, clustering_method, source, model):
    # df_file: dataframe containing texts in column 'Content'
    # result_df: dataframe containing cluster integers in column 'cluster' 

    labels_int = result_df['cluster'].astype(int).values
    n_clusters = len(np.unique(labels_int))

    result_df = result_df.reset_index(drop=True)

    for label in range(n_clusters):
        indices = np.where(labels_int == label)[0]
        cluster_texts = df_file['Content'].iloc[indices].astype(str).tolist()
        top_terms = BERTopic_cluster_topic(cluster_texts, other_texts=None, n_terms=5)

        # How_many_terms = 3
        # topic_name = None
        # term_counts = pd.Series(top_terms).value_counts()
        # top_occurring_terms = term_counts.index[:How_many_terms]

        if len(top_terms) > 0:
            topic_name = " ".join(top_terms)
        else:
            topic_name = f"No words found for cluster {label}"

        result_df.loc[result_df.index[indices], 'BERTopic_topic_name'] = topic_name

    output_file = f'evaluations/outputs/manualrun_{model}_{source}_clusters_{clustering_method}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"BERTopic cluster naming for {source} and {model} saved for {clustering_method}")


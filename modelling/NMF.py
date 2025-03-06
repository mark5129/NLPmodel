import pandas as pd
import os
# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def NMF_model(documents, current_id, doc_type):
    # Convert the documents into a TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=parameters['max_features'], stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Fit the NMF model
    nmf_model = NMF(n_components=parameters['num_topics'], random_state=parameters['random_state'])
    nmf_model.fit(tfidf_matrix)

    # Save the topics to a CSV file
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-parameters['n_top_words'] - 1:-1]]
        topics.append({"Topic": topic_idx, "Words": " ".join(topic_words)})

    topics_df = pd.DataFrame(topics)

    # Save embeddings to CSV
    output_dir = 'outputs/NMF'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    topics_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_nmf_topics.csv'), index=False)
    print(f'NMF Topics saved for ID {current_id}')

    # Get the topic distribution for each document
    doc_topic_matrix = nmf_model.transform(tfidf_matrix)

    # Save the document-topic matrix to a CSV file
    doc_topic_df = pd.DataFrame(doc_topic_matrix)
    doc_topic_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_nmf_doc_topics.csv'), index=False)
    print(f'Document-Topic matrix saved for ID {current_id}')
# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import pandas as pd
import os


# Implement XML-Roberta topic modelling

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation

def XLM_Roberta_model(text_column, current_id, doc_type):

    # Load the XLM-Roberta model
    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    # Generate embeddings
    embeddings = model.encode(text_column, show_progress_bar=True, batch_size=32)

    # Convert the embeddings to a DataFrame
    embeddings_df = pd.DataFrame(embeddings)

    # Fit the LDA model
    lda_model = LatentDirichletAllocation(n_components=parameters['num_topics'], random_state=parameters['random_state'])
    lda_model.fit(embeddings_df)

    # Save the topics to a CSV file
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        topic_words = [str(i) for i in topic.argsort()[:-parameters['n_top_words'] - 1:-1]]
        topics.append({"Topic": topic_idx, "Words": " ".join(topic_words)})

    topics_df = pd.DataFrame(topics)

    # Save the topics to a CSV file
    output_dir = 'outputs/XLM_Roberta_topics'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    topics_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_XLM_Roberta_topics.csv'), index=False)
    print('XLM-Roberta Topics saved')

    # Get the topic distribution for each document
    doc_topic_matrix = lda_model.transform(embeddings_df)

    # Save the document-topic matrix to a CSV file
    doc_topic_df = pd.DataFrame(doc_topic_matrix)
    doc_topic_df.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_XLM_Roberta_doc_topics.csv'), index=False)
    print('Document-Topic matrix saved')

    return doc_topic_matrix, lda_model


reg_media = pd.read_csv(parameters['reg_media_stemmed_dir'])
pro_media = pd.read_csv(parameters['pro_media_stemmed_dir'])

reg_text_column = reg_media['Content']
pro_text_column = pro_media['Full text']

XLM_Roberta_model(reg_text_column, '101', 'reg')
XLM_Roberta_model(pro_text_column, '101', 'pro')
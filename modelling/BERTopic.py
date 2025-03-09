# load parameters from yaml file.
import pandas as pd
import os
from bertopic import BERTopic
import yaml
import numpy as np
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import pandas as pd

def BERTopicModel(text_column, current_id, doc_type):
    """
    Analyzes the text data and returns the topics and the trained BERTopic model.

    Args:
    text_column (list of str): The text data to analyze.

    Returns:
    topics (list): A list of topic assignments for each document.
    topic_model (BERTopic): The trained BERTopic model.
    """

    # Initialize and fit BERTopic
    topic_model = BERTopic(nr_topics="auto", min_topic_size=5) 
    topics, probs = topic_model.fit_transform(text_column)

    output_dir = "modelling/outputs/BERTopic"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_topics = pd.DataFrame({
        "Topic": topics,
        "Probability": probs
    })

    df_topics.to_csv(os.path.join(output_dir, f"{doc_type}_BERTopic_results.csv"), index=False)
    print(f"BERTopic results saved: {doc_type}_BERTopic_results.csv")

    if topic_model.embedding_model:
        embeddings = topic_model.embedding_model.embed(text_column.tolist())
        # make embeddings to pd df
        df_embeddings = pd.DataFrame(embeddings)
        # save embeddings
        df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_BERTopic_embeddings.csv'))
        print(f"✅ Saved embeddings for {doc_type}!")

    # ✅ Save the trained model
    topic_model.save(os.path.join(output_dir, f"{doc_type}_BERTopic_model.pkl"))
    print(f"✅ Saved BERTopic model for {doc_type}!")

    return topics, topic_model

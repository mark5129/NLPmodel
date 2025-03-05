# load parameters from yaml file.
import pandas as pd
import os
from bertopic import BERTopic
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import pandas as pd
reg_media = pd.read_csv(parameters['reg_media_stemmed_dir'])
pro_media = pd.read_csv(parameters['pro_media_stemmed_dir'])

print("test0")

from bertopic import BERTopic

def BERTopicModel(text_column, doc_type):
    """
    Analyzes the text data and returns the topics and the trained BERTopic model.

    Args:
    text_column (list of str): The text data to analyze.

    Returns:
    topics (list): A list of topic assignments for each document.
    topic_model (BERTopic): The trained BERTopic model.
    """
    
    # Initialize and fit BERTopic
    topic_model = BERTopic(nr_topics=parameters['rn_topics'])
    topics, probs = topic_model.fit_transform(text_column)

    print("test1")
    output_dir = "outputs/BERTopic"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_topics = pd.DataFrame({
        "Topic": topics,
        "Probability": probs
    })

    df_topics.to_csv(os.path.join(output_dir, f"{doc_type}_BERTopic_results.csv"), index=False)
    print(f"BERTopic results saved: {doc_type}_BERTopic_results.csv")

    return topics, topic_model

print("test2")

reg_text_column = reg_media['Content']
pro_text_column = pro_media['Full text']

print("test3")

# Capture the output from the function
reg_topics, reg_topic_model = BERTopicModel(reg_text_column, "reg_media")
pro_topics, pro_topic_model = BERTopicModel(pro_text_column, "pro_media")

print("test4")

# Print the output
print("Regular Media Topics:", reg_topics)
print("Regular Media Topics:", reg_topic_model)
print("Professional Media Topics:", pro_topics)
print("Professional Media Topics:", pro_topic_model)
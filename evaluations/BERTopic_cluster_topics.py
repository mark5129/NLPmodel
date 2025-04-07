import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
from umap import UMAP
import os
import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# def BERTopic_cluster_topic(cluster_texts, language='english', n_terms=5):
#     # Define stop words for different languages
#     stop_words = {
#         'english': stopwords.words('english')
#     }
    
#     cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
#     if not cluster_texts:
#         return []

#     try:
#         # Initialize BERTopic
#         topic_model = BERTopic(language=language, nr_topics=parameters['num_topics'])
#         topics, _ = topic_model.fit_transform(cluster_texts)

#         # Get the top terms for each topic
#         topic_info = topic_model.get_topic_info()
        
#         # For each topic, retrieve the top terms
#         top_terms = []
#         for topic in topic_info['Topic']:
#             if topic == -1:  # Skip outliers
#                 continue
#             terms = topic_model.get_topic(topic)[:n_terms]  # Get the top n_terms from the topic
#             # Extract the term names (words) from the tuples
#             topic_terms = [term[0] for term in terms]  # term[0] gives the word itself
#             top_terms.extend(topic_terms)  # Add them to the list

#         # Ensure to return only unique terms and limit to n_terms
#         top_terms = list(dict.fromkeys(top_terms))  # Remove duplicates
#         return top_terms[:n_terms]  # Return only the first n_terms (up to 5 terms)

#     except Exception as e:
#         print(f"Error in get_cluster_topic_with_bertopic: {e}")
#     return []


def BERTopic_cluster_topic(cluster_texts, nr_words):
    """
    Given a list of strings (texts), fit a BERTopic model and return
    a dictionary where each key is a topic ID and each value is
    a list of the top words for that topic.

    :param texts: List[str]
        The input documents to be clustered and modeled by BERTopic.
    :param nr_words: int
        How many words per topic to retrieve. Defaults to 10.
    :return: Dict[int, List[str]]
        A dictionary mapping topic IDs to lists of words.
    """
    from bertopic import BERTopic

    # 1. Create and fit the BERTopic model
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(cluster_texts)

    # 2. Retrieve all unique topic labels (excluding outliers labeled as -1)
    unique_topics = sorted(set(topics) - {-1})

    # 3. For each topic, retrieve its most important words
    top_terms = {}
    for topic_id in unique_topics:
        # get_topic() returns a list of (word, score) tuples
        words_with_scores = topic_model.get_topic(topic_id)
        # Slice to top-n words if desired
        words_only = [word for word, _ in words_with_scores[:nr_words]]
        top_terms[topic_id] = words_only

    return top_terms



def Bertopic_clustering_naming(embeddings, df, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]

    # Adjust perplexity based on the number of samples
    perplexity = min(30, (n_samples - 1) // 3)

    # Step 1: Create a data map using t-SNE
    #tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    #data_map = tsne.fit_transform(embeddings)
    
    umapmodel = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    data_map = umapmodel.fit_transform(embeddings)

    # Step 2: Perform hierarchical clustering
    n_clusters_list = [parameters['num_topics']]  # Adjust these numbers for your desired hierarchy levels
    labels_layers = []
    topics_names = pd.DataFrame(columns=['topic_int', 'topic_names'])

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=parameters['random_state'])
        labels_int = kmeans.fit_predict(data_map)

        used_topics = set()

        # Generate topic names for each cluster
        label_topic_map = {}
        for label in range(n_clusters):
            indices = np.where(labels_int == label)[0]
            if len(indices) == 0:
                label_topic_map[label] = f"{label}: No data"
                continue
            cluster_texts = df['Content'].iloc[indices].astype(str).tolist()
            top_terms = BERTopic_cluster_topic(cluster_texts, parameters['nr_words'])

            # We pick the first subtopic from this dict and join its words into a single string.
            if top_terms:
                # Grab the first subtopic ID
                first_subtopic_id = sorted(top_terms.keys())[0]
                # Get the words for that subtopic
                words_list = top_terms[first_subtopic_id]
                # Join them with commas (or spaces, if you prefer)
                topic_name = ", ".join(words_list)

                # Optional check if we've used that exact phrase before
                if topic_name in used_topics:
                    topic_name = f"{topic_name} (cluster_{label})"
                used_topics.add(topic_name)
                
            else:
                # If no subtopics were found by BERTopic (rare), fallback
                topic_name = f"Cluster {label}"


            label_topic_map[label] = f"{label}: {topic_name}"

            # Save topic names for each cluster
            topics_names.loc[label, 'topic_int'] = label
            topics_names.loc[label, 'topic_names'] = topic_name

        # Convert integer labels to topic names
        labels_topic = np.array([label_topic_map.get(label, f"{label}: Unknown") for label in labels_int])
        labels_layers.append(labels_topic)

    # Save embeddings to CSV
    output_dir = 'evaluations/outputs/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_labels_int = pd.DataFrame(labels_int)
    df_labels_int.columns = ['topic_int']

    df_labels_int = df_labels_int.merge(topics_names, on='topic_int', how='left')

    sources = df['Source']

    df_labels_int['Source'] = sources
    
    df_labels_int.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_{model_name}_Kmeans.csv'), index=False)
    print(f'{model_name} k-means clustering saved for ID {current_id}')
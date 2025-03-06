import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import datamapplot
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

def get_cluster_topic(cluster_texts, language='english', n_terms=5):
    # Define stop words for different languages
    stop_words = {
        'english': stopwords.words('english')
    }
    cluster_texts = [text for text in cluster_texts if isinstance(text, str) and text.strip()]
    if not cluster_texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words=stop_words[language],
        max_features=1000,
        ngram_range=(2, 3)  # Use bigrams and trigrams
    )
    try:
        X = vectorizer.fit_transform(cluster_texts)
        if X.shape[1] == 0:
            return []

        tf_idf_sum = X.sum(axis=0).A1  # Sum TF-IDF scores across all documents
        terms = vectorizer.get_feature_names_out()

        top_indices = tf_idf_sum.argsort()[::-1]
        top_terms = [terms[i] for i in top_indices]

        return top_terms
    except Exception as e:
        print(f"Error in get_cluster_topic: {e}")
    return []
        

def data_mapplot_with_naming(embeddings, df, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]

    # Adjust perplexity based on the number of samples
    perplexity = min(30, (n_samples - 1) // 3)

    # Step 1: Create a data map using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    data_map = tsne.fit_transform(embeddings)

    # Step 2: Perform hierarchical clustering
    n_clusters_list = [10]  # Adjust these numbers for your desired hierarchy levels
    labels_layers = []

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
            top_terms = get_cluster_topic(cluster_texts, language='english', n_terms=5)

            # Select the first unused term as the topic name
            topic_name = None
            for term in top_terms:
                if term not in used_topics:
                    topic_name = term
                    used_topics.add(term)
                    break

            if topic_name is None:
                # All terms have been used; default to the highest scoring term with cluster label
                topic_name = f"{top_terms[0]} {label}" if top_terms else f"Cluster {label}"

            label_topic_map[label] = f"{label}: {topic_name}"

        # Convert integer labels to topic names
        labels_topic = np.array([label_topic_map.get(label, f"{label}: Unknown") for label in labels_int])
        labels_layers.append(labels_topic)

    # Step 3: Prepare hover text
    hover_text = df['Title'].astype(str).tolist()

    # Step 4: Create a color palette
    color_palette = list(mcolors.TABLEAU_COLORS.values())

    # Step 5: Generate marker colors using the last layer of labels
    labels = labels_layers[-1]  # Use the last layer for coloring

    # Create a color mapping
    unique_labels = np.unique(labels)
    color_mapping = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

    # Generate marker colors
    marker_color_array = [color_mapping[label] for label in labels]

    # Step 6: Set marker sizes
    marker_size_array = df['Content'].str.len().values.astype(np.float32)
    min_size, max_size = 5, 15
    # Normalize marker sizes between min_size and max_size
    if marker_size_array.max() != marker_size_array.min():
        marker_size_array = min_size + (max_size - min_size) * (
            (marker_size_array - marker_size_array.min()) / (marker_size_array.max() - marker_size_array.min())
        )
    else:
        marker_size_array = np.full_like(marker_size_array, (min_size + max_size) / 2)

    # Step 7: Set point radius min and max pixels
    point_radius_min_pixels = 2
    point_radius_max_pixels = 10

    # Create the interactive plot
    try:
        plot = datamapplot.create_interactive_plot(
            data_map,
            *labels_layers,  # Use the labels with topic names
            hover_text=hover_text,
            font_family="Merriweather",
            title=f"{model_name}",
            sub_title=f"{model_name} Interactive plot",
            enable_search=True,
            darkmode=True,
            marker_color_array=marker_color_array,
            marker_size_array=marker_size_array,
            point_radius_min_pixels=point_radius_min_pixels,
            point_radius_max_pixels=point_radius_max_pixels,
            point_line_width=0,
            cluster_boundary_polygons=False,  # Disable if not needed
            cluster_boundary_line_width=2,
        )
        # Save Visualization to CSV
        output_dir = 'visualizations/outputs/'

        # Save the plot to an HTML file
        plot.save(f"{output_dir}{current_id}_{doc_type}_{model_name}_datamapplot.html")

        print("Plot saved successfully.")
    except Exception as e:
        print(f"Error creating or displaying the plot: {e}")


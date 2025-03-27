import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import datamapplot
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def data_mapplot_with_naming(k_means,embeddings, df, current_id, doc_type, model_name):

    # Ensure embeddings are in NumPy array format
    embeddings = np.array(embeddings)

    # Number of samples
    n_samples = embeddings.shape[0]

    # Adjust perplexity based on the number of samples
    perplexity = min(30, (n_samples - 1) // 3)

    # Step 1: Create a data map using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=parameters['random_state'])
    data_map = tsne.fit_transform(embeddings)

    # Step 3: Prepare hover text
    hover_text = df['Title'].astype(str).tolist()

    # Step 4: Create a color palette
    color_palette = list(mcolors.TABLEAU_COLORS.values())

    
    # Step 5: Generate marker colors using the last layer of labels
    labels = k_means['topic_int']
    labels_layers = []
    labels_layers = [list(k_means['labels_layer'])]  # Convert to a list

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
            sub_title=f"Interactive plot",
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

        print(f"{model_name} with {doc_type}: Data map plot saved successfully for {current_id}")
    except Exception as e:
        print(f"Error creating or displaying the plot: {e}")


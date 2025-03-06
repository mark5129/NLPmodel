import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import datamapplot
import matplotlib.colors as mcolors
import pandas as pd

# Load the embeddings from csv file
embeddings = pd.read_csv('outputs/MiniLm12/2583467673_reg_MiniLm12_embeddings.csv')
# Ensure embeddings are in NumPy array format
embeddings = np.array(embeddings)

df = pd.read_csv('data/reg_media_stemmed_eng.csv')

# Number of clusters
n_clusters = 10  # You can adjust this as needed

# Step 1: Create a data map using t-SNE
n_samples = embeddings.shape[0]
perplexity = min(30, (n_samples - 1) // 3)

tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
data_map = tsne.fit_transform(embeddings)

# Step 2: Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_int = kmeans.fit_predict(data_map)

# Assign generic cluster names
label_topic_map = {label: f"Cluster {label + 1}" for label in range(n_clusters)}
labels_topic = np.array([label_topic_map.get(label, "Unknown") for label in labels_int])

# Step 3: Prepare hover text
hover_text = df['Content'].astype(str).tolist()

# Step 4: Create a color palette
color_palette = list(mcolors.TABLEAU_COLORS.values())

# Step 5: Generate marker colors
unique_labels = np.unique(labels_topic)
color_mapping = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}
marker_color_array = [color_mapping[label] for label in labels_topic]

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
        labels_topic,  # Use the generic cluster names
        hover_text=hover_text,
        font_family="Merriweather",
        title="Interviews",
        sub_title="Interactive plot with Generic Cluster Names",
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
    output_dir = 'Visualizations/outputs/'

    # Save the plot to an HTML file
    plot.save(f"{output_dir}Interviews: Clusters_Generic_Names.html")
    print("Plot with generic cluster names saved successfully.")
except Exception as e:
    print(f"Error creating or saving the plot: {e}")

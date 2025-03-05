import numpy as np
import pandas as pd
import json
import datamapplot
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import torch
import time
import yaml
from sentence_transformers import SentenceTransformer

# ✅ Load parameters
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# ✅ Load dataset
df = pd.read_csv(parameters['reg_media_stemmed_dir'])
text_data = df['Content'].tolist()

# ✅ Load SPECTER2 model
model_name = "allenai/specter"
model = SentenceTransformer(model_name)

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Generate embeddings with progress tracking
start_time = time.time()
embeddings = model.encode(text_data, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"✅ Embeddings generated in {elapsed_time:.2f} seconds.")

# ✅ Ensure embeddings are in NumPy array format
embeddings = np.array(embeddings)

# ✅ Number of clusters
n_clusters = 10  # Adjust as needed

# ✅ Step 1: Create a data map using t-SNE
n_samples = embeddings.shape[0]
perplexity = min(30, (n_samples - 1) // 3)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
data_map = tsne.fit_transform(embeddings)

# ✅ Step 2: Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_int = kmeans.fit_predict(data_map)

# ✅ Step 3: Prompt user to define cluster names
def get_user_defined_names(n_clusters):
    """Prompt user to input names for each cluster."""
    print(f"\nPlease provide names for each of the {n_clusters} clusters.")
    cluster_names = []
    for i in range(n_clusters):
        user_input = input(f"Enter name for Cluster {i + 1}: ").strip()
        if user_input:
            cluster_names.append(user_input)
        else:
            cluster_names.append(f"Cluster {i + 1}")
    return cluster_names

cluster_names = get_user_defined_names(n_clusters)

# ✅ Step 4: Assign cluster names
label_topic_map = {label: cluster_names[label] for label in range(n_clusters)}
labels_topic = np.array([label_topic_map.get(label, "Unknown") for label in labels_int])

# ✅ Step 5: Save cluster names for future use
def save_cluster_names(cluster_names, filename="cluster_names.json"):
    with open(filename, 'w') as f:
        json.dump(cluster_names, f, indent=4)
    print(f"\n✅ Cluster names saved to {filename}.")

save_cluster_names(cluster_names)

# ✅ Step 6: Prepare hover text
hover_text = df['Content'].astype(str).tolist()

# ✅ Step 7: Create a color palette
color_palette = list(mcolors.TABLEAU_COLORS.values())

# ✅ Step 8: Generate marker colors
unique_labels = np.unique(labels_topic)
color_mapping = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}
marker_color_array = [color_mapping[label] for label in labels_topic]

# ✅ Step 9: Set marker sizes
marker_size_array = df['Content'].str.len().values.astype(np.float32)
min_size, max_size = 5, 15

# Normalize marker sizes between min_size and max_size
if marker_size_array.max() != marker_size_array.min():
    marker_size_array = min_size + (max_size - min_size) * (
        (marker_size_array - marker_size_array.min()) / (marker_size_array.max() - marker_size_array.min())
    )
else:
    marker_size_array = np.full_like(marker_size_array, (min_size + max_size) / 2)

# ✅ Step 10: Set point radius min and max pixels
point_radius_min_pixels = 2
point_radius_max_pixels = 10

# ✅ Step 11: Create the interactive plot
try:
    plot = datamapplot.create_interactive_plot(
        data_map,
        labels_topic,  # Use the user-defined cluster names
        hover_text=hover_text,
        font_family="Merriweather",
        title="SPECTER2 Clusters (t-SNE)",
        sub_title="Interactive plot with User-Defined Cluster Names",
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

    # ✅ Save the plot
    plot_filename = "outputs/Specter2_op/SPECTER2_tSNE_User_Clusters.html"
    plot.save(plot_filename)
    print(f"✅ Plot with user-defined cluster names saved successfully as {plot_filename}.")

except Exception as e:
    print(f"❌ Error creating or saving the plot: {e}")

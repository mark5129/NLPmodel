import numpy as np
import pandas as pd
import datamapplot
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import torch
import time
import yaml
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Load parameters
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Load dataset
df = pd.read_csv(parameters['reg_media_stemmed_dir'])
text_data = df['Content'].tolist()

# Load SPECTER2 model & tokenizer
model_name = "allenai/specter"
from sentence_transformers import SentenceTransformer

# ✅ Use the correct SPECTER2 model from Hugging Face
model = SentenceTransformer("allenai/specter")

# Sample text
text = ["This is an example document for SPECTER2 embeddings."]

# ✅ Generate embeddings
embeddings = model.encode(text)
print("✅ Embeddings shape:", embeddings.shape)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate embeddings with progress tracking
start_time = time.time()


# ✅ Generate embeddings for all documents
embeddings = model.encode(text_data, batch_size=32, show_progress_bar=True, convert_to_numpy=True)



end_time = time.time()
elapsed_time = end_time - start_time

print(f"✅ Embeddings generated in {elapsed_time:.2f} seconds.")

# Convert embeddings list to NumPy array
embeddings = np.vstack(embeddings)

# Number of clusters
n_clusters = 10

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
# Normalize marker sizes
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
        labels_topic,
        hover_text=hover_text,
        font_family="Merriweather",
        title="Specter Clusters (t-SNE with SPECTER2)",
        sub_title="Interactive plot with Generic Cluster Names",
        enable_search=True,
        darkmode=True,
        marker_color_array=marker_color_array,
        marker_size_array=marker_size_array,
        point_radius_min_pixels=point_radius_min_pixels,
        point_radius_max_pixels=point_radius_max_pixels,
        point_line_width=0,
        cluster_boundary_polygons=False,
        cluster_boundary_line_width=2,
    )

    # Save Visualization to CSV
    output_dir = 'Visualizations/outputs/'

    # Save the plot
    plot.save(f"{output_dir}pro_media_SPECTER2_tSNE_Clusters.html")
    print("✅ Plot with SPECTER2 embeddings saved successfully.")
except Exception as e:
    print(f"❌ Error creating or saving the plot: {e}")

import numpy as np
import pandas as pd
from bertopic import BERTopic
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the trained BERTopic model
topic_model = BERTopic.load("outputs/BERTopic/pro_media_BERTopic_model.pkl")

# ✅ Load topic assignments
df = pd.read_csv("outputs/BERTopic/pro_media_BERTopic_results.csv")

# ✅ Load document embeddings
try:
    embeddings = np.load("outputs/BERTopic/pro_media_BERTopic_embeddings.npy")
    print("✅ Loaded precomputed embeddings.")
except FileNotFoundError:
    print("❌ No precomputed embeddings found.")
    exit()

# ✅ Reduce embeddings using UMAP
umap_model = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# ✅ Plot topic clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df["Topic"], palette="tab10")
plt.title("BERTopic Topic Clusters (2D UMAP)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Topics")

# Save embeddings to CSV
output_dir = 'Visualizations/outputs/'

# save plt figure as image

plt.savefig(f"{output_dir}pro_media_BERTopic_clusters.png")
print(f"✅ Saved BERTopic clusters plot as pro_media_BERTopic_clusters.png")


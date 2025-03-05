from bertopic import BERTopic
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ✅ Load BERTopic model correctly
topic_model = BERTopic.load("outputs/BERTopic/pro_media_BERTopic_model.pkl")

# ✅ Load topic assignments
df = pd.read_csv("outputs/BERTopic/pro_media_BERTopic_results.csv")

# ✅ Check topic distribution
topic_counts = df["Topic"].value_counts()
print(topic_counts)

# ✅ Try getting document embeddings from BERTopic
if hasattr(topic_model, "embedding_model") and topic_model.embedding_model is not None:
    print("✅ Using document embeddings...")
    embeddings = topic_model.transform(df["Topic"].astype(str))[1]  # Get document embeddings
elif hasattr(topic_model, "topic_embeddings_") and topic_model.topic_embeddings_ is not None:
    print("⚠️ No document embeddings found. Using topic embeddings...")
    embeddings = topic_model.topic_embeddings_
else:
    print("❌ No valid embeddings found. Exiting.")
    exit()

# ✅ Convert embeddings to a NumPy array and check shape
embeddings = np.array(embeddings)

if embeddings.ndim == 1:  # If it's 1D, reshape it
    print("⚠️ Reshaping 1D embeddings to 2D...")
    embeddings = embeddings.reshape(-1, 1)

print(f"Embedding shape: {embeddings.shape}")  # Debugging

# ✅ Apply UMAP
umap_model = umap.UMAP(n_neighbors=3, n_components=2, min_dist=0.3, random_state=42, metric='cosine')
reduced_embeddings = umap_model.fit_transform(embeddings)

# ✅ Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df["Topic"], palette="tab10")
plt.title("BERTopic Topic Clusters (2D UMAP)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Topics")
plt.show()

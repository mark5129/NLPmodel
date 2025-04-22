import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === KONFIGURASJON ===
EMBED_PATH = "NewPipeline/clustering_outputs/Specter2_merged_embeddings.csv"

# === Last inn Specter2-embeddings ===
df = pd.read_csv(EMBED_PATH)

# === Fjern eventuelle ikke-numeriske kolonner (index, ID osv.)
vecs = df.select_dtypes(include=[np.number]).values

# === Beregn cosine similarity mellom alle tekster
cosine_matrix = cosine_similarity(vecs)

# === Hent bare øvre trekant for å unngå dobbelttelling og diagonalen
upper = np.triu_indices_from(cosine_matrix, k=1)
average_cosine = cosine_matrix[upper].mean()

# === Print resultat
print("\n=== GJENNOMSNITTLIG COSINE SIMILARITY FOR ALLE TEKSTER (Specter2) ===\n")
print(f"📊 Average cosine similarity: {round(average_cosine, 4)}")

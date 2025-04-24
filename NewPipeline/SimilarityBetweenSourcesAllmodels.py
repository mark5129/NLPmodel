import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# === KONFIGURASJON ===
MODELS = ["Specter2", "MiniLm12", "XLM_Roberta"]
BASE_PATH = "NewPipeline/clustering_outputs"
RESULTS_PATH = "NewPipeline/results"
SOURCES = ["sci", "pro", "reg"]

# Sørg for at resultatsmappen eksisterer
os.makedirs(RESULTS_PATH, exist_ok=True)

results = []

for model in MODELS:
    print(f"\n🔍 Behandler modell: {model}")

    # === Last inn embeddings og clustering
    embed_path = f"{BASE_PATH}/hdbscan_{model}_merged_embeddings.csv"
    cluster_path = f"{BASE_PATH}/hdbscan_{model}_merged_clustering.csv"

    embeddings = pd.read_csv(embed_path)
    clustering = pd.read_csv(cluster_path)

    embeddings["index"] = embeddings.index
    clustering["index"] = clustering.index

    merged = pd.merge(clustering[["index", "Source"]], embeddings, on="index")

    for source in SOURCES:
        df = merged[merged["Source"] == source]
        if len(df) < 2:
            continue

        vecs = df.select_dtypes(include=[np.number]).drop(columns=["index"]).values
        cosine_matrix = cosine_similarity(vecs)
        upper = np.triu_indices_from(cosine_matrix, k=1)
        avg_sim = cosine_matrix[upper].mean()

        results.append({
            "Model": model,
            "Source": source,
            "Size": len(df),
            "Intra-source Cosine Similarity": round(avg_sim, 4)
        })

# === Lag og print tabell
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Model", "Source"])

print("\n=== INTRA-SOURCE COSINE SIMILARITY PER MODEL ===\n")
print(results_df.to_string(index=False))

# === LaTeX-tabell (valgfritt)
latex_table = "\\begin{tabular}{l l r r}\n\\toprule\n"
latex_table += "Model & Source & Size & Intra-source Cosine Similarity \\\\\n\\midrule\n"
for row in results:
    latex_table += f"{row['Model']} & {row['Source']} & {row['Size']} & {row['Intra-source Cosine Similarity']} \\\\\n"
latex_table += "\\bottomrule\n\\end{tabular}"

print("\n\n=== LATEX-TABELL ===\n")
print(latex_table)

# === Lagre resultater som CSV
results_df.to_csv(f"{RESULTS_PATH}/intra_source_similarity_results.csv", index=False)

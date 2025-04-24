import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# === KONFIGURASJON ===
MODELS = ["Specter2", "MiniLm12", "XLM_Roberta"]
BASE_PATH = "NewPipeline/clustering_outputs"
RESULTS_PATH = "NewPipeline/results"
SOURCE_PAIRS = [("sci", "pro"), ("sci", "reg"), ("pro", "reg")]

# Sørg for at resultatsmappen eksisterer
os.makedirs(RESULTS_PATH, exist_ok=True)

all_results = []

for model in MODELS:
    print(f"\n🔍 Behandler modell: {model}")

    # === Last inn embeddings og clustering ===
    embed_path = f"{BASE_PATH}/hdbscan_{model}_merged_embeddings.csv"
    cluster_path = f"{BASE_PATH}/hdbscan_{model}_merged_clustering.csv"

    embeddings = pd.read_csv(embed_path)
    clustering = pd.read_csv(cluster_path)

    # === Legg til index for sammenslåing
    embeddings["index"] = embeddings.index
    clustering["index"] = clustering.index

    # === Merge for å få Source-kolonne inn i embeddings
    merged = pd.merge(clustering[["index", "Source"]], embeddings, on="index")

    # === Gjør klar for cosine-beregning
    for source_a, source_b in SOURCE_PAIRS:
        df_a = merged[merged["Source"] == source_a]
        df_b = merged[merged["Source"] == source_b]

        N = min(len(df_a), len(df_b))
        if N < 2:
            continue

        # Sample likt antall fra begge
        sample_a = df_a.sample(n=N, random_state=42)
        sample_b = df_b.sample(n=N, random_state=42)

        vecs_a = sample_a.select_dtypes(include=[np.number]).drop(columns=["index"]).values
        vecs_b = sample_b.select_dtypes(include=[np.number]).drop(columns=["index"]).values

        # Beregn cosine similarity mellom parene
        sims = [cosine_similarity([vecs_a[i]], [vecs_b[i]])[0][0] for i in range(N)]
        avg_sim = np.mean(sims)

        all_results.append({
            "Model": model,
            "Pair": f"{source_a}-{source_b}",
            "Sample Size": N,
            "Average Cosine Similarity": round(avg_sim, 4)
        })

# === Lag resultat-tabell
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=["Model", "Pair"])

print("\n=== COSINE SIMILARITY MELLOM KILDER – MERGED EMBEDDINGS ===\n")
print(results_df.to_string(index=False))

# === LaTeX-tabell (valgfritt)
latex_table = "\\begin{tabular}{l l r}\n\\toprule\n"
latex_table += "Model & Source Pair & Average Cosine Similarity \\\\\n\\midrule\n"
for row in all_results:
    latex_table += f"{row['Model']} & {row['Pair']} & {row['Average Cosine Similarity']} \\\\\n"
latex_table += "\\bottomrule\n\\end{tabular}"

print("\n\n=== LATEX-TABELL ===\n")
print(latex_table)

# === Lagre resultater som CSV
results_df.to_csv(f"{RESULTS_PATH}/cosine_similarity_results.csv", index=False)

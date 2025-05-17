# evaluations/cosine_compare_all_models.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# === KONFIGURASJON ===
BASE_PATH = "modelling/outputs"
MODELS = ["Specter2Actually", "MiniLm12", "XLM_Roberta"]
FILES = {
    "sci": "1303156299_sci_{}_embeddings.csv",
    "pro": "1303156299_pro_{}_embeddings.csv",
    "reg": "1303156299_reg_{}_embeddings.csv"
}
N_EMBEDDING_DIM = 768

all_results = []

# === LOOP OVER MODELLER ===
for model in MODELS:
    print(f"\n🔍 Behandler modell: {model}")
    
    dfs = {}
    for label, template in FILES.items():
        path = os.path.join(BASE_PATH, model, template.format(model))
        if not os.path.exists(path):
            print(f"⚠️  Fil ikke funnet: {path}")
            continue
        df = pd.read_csv(path)
        dfs[label] = df

    if not all(k in dfs for k in ["sci", "pro", "reg"]):
        print(f"❌ Hopper over {model} – mangler data.")
        continue

    # Finn minste tilgjengelige antall dokumenter
    min_docs = min(len(df) for df in dfs.values())
    print(f"📊 Bruker {min_docs} dokumenter per kilde.")

    embeddings = {}
    for label, df in dfs.items():
        sample_df = df.sample(n=min_docs, random_state=42)
        embeddings[label] = sample_df.iloc[:, -N_EMBEDDING_DIM:].values

    # Definér par-kombinasjoner
    pairs = [("sci", "pro"), ("sci", "reg"), ("pro", "reg")]

    # Beregn cosine similarities med batch-matrise
    for label_a, label_b in pairs:
        emb_a = embeddings[label_a]
        emb_b = embeddings[label_b]

        sim_matrix = cosine_similarity(emb_a, emb_b)
        avg_sim = sim_matrix.mean()

        all_results.append({
            "Model": model,
            "Pair": f"{label_a}-{label_b}",
            "Average Cosine Similarity": round(avg_sim, 4)
        })

# === SKRIV UT RESULTATER ===
print("\n=== GJENNOMSNITTLIG COSINE SIMILARITY PER MODELL & PAR ===\n")
for row in all_results:
    print(f"{row['Model']:16s} | {row['Pair']:20s} | Similarity: {row['Average Cosine Similarity']}")

# === GENERÉR LATEX-TABELL ===
latex_table = "\\begin{tabular}{l l r}\n\\toprule\n"
latex_table += "Model & Pair & Average Cosine Similarity \\\\\n\\midrule\n"
for row in all_results:
    latex_table += f"{row['Model']} & {row['Pair']} & {row['Average Cosine Similarity']} \\\\\n"
latex_table += "\\bottomrule\n\\end{tabular}"

print("\n\n=== LATEX-TABELL ===\n")
print(latex_table)

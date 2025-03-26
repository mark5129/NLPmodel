import os
import pandas as pd
from sklearn.metrics import silhouette_score

def evaluate_silhouette_scores(base_dir="evaluations/outputs", suffix="_Kmeans.csv"):
    results = []

    for filename in os.listdir(base_dir):
        if filename.endswith("_embeddings_MiniLm12_Kmeans.csv") or \
           filename.endswith("_embeddings_Specter2Actually_Kmeans.csv") or \
           filename.endswith("_embeddings_XLM_Roberta_Kmeans.csv"):
            continue  # skip already evaluated
            
        if "_embeddings_" in filename and filename.endswith(".csv"):
            # Extract base
            embedding_path = os.path.join(base_dir, filename)
            prefix = filename.replace("_embeddings_", "_")
            label_file = prefix.replace(".csv", "_Kmeans.csv")
            label_path = os.path.join(base_dir, label_file)

            if os.path.exists(label_path):
                # Load data
                embeddings = pd.read_csv(embedding_path)
                labels_df = pd.read_csv(label_path)

                if "cluster" in labels_df.columns:
                    labels = labels_df["cluster"].values
                elif "label" in labels_df.columns:
                    labels = labels_df["label"].values
                else:
                    print(f"⚠️ Missing cluster column in: {label_path}")
                    continue

                try:
                    score = silhouette_score(embeddings.values, labels)
                    results.append({
                        "model": filename.split("_")[-2],  # like 'MiniLm12'
                        "file": filename,
                        "score": round(score, 4)
                    })
                except Exception as e:
                    print(f"❌ Error with {filename}: {e}")
            else:
                print(f"❌ Missing label file for: {filename}")

    # Print results
    print("\n✅ Silhouette Scores:")
    for r in results:
        print(f"{r['model']:>20}: {r['score']:.4f} ({r['file']})")

    return results

# Run it
evaluate_silhouette_scores("evaluations/outputs")

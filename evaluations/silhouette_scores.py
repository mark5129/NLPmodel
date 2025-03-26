import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from pathlib import Path

def find_label_column(df):
    for col in ['cluster', 'label', 'topic_int']:
        if col in df.columns:
            return df[col].values
    return None

def evaluate_silhouette_score(embedding_path, label_path):
    try:
        embeddings = pd.read_csv(embedding_path)
        labels_df = pd.read_csv(label_path)

        labels = find_label_column(labels_df)
        if labels is None:
            print(f"⚠️ No 'cluster', 'label' or 'topic_int' column found in: {label_path}")
            return None, None

        score = silhouette_score(embeddings.values, labels)
        return os.path.basename(label_path), score

    except Exception as e:
        print(f"❌ Error processing {embedding_path}: {e}")
        return None, None

def main():
    output_dir = "evaluations/outputs"
    Path("evaluations").mkdir(exist_ok=True)  # Ensure 'evaluations' folder exists
    kmeans_files = list(Path(output_dir).glob("*_Kmeans.csv"))

    results = []
    for kmeans_file in kmeans_files:
        file_stem = kmeans_file.stem
        label_path = str(kmeans_file)
        if "merged_embeddings" in file_stem:
            model = file_stem.split("_")[-2]
            run_id = file_stem.split("_")[0]
            embed_path = f"modelling/outputs/{model}/{run_id}_merged_embeddings_{model}_embeddings.csv"
        elif "manualrun" in file_stem:
            model = file_stem.split("_")[-2]
            embed_path = f"modelling/outputs/{model}/{file_stem}.csv"
        else:
            print(f"⚠️ Skipping unknown format: {file_stem}")
            continue

        if not os.path.exists(embed_path):
            print(f"❌ Missing embedding file for: {kmeans_file.name}")
            continue

        label_file, score = evaluate_silhouette_score(embed_path, label_path)
        if label_file and score is not None:
            print(f"✅ {label_file}: Silhouette Score = {score:.4f}")
            results.append({"file": label_file, "score": score})

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv("evaluations/silhouette_scores.csv", index=False)
        print("📄 Saved silhouette scores to evaluations/silhouette_scores.csv")
    else:
        print("⚠️ No valid silhouette scores computed.")

if __name__ == "__main__":
    main()

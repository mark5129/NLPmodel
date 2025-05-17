import pandas as pd

def assign_main_topic_name(doc_type, model):
    # Last inn filen
    kmeansfile = pd.read_csv(f"evaluations/outputs/manualrun_{doc_type}_{model}_Kmeans.csv")

    # Finn det mest forekommende topic_names for hver topic_int
    dominant_names = (
        kmeansfile.groupby("topic_int")["topic_names_from_bertopic"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    # Legg til ny kolonne med det dominerende navnet for hver rad
    kmeansfile["main_topic_name"] = kmeansfile["topic_int"].map(dominant_names)

    # Lagre ny fil
    kmeansfile.to_csv(f"evaluations/outputs/manualrun_{doc_type}_{model}_Kmeans.csv", index=False)
    
    print(f"{model}✅ Lagret med ny kolonne: main_topic_name")



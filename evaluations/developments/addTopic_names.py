import pandas as pd

def addBERTopic_names(doc_type,model):
    try:
        # Last inn hovedfil og kildefil
        kmeansfile = pd.read_csv(f'evaluations/outputs/manualrun_{doc_type}_{model}_Kmeans.csv')
        Bertopicnames = pd.read_csv(f'evaluations/outputs/manualrun_{model}_{doc_type}_output_clusters.csv')

        # Anta at rekkefølgen er lik og legg til kolonnen 'topic_names'
        kmeansfile["topic_names_from_bertopic"] = Bertopicnames["topic_names"]

            
        # Replace underscores with commas in 'labels_layer'
        kmeansfile["topic_names_from_bertopic"] = kmeansfile["topic_names_from_bertopic"].str.replace('_', ', ')

        # Lagre ny fil
        kmeansfile.to_csv(f'evaluations/outputs/manualrun_{doc_type}_{model}_Kmeans.csv', index=False)
        print(f"{model}✅ Lagret med ny kolonne:")

    except Exception as e:
        print(f"❌ Feil for {model}: {e}")
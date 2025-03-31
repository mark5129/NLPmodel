#pip install openai
from openai import OpenAI
import yaml
import pandas as pd
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# OpenAI API key
client = OpenAI(

)

#usage = 'topic_name'
usage = 'resume'

k_means_path = 'evaluations/outputs/manualrun_merged_embeddings_XLM_Roberta_Kmeans.csv'
texts_path = 'data/merged_media_stemmed_eng.csv'

k_means = pd.read_csv(k_means_path)
all_texts = pd.read_csv(texts_path)

# Get the distinct values of the 'topic_int' column
n_clusters = k_means['topic_int'].unique()

# Sort the distinct values
n_clusters.sort()

# Create a list to store the results
results = []

for n_cluster in n_clusters:
    print(f"Processing n_clusters: {n_cluster}")
    
    cluster_texts = all_texts[k_means['topic_int'] == n_cluster]

    not_cluster_texts = all_texts[k_means['topic_int'] != n_cluster].sample(frac=0.05, random_state=parameters['random_state'])

    # append all texts together in one string and save it to a variable
    Not_cluster_texts = ' \n'.join(not_cluster_texts['Content'].astype(str).tolist())
    joined_cluster_texts = ' \n'.join(cluster_texts['Content'].astype(str).tolist())

    if usage == 'topic_name':

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You have to act like a topic model. You are first being fed a group of documents that are appended together in a string"},
                {"role": "user", "content": Not_cluster_texts},
                {"role": "system", "content": "Once you have read all the documents, then i will feed you a selected group of the documents. Read these as well and come up with a topic name that differentiates these documents from the rest"},  # System prompt
                {"role": "system", "content": "You are not allowed to write anything else than the topic name, just a topic name consisting of max 3 words"}, 
                {"role": "user", "content": joined_cluster_texts}
            ],
            max_tokens=10000
        )

        topic_name = completion.choices[0].message.content
        print(f'{n_cluster}: {topic_name}')
        
        # Append the cluster and topic name to the results list
        results.append({"Cluster": n_cluster, "Topic Name": topic_name})

    elif usage == 'resume':
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
            {"role": "system", "content": "You have to act like a summarization model. You are first being fed a group of documents that are appended together in a string."},
            {"role": "user", "content": Not_cluster_texts},
            {"role": "system", "content": "Once you have read all the documents, then I will feed you a selected group of the documents. Read these as well and generate a concise summary that captures the essence of these documents."},  # System prompt
            {"role": "system", "content": "You are not allowed to write anything else than the summary, just a concise summary of max 100 words."}, 
            {"role": "user", "content": joined_cluster_texts}
            ],
            max_tokens=10000
        )

        summary = completion.choices[0].message.content
        print(f'{n_cluster}: {summary}')
        
        # Append the cluster and summary to the results list
        results.append({"Cluster": n_cluster, "Summary": summary})

# Save the results to a CSV file
output_path = f'evaluations/developments/cluster_{usage}.csv'
pd.DataFrame(results).to_csv(output_path, index=False)

print(f"Results saved to {output_path}")

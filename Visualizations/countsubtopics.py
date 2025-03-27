import pandas as pd
import numpy as np

# This script only runs on embeddings from various models.

# Determine the latest run ID to know which embeddings to use
# Important that this number is the same as the one used in the manual eval script also
Latest_run_id = '1303156299'

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_{model}_embeddings.csv')
    #kmeans = pd.read_csv(f'evaluations/outputs/manualrun_{model}_merged_output_clusters.csv')
    kmeans = pd.read_csv(f'evaluations/outputs/manualrun_merged_{model}_Kmeans.csv')

    # Columns in kmeans: topic_int,topic_names,labels_layer,Source,topic_names_from_bertopic,main_topic_name

    # For each model i want to count the number of main_topic_name in each topic_int

    # Create a new DataFrame to store the counts
    counts_df = pd.DataFrame(columns=['topic_int', 'main_topic_name', 'count'])
    # Iterate through each unique topic_int
    for topic in kmeans['topic_int'].unique():
        for subtopic in kmeans['topic_names_from_bertopic'].unique():
            # Filter the DataFrame for the current topic_int
            filtered_df = kmeans[kmeans['topic_int'] == topic]
            # Count the occurrences of each distinct main_topic_name
            counts = filtered_df.groupby('topic_names_from_bertopic').size().reset_index(name='count')
            counts['topic_int'] = topic  # Add the topic_int to the counts DataFrame
            counts_df = pd.concat([counts_df, counts], ignore_index=True)

    # Save the counts DataFrame to a CSV file
    counts_df.to_csv(f'Visualizations/subtopiccount/manualrun_{model}_merged_output_counts.csv', index=False)


for model in which_model:
    embeddings = pd.read_csv(f'modelling/outputs/{model}/{Latest_run_id}_merged_embeddings_{model}_embeddings.csv')
    #kmeans = pd.read_csv(f'evaluations/outputs/manualrun_{model}_merged_embeddings_output_clusters.csv')
    kmeans = pd.read_csv(f'evaluations/outputs/manualrun_merged_embeddings_{model}_Kmeans.csv')

    # Create a new DataFrame to store the counts
    counts_df = pd.DataFrame(columns=['topic_int', 'main_topic_name', 'count'])
    # Iterate through each unique topic_int
    for topic in kmeans['topic_int'].unique():
        filtered_df = kmeans[kmeans['topic_int'] == topic]
        for subtopic in kmeans['topic_names_from_bertopic'].unique():
            # Filter the DataFrame for the current topic_int
            filtered_df = kmeans[kmeans['topic_names_from_bertopic'] == subtopic]
            # Count the occurrences of topic_names_from_bertopic

            # Create an empty DataFrame with columns topic_int, subtopic, and count
            df = pd.DataFrame(columns=['topic_int', 'subtopic', 'count'])
            
            # count rows
            count = int(filtered_df.shape[0])

            array = np.array([topic, subtopic, count])
            print(array)

            break
            # Create a DataFrame for the current topic_int and subtopic
            topic = pd.Series(topic)
            subtopic = pd.Series(subtopic)
            count = pd.Series(count)
            
            counts_df = pd.concat([topic, subtopic, count], ignore_index=True)

    # Save the counts DataFrame to a CSV file
    counts_df.to_csv(f'Visualizations/subtopiccount/manualrun_{model}_merged_embeddings_output_counts.csv', index=False)



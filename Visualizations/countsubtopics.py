import pandas as pd
import numpy as np

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Which source
doc_type = ['merged', 'merged_embeddings']

for doc in doc_type:
    for model in which_model:
        kmeans = pd.read_csv(f'evaluations/outputs/manualrun_{doc}_{model}_Kmeans.csv')
        
        # Count occurrences
        df_count = kmeans.groupby(['topic_int', 'topic_names_from_bertopic']).size().reset_index(name='count')

        # Count total occurrences per topic
        df_total = kmeans.groupby('topic_int').size().reset_index(name='total_count')

        # Merge total count into df_count
        df_count = df_count.merge(df_total, on='topic_int')

        # Calculate percentage
        df_count['percentage'] = round((df_count['count'] / df_count['total_count']),3)

        # Save the counts DataFrame to a CSV file
        df_count.to_csv(f'Visualizations/subtopiccount/manualrun_{model}_{doc}_output_counts.csv', index=False)

        # Find the row with the maximum percentage for each topic_int
        df_max_percentage = df_count.loc[df_count.groupby('topic_int')['percentage'].idxmax()]

        # Save the DataFrame with the maximum percentage rows to a CSV file
        df_max_percentage.to_csv(f'Visualizations/subtopiccount/manualrun_{model}_{doc}_max_percentage.csv', index=False)
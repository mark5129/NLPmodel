import pandas as pd
import numpy as np

#Load embeddings from embeddings file
which_model = ['XLM_Roberta', 'Specter2Actually', 'MiniLm12']

# Which source
doc_type = ['merged_embeddings']

Top_what = 5

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

        # Filter rows where 'is_closest_half' is True
        kmeans_filtered = kmeans[kmeans['is_closest_half'] == True]

        # Count occurrences for filtered data
        df_count_filtered = kmeans_filtered.groupby(['topic_int', 'topic_names_from_bertopic']).size().reset_index(name='filtered_count')

        # Count total occurrences per topic for filtered data
        df_total_filtered = kmeans_filtered.groupby('topic_int').size().reset_index(name='filtered_total_count')

        # Merge total count into df_count for filtered data
        df_count_filtered = df_count_filtered.merge(df_total_filtered, on='topic_int')

        # Calculate percentage for filtered data
        df_count_filtered['filtered_percentage'] = round((df_count_filtered['filtered_count'] / df_count_filtered['filtered_total_count']), 3)

        # Merge filtered data into the original DataFrame
        df_combined = df_count.merge(df_count_filtered, on=['topic_int', 'topic_names_from_bertopic'], how='left')

        # Add a calculated column that divides filtered percentage by percentage
        df_combined['filtered_to_total_ratio'] = round(df_combined['filtered_percentage'] / df_combined['percentage'],3)

        # Save the combined DataFrame to a CSV file
        #df_combined.to_csv(f'analysis/subtopiccount/manualrun_{model}_{doc}_top{Top_what}_percentage.csv', index=False)

        # Find the row with the maximum percentage for each topic_int
        #df_max_percentage = df_count.loc[df_count.groupby('topic_int')['percentage'].idxmax()]

        # Save the DataFrame with the maximum percentage rows to a CSV file
        #df_max_percentage.to_csv(f'analysis/subtopiccount/manualrun_{model}_{doc}_max_percentage.csv', index=False)

        # Find the top 5 rows with the highest percentages for each topic_int
        df_top5_percentage = df_combined.sort_values('percentage', ascending=False).groupby('topic_int').head(Top_what)

        # Sort by topic_int and percentage
        df_top5_percentage = df_top5_percentage.sort_values(['topic_int', 'percentage'], ascending=[True, False])
        # Save the DataFrame with the top 5 percentage rows to a CSV file
        df_top5_percentage.to_csv(f'analysis/subtopiccount/manualrun_{model}_{doc}_top{Top_what}_percentage.csv', index=False)
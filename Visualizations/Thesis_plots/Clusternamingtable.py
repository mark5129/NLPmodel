import pandas as pd

model = 'XLM_Roberta'

kmeans = pd.read_csv(f'evaluations/outputs/manualrun_merged_embeddings_{model}_Kmeans.csv')

# Columns: topic_int,topic_names,labels_layer,Source,topic_names_from_bertopic,main_topic_name

# Group by 'topic_int' and aggregate the following collumns 'topic_names', main_topic_name

kmeans_grouped = kmeans.groupby('topic_int').agg({
    'topic_names': lambda x: ', '.join(x.unique()),
    'main_topic_name': lambda x: ', '.join(x.unique())
}).reset_index()
# Rename the columns
kmeans_grouped.columns = ['topic_int', 'TF-IDF', 'Most Occuring BERTopic Name']
# Save to CSV
kmeans_grouped.to_csv(f'Visualizations/Thesis_plots/Clusternaming table.csv', index=False)
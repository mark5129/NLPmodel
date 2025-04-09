import pandas as pd

def naming_tableIndividual(Embedd_model, cluster_model):
    
    pro_df = pd.read_csv(f"evaluations/outputs/manualrun_{Embedd_model}_pro_clusters_{cluster_model}.csv")
    reg_df = pd.read_csv(f"evaluations/outputs/manualrun_{Embedd_model}_reg_clusters_{cluster_model}.csv")
    sci_df = pd.read_csv(f"evaluations/outputs/manualrun_{Embedd_model}_sci_clusters_{cluster_model}.csv")

    # Merge the DataFrames
    merged_df = pd.concat([pro_df, reg_df, sci_df], ignore_index=True)

    # remove columns x, y
    merged_df.drop(columns=['x', 'y'], inplace=True)

    # Group merged_df by columns Source and cluster, count occurrences, and add the count as a new column
    count_df = merged_df.groupby(['Source', 'cluster'], as_index=False).size().rename(columns={'size': 'count'})
    merged_df = pd.merge(count_df, merged_df, on=['Source', 'cluster'], how='inner')
    
    merged_df.sort_values(by=['Source', 'cluster'], inplace=True)

    # only keep distinct rows
    merged_df.drop_duplicates(subset=['Source', 'cluster'], inplace=True)

    # add a column that counts the number of words that are in both columns TF_IDF_topic_name and BERTopic_topic_name
    merged_df['count_words'] = merged_df.apply(lambda row: len(set(row['TF_IDF_topic_name'].split()) & set(row['BERTopic_topic_name'].split())), axis=1)

    # Save the merged DataFrame to a CSV file
    output_file = f"Visualizations/tables/namingtable_{Embedd_model}_{cluster_model}.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Saved namingtable {Embedd_model} {cluster_model}")


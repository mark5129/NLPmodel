import pandas as pd

def naming_tableIndividual(merged_df):

    # Group merged_df by columns Source and cluster, count occurrences, and add the count as a new column
    count_df = merged_df.groupby(['Source', 'cluster'], as_index=False).size().rename(columns={'size': 'count'})
    merged_df = pd.merge(count_df, merged_df, on=['Source', 'cluster'], how='inner')
    
    merged_df.sort_values(by=['Source', 'cluster'], inplace=True)

    # only keep distinct rows
    merged_df.drop_duplicates(subset=['Source', 'cluster'], inplace=True)

    return merged_df


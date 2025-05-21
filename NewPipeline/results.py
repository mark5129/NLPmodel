import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plots.set_plot_style import set_style


set_style()
# Load your data
minilm_df = pd.read_csv('NewPipeline/clustering_outputs/AP_MiniLm12_merged_naming_clusters.csv')
specter_df = pd.read_csv('NewPipeline/clustering_outputs/AP_Specter2_merged_naming_clusters.csv')
xlmroberta_df = pd.read_csv('NewPipeline/clustering_outputs/AP_XLM_Roberta_merged_naming_clusters.csv')

# Add a column to identify the model
minilm_df['Model'] = 'MiniLM-L12'
specter_df['Model'] = 'Specter2'
xlmroberta_df['Model'] = 'XLM-Roberta'

# Combine all dataframes
combined_df = pd.concat([minilm_df, specter_df, xlmroberta_df])

# Calculate cluster sizes
combined_df['Cluster Size'] = combined_df[['pro', 'reg', 'sci']].sum(axis=1)

# Function to compute summary statistics for each model
def extract_model_stats(df, model_name):
    total_clusters = df['cluster'].nunique()
    docs_per_cluster = df[['pro', 'reg', 'sci']].sum(axis=1)
    median_cluster_size = docs_per_cluster.median()
    largest_cluster_size = docs_per_cluster.max()
    
    # Calculate clusters with more than one source
    more_than_one_source_clusters = df[(df[['pro', 'reg', 'sci']] > 0).sum(axis=1) > 1].shape[0]
    percent_more_than_one_source = round(100 * more_than_one_source_clusters / total_clusters, 1)
    
    # Calculate clusters where all three sources are represented
    all_three_sources_clusters = df[(df[['pro', 'reg', 'sci']] > 0).sum(axis=1) == 3].shape[0]
    percent_all_three_sources = round(100 * all_three_sources_clusters / total_clusters, 1)

    return {
        "Model": model_name,
        "Total Clusters": total_clusters,
        "Median Size": median_cluster_size,
        "Largest Cluster": largest_cluster_size,
        "% >1 Source": f"{percent_more_than_one_source}%",
        "% All 3 Sources": f"{percent_all_three_sources}%"
    }

# Generate stats for each model
stats = [
    extract_model_stats(specter_df, "Specter2"),
    extract_model_stats(xlmroberta_df, "XLM-Roberta"),
    extract_model_stats(minilm_df, "MiniLM-L12")
]

# Convert to DataFrame for display or export
comparison_df = pd.DataFrame(stats)
print(comparison_df)
comparison_df.to_csv('NewPipeline/model_comparison_table.csv', index=False)

# Create the box plot using plt
plt.figure(figsize=(10, 6))

# Prepare data for plotting
models = combined_df['Model'].unique()
data = [combined_df[combined_df['Model'] == model]['Cluster Size'] for model in models]

# Create the box plot
plt.boxplot(data, labels=models, patch_artist=True, boxprops=dict(facecolor='none', color='black'),medianprops=dict(color='red'))

# Add titles and labels
plt.title('Distribution of Cluster Sizes by Model', fontsize=16)
#plt.xlabel('Model')
plt.ylabel('Cluster Size')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

# Define a function to calculate the number of clusters for each combination
def calculate_cluster_combinations(df):
    reg_only = df[(df['reg'] > 0) & (df['pro'] == 0) & (df['sci'] == 0)].shape[0]
    pro_only = df[(df['pro'] > 0) & (df['reg'] == 0) & (df['sci'] == 0)].shape[0]
    sci_only = df[(df['sci'] > 0) & (df['reg'] == 0) & (df['pro'] == 0)].shape[0]
    reg_pro = df[(df['reg'] > 0) & (df['pro'] > 0) & (df['sci'] == 0)].shape[0]
    reg_sci = df[(df['reg'] > 0) & (df['sci'] > 0) & (df['pro'] == 0)].shape[0]
    pro_sci = df[(df['pro'] > 0) & (df['sci'] > 0) & (df['reg'] == 0)].shape[0]
    all_three = df[(df['reg'] > 0) & (df['pro'] > 0) & (df['sci'] > 0)].shape[0]
    
    # Calculate clusters with only one document
    one_doc_clusters = df[(df[['pro', 'reg', 'sci']].sum(axis=1) == 1)].shape[0]
    
    return {
        "Reg Only": reg_only,
        "Pro Only": pro_only,
        "Sci Only": sci_only,
        "Reg + Pro": reg_pro,
        "Reg + Sci": reg_sci,
        "Pro + Sci": pro_sci,
        "All Three": all_three,
        "One Doc Clusters": one_doc_clusters
    }

# Combine the results into a DataFrame
combinations_df = pd.DataFrame({
    "MiniLM-L12": calculate_cluster_combinations(minilm_df),
    "Specter2": calculate_cluster_combinations(specter_df),
    "XLM-Roberta": calculate_cluster_combinations(xlmroberta_df)
}).T

# Transpose the DataFrame to switch axes
combinations_df_transposed = combinations_df.T

# Ensure "One Doc Clusters" is included in the DataFrame
print(combinations_df_transposed)

# Plot the bar chart
combinations_df_transposed.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Number of Clusters by Source Combinations', fontsize=16)
plt.xlabel('Source Combination', fontsize=12)
plt.ylabel('Number of Clusters', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Model', fontsize=10)
plt.tight_layout()

# Show the plot
plt.show()

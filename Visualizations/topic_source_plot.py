import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def topic_source_plot(data, current_id, doc_type, model_name):
    # Define a consistent color palette for the sources
    palette = {
        "Sci Media": "#1f77b4",  # Blue
        "Pro Media": "#ff7f0e",  # Orange
        "Reg Media": "#2ca02c"   # Green
    }

    # Count the instances
    count_data = data.groupby(['topic_names', 'Source']).size().reset_index(name='Count')

    # Calculate the proportion
    source_counts = data['Source'].value_counts().to_dict()
    count_data['Proportion'] = count_data.apply(lambda row: row['Count'] / source_counts[row['Source']], axis=1)

    # Sort the data by topic names
    count_data = count_data.sort_values(by='topic_names')

    # Plot the data
    plt.figure(figsize=(12, 8))
    sns.barplot(x='topic_names', y='Proportion', hue='Source', data=count_data, palette=palette)
    plt.ylabel('Proportion')
    plt.title(f'Proportion of Topic Names by Source ({model_name})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to make room for the rotated x labels
    
    # Save Visualization to CSV
    output_dir = 'Visualizations/outputs/'

    plt.savefig(f"{output_dir}{current_id}_{doc_type}_{model_name}_sourceplot.png")
    print(f"Saved sourceplot for {model_name} to {output_dir}")



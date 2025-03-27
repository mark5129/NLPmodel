import pandas as pd

# Load the data file for the merged documents
df_file = pd.read_csv('data/merged_media_stemmed_eng.csv')

cluster_texts = df_file['Content'].astype(str).tolist()
Cluster_source = df_file['Source']


# I want to count the number of letters in each text and make a boxplot of the distribution

text_lengths = [len(text) for text in cluster_texts]

df_file['text_lengths'] = text_lengths

import matplotlib.pyplot as plt

# Group text lengths by source
sources = df_file['Source'].unique()
source_text_lengths = [df_file[df_file['Source'] == source]['text_lengths'].tolist() for source in sources]

# How many texts within each source have more than 10000 letters?
for source in sources:
    source_texts = df_file[df_file['Source'] == source]['text_lengths']
    count = len([length for length in source_texts if length > 10000])
    print(f"Source '{source}' has {count} texts with more than 10000 letters.")

# Create boxplot
plt.boxplot(source_text_lengths, labels=sources)
plt.title('Distribution of text lengths by Source')
plt.xlabel('Source')
plt.ylabel('Number of letters')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Add a horizontal line at 10000 letters
plt.axhline(y=10000, color='red', linestyle='--', label='10000 letters')

# Add a text box with the number of texts with more than 10000 letters in each source
text_box_content = "\n".join(
    [f"{source}: {len([length for length in df_file[df_file['Source'] == source]['text_lengths'] if length > 10000])} texts"
     for source in sources]
)
plt.gcf().text(0.95, 0.90, text_box_content, fontsize=10, verticalalignment='top', horizontalalignment='right', 
               bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()



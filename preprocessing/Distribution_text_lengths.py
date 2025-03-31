import pandas as pd

import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


# Define limit 

Number_characters = 15000
# Load the data file for the merged documents
pro_media_df = pd.read_csv(parameters['pro_media_translated_dir'])
reg_media_df = pd.read_csv(parameters['reg_media_translated_dir'])
sci_media_df = pd.read_csv(parameters['sci_media_dir'])

# Define the columns to keep from each dataframe
pro_media_columns = ['Date', 'Title', 'Outlet', 'Content']  # Replace with actual column names
reg_media_columns = ['Date', 'Title', 'Outlet', 'Content']  # Replace with actual column names
sci_media_columns = ['Date', 'Title', 'Outlet', 'Content']  # Replace with actual column names

# Select the specified columns from each dataframe
pro_media_df = pro_media_df[pro_media_columns]
reg_media_df = reg_media_df[reg_media_columns]
sci_media_df = sci_media_df[sci_media_columns]

# Add a column to determine the source of the text
pro_media_df['Source'] = 'Pro Media'
reg_media_df['Source'] = 'Reg Media'
sci_media_df['Source'] = 'Sci Media'

# Rename columns to have the same names for merging
pro_media_df.columns = reg_media_df.columns

# Merge the dataframes
df_file = pd.concat([pro_media_df, reg_media_df, sci_media_df])

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
    count = len([length for length in source_texts if length > Number_characters])
    print(f"Source '{source}' has {count} texts with more than {Number_characters} letters.")

# Create boxplot
plt.boxplot(source_text_lengths, labels=sources)
plt.title('Distribution of document lengths by Source')
#plt.xlabel('Source')
plt.ylabel('Number of characters')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Add a horizontal line at 10000 letters
plt.axhline(y=Number_characters, color='red', linestyle='--', label='{Number_characters} letters')

# Add a text box with the number of texts with more than 10000 letters in each source
text_box_content = "\n".join(
    [f"{source}: {len([length for length in df_file[df_file['Source'] == source]['text_lengths'] if length > Number_characters])} texts"
     for source in sources]
)
plt.gcf().text(0.95, 0.90, text_box_content, fontsize=10, verticalalignment='top', horizontalalignment='right', 
               bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('preprocessing/text_lengths_distribution.png', dpi=300)

# Print the number of totalt texts for each source and the number of texts with more than 10000 letters
for source in sources:
    total_count = len(df_file[df_file['Source'] == source])
    print(f"Source '{source}' has {total_count} texts in total.")
    count = len([length for length in df_file[df_file['Source'] == source]['text_lengths'] if length > Number_characters])
    print(f"Source '{source}' has {count} texts with more than {Number_characters} letters.")

# What is the total number of texts in the dataset?
total_texts = len(df_file)
print(f"Total number of texts in the dataset: {total_texts}")

# What is the total number of texts with less than 10000 letters in the dataset?
total_texts_less_than_limit = len([length for length in df_file['text_lengths'] if length < Number_characters])
print(f"Total number of texts with less than {Number_characters} letters in the dataset: {total_texts_less_than_limit}")



import pandas as pd


# Load the CSV files
pro_media_df = pd.read_csv('data/pro_media_stemmed_eng.csv')
reg_media_df = pd.read_csv('data/reg_media_stemmed_eng.csv')

# Define the columns to keep from each dataframe
pro_media_columns = ['Date', 'Title', 'Outlet', 'Content']  # Replace with actual column names
reg_media_columns = ['Date', 'Title', 'Outlet', 'Content']  # Replace with actual column names

# Select the specified columns from each dataframe
pro_media_df = pro_media_df[pro_media_columns]
reg_media_df = reg_media_df[reg_media_columns]

# Add a column to determine the source of the text
pro_media_df['Source'] = 'Pro Media'
reg_media_df['Source'] = 'Reg Media'

# Rename columns to have the same names for merging
pro_media_df.columns = reg_media_df.columns

# Merge the dataframes
merged_df = pd.concat([pro_media_df, reg_media_df])

# Save the merged dataframe to a new CSV file
merged_df.to_csv('data/merged_media_stemmed_eng.csv', index=False)
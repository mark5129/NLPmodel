# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import pandas as pd

def Merging_textfiles(preprocess_step):
    # Load the CSV files
    pro_media_df = pd.read_csv(parameters[f'pro_media_{preprocess_step}_dir'])
    reg_media_df = pd.read_csv(parameters[f'reg_media_{preprocess_step}_dir'])
    sci_media_df = pd.read_csv(parameters[f'sci_media_{preprocess_step}_dir'])

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
    merged_df = pd.concat([pro_media_df, reg_media_df, sci_media_df])

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(parameters['merged_stemmed_dir'], index=False)

    print('Text files are merged successfully!')

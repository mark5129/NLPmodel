# import libraries
import pandas as pd

# import functions for preprocessing
from preprocessing.Clean_rows import clean_rows
from preprocessing.stopwords import remove_stopwords
from preprocessing.stemming import stemming
from preprocessing.Merge_textfiles import Merging_textfiles


from modelling.outputs.logging import log_parameters

# import functions for modelling
from modelling.XLM_Roberta import XLM_Roberta_model
from modelling.minilm12 import MiniLM12
from modelling.Specter2Actually import Specter2ActuallyModel
from modelling.mergeEmbeddings import merge_embeddings

# load parameters from yaml file.
import yaml
import os
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Save the parameters to a CSV file
current_id, time = log_parameters(parameters)

sources = ['pro', 'reg', 'sci']

if parameters['preprocess_data'] == True:
    
    for source in sources:
        
        # Perform cleaning on pro_media.csv
        df = pd.read_csv(parameters[f'{source}_media_translated_dir']) # Load the CSV file
        df = clean_rows(df) # Clean the rows of the dataframe
        df['Content'] = df['Content'].apply(lambda x: remove_stopwords(x, 'english')) # Apply the function to the text column
        df.to_csv(parameters[f'{source}_media_cleaned_dir'], index=False) # Save the cleaned text to a new CSV file
        print(f'{source}_media.csv is cleaned')

        # Perform sentence stemming on pro_media.csv
        df = pd.read_csv(parameters[f'{source}_media_cleaned_dir']) # Load the CSV file
        df['Content'] = df['Content'].apply(lambda x: stemming(x, 'english'))
        df.to_csv(parameters[f'{source}_media_stemmed_dir'], index=False) # Save the embeddings to a new CSV file
        print(f'{source}_media.csv is stemmed')



    Merging_textfiles()
    
else:
    print('Data preprocessing is turned off in parameters.yaml')

if parameters['train_model'] == True:

    for source in sources:

        df_file = pd.read_csv(parameters[f'{source}_media_stemmed_dir'])

        df_text_column = df_file['Content']
        
        if parameters['train_minilm12'] == True:
            MiniLM12(df_text_column, current_id, source)
            print('MiniLM12 embeddings are generated.')
        
        if parameters['train_xlm_roberta'] == True:
            XLM_Roberta_model(df_text_column, current_id, source)
            print('XLM-Roberta embeddings are generated.')
        
        if parameters['train_specter2_Actually'] == True:
            Specter2ActuallyModel(df_text_column, current_id, source)

            print('Specter2 embeddings are generated.')

    merge_embeddings('MiniLm12', current_id)
    merge_embeddings('Specter2Actually', current_id)
    merge_embeddings('XLM_Roberta', current_id)
    print(f'Embeddings are merged')

else:
    print('Model training is turned off in parameters.yaml')
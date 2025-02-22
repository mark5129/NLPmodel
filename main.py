# import libraries
import pandas as pd

# import functions
from preprocessing.stopwords import remove_stopwords
from outputs.logging import log_parameters, update_header_if_needed
from modelling.LDA import LDAModel
#from preprocessing.sentence_transforming import sentence_transformer

# load parameters from yaml file.
import yaml
import os
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Save the parameters to a CSV file
log_parameters(parameters)

if parameters['preprocess_data'] == True:
    
    # Perform cleaning on pro_media.csv
    df_pro = pd.read_csv(parameters['pro_media_dir']) # Load the CSV file
    df_pro['Full text'] = df_pro['Full text'].apply(lambda x: remove_stopwords(x, 'danish')) # Apply the function to the text column
    df_pro.to_csv(parameters['pro_media_cleaned_dir'], index=False) # Save the cleaned text to a new CSV file
    print('pro_media.csv is cleaned')

    # Perform cleaning on reg_media.csv
    df_reg = pd.read_csv(parameters['reg_media_dir'])
    df_reg['Content'] = df_reg['Content'].apply(lambda x: remove_stopwords(x, 'danish'))
    df_reg.to_csv('reg_media_cleaned_dir', index=False)
    print('reg_media.csv is cleaned')

    # Perform sentence embedding on pro_media.csv
    #df_pro = pd.read_csv(parameters['pro_media_cleaned_dir']) # Load the CSV file
    #df_pro['Full text'] = sentence_transformer(df_pro['Full text']) # Encode the text column
    #df_pro.to_csv(parameters['pro_media_embedded_dir'], index=False) # Save the embeddings to a new CSV file
    #print('pro_media.csv is embedded')

    # Perform cleaning on reg_media.csv
    #df_reg = pd.read_csv(parameters['reg_media_cleaned_dir'])
    #df_reg['Content'] = sentence_transformer(df_reg['Content'])
    #df_reg.to_csv(parameters['reg_media_embedded_dir'], index=False)
    #print('reg_media.csv is embedded')
    
else:
    print('Data preprocessing is turned off in parameters.yaml')

if parameters['train_model'] == True:

    reg_media = pd.read_csv(parameters['reg_media_cleaned_dir'])
    pro_media = pd.read_csv(parameters['pro_media_cleaned_dir'])

    number_of_topics = parameters['num_topics']
    reg_text_column = reg_media['Content']
    pro_text_column = pro_media['Full text']

    LDAModel(reg_text_column, number_of_topics)

    LDAModel(pro_text_column, number_of_topics)

else:
    print('Model training is turned off in parameters.yaml')

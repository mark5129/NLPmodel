# import libraries
import pandas as pd
import seaborn as sns
import os
import csv
# import functions for preprocessing
from preprocessing.stopwords import remove_stopwords
from preprocessing.stemming import stemming
#from preprocessing.sentence_transforming import sentence_transformer

from outputs.logging import log_parameters, update_header_if_needed

# import functions for modelling
from modelling.LDA import LDAModel
from modelling.Specter2 import Specter2Model
from modelling.NMF import NMF_model
from modelling.XLM_Roberta import XLM_Roberta_model
from modelling.BERTopic import BERTopicModel

# load parameters from yaml file.
import yaml
import os
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Save the parameters to a CSV file
current_id, time = log_parameters(parameters)

if parameters['preprocess_data'] == True:
    
    # Perform cleaning on pro_media.csv
    df_pro = pd.read_csv(parameters['pro_media_translated_dir']) # Load the CSV file
    df_pro['Full text'] = df_pro['Full text'].apply(lambda x: remove_stopwords(x, 'english')) # Apply the function to the text column
    df_pro.to_csv(parameters['pro_media_cleaned_dir'], index=False) # Save the cleaned text to a new CSV file
    print('pro_media.csv is cleaned')

    # Perform cleaning on reg_media.csv
    df_reg = pd.read_csv(parameters['reg_media_translated_dir'])
    df_reg['Content'] = df_reg['Content'].apply(lambda x: remove_stopwords(x, 'english'))
    df_reg.to_csv(parameters['reg_media_cleaned_dir'], index=False)
    print('reg_media.csv is cleaned')

    # Perform sentence stemming on pro_media.csv
    df_pro = pd.read_csv(parameters['pro_media_cleaned_dir']) # Load the CSV file
    df_pro['Full text'] = df_pro['Full text'].apply(lambda x: stemming(x, 'english'))
    df_pro.to_csv(parameters['pro_media_stemmed_dir'], index=False) # Save the embeddings to a new CSV file
    print('pro_media.csv is stemmed')

    # Perform cleaning on reg_media.csv
    df_reg = pd.read_csv(parameters['reg_media_cleaned_dir'])
    df_reg['Content'] = df_reg['Content'].apply(lambda x: stemming(x, 'english'))
    df_reg.to_csv(parameters['reg_media_stemmed_dir'], index=False)
    print('reg_media.csv is stemmed')
    
else:
    print('Data preprocessing is turned off in parameters.yaml')

if parameters['train_model'] == True:

    if parameters['what_data'] == 'raw':
        reg_media = pd.read_csv(parameters['reg_media_translated_dir'])
        pro_media = pd.read_csv(parameters['pro_media_translated_dir'])
    elif parameters['what_data'] == 'cleaned':
        reg_media = pd.read_csv(parameters['reg_media_cleaned_dir'])
        pro_media = pd.read_csv(parameters['pro_media_cleaned_dir'])
    elif parameters['what_data'] == 'stemmed':
        reg_media = pd.read_csv(parameters['reg_media_stemmed_dir'])
        pro_media = pd.read_csv(parameters['pro_media_stemmed_dir'])
    else:
        print('No or wrong data source selected in parameters.yaml')

    reg_text_column = reg_media['Content']
    pro_text_column = pro_media['Full text']


    if parameters['train_lda'] == True:
        
        LDAModel(reg_text_column, 'english', current_id, 'reg')
        LDAModel(pro_text_column, 'english', current_id, 'pro')
        print('LDA model is trained')

    if parameters['train_specter2'] == True:

        Specter2Model(reg_text_column, current_id, 'reg')
        Specter2Model(pro_text_column, current_id, 'pro')
        print('Specter2 embeddings are generated.')

    if parameters['train_nmf'] == True:

        NMF_model(reg_text_column, current_id, 'reg')
        NMF_model(pro_text_column, current_id, 'pro')
        print('NMF model is trained')
    
    if parameters['train_bert'] == True:
        reg_topics, reg_topic_model = BERTopicModel(reg_text_column, "reg_media")
        pro_topics, pro_topic_model = BERTopicModel(pro_text_column, "pro_media")


else:
    print('Model training is turned off in parameters.yaml')
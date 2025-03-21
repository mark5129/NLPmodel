# import libraries
import pandas as pd
import seaborn as sns
import os
import csv
# import functions for preprocessing
from preprocessing.Clean_rows import clean_rows
from preprocessing.stopwords import remove_stopwords
from preprocessing.stemming import stemming

#from preprocessing.sentence_transforming import sentence_transformer
from modelling.outputs.logging import log_parameters, update_header_if_needed

# import functions for modelling
from modelling.LDA import LDAModel
from modelling.Specter2 import Specter2Model
from modelling.NMF import NMF_model
from modelling.XLM_Roberta import XLM_Roberta_model
from modelling.BERTopic import BERTopicModel
from modelling.minilm12 import MiniLM12
from modelling.Specter2Actually import Specter2ActuallyModel


# Import functions for evaluation
from evaluations.TFIDF_cluster_topics import TFIDF_clustering
from evaluations.mergeEmbeddings import merge_embeddings

# import functions for visualisation
from Visualizations.datamapplot_with_naming import data_mapplot_with_naming
from Visualizations.topic_source_plot import topic_source_plot
# Import functions for visualisation
from Visualizations.Bokeh import create_bokeh_plot  # New import for interactive Bokeh plots


# load parameters from yaml file.
import yaml
import os
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Save the parameters to a CSV file
current_id, time = log_parameters(parameters)

if parameters['preprocess_data'] == True:

    # Clean sci_media.
    clean_rows
    
    # Perform cleaning on pro_media.csv
    df_pro = pd.read_csv(parameters['pro_media_translated_dir']) # Load the CSV file
    df_pro['Content'] = df_pro['Content'].apply(lambda x: remove_stopwords(x, 'english')) # Apply the function to the text column
    df_pro.to_csv(parameters['pro_media_cleaned_dir'], index=False) # Save the cleaned text to a new CSV file
    print('pro_media.csv is cleaned')

    # Perform cleaning on reg_media.csv
    df_reg = pd.read_csv(parameters['reg_media_translated_dir'])
    df_reg['Content'] = df_reg['Content'].apply(lambda x: remove_stopwords(x, 'english'))
    df_reg.to_csv(parameters['reg_media_cleaned_dir'], index=False)
    print('reg_media.csv is cleaned')

    # Perform cleaning on sci_media.csv
    df_sci = pd.read_csv(parameters['sci_media_dir'])
    df_sci = clean_rows(df_sci)
    df_sci['Content'] = df_sci['Content'].apply(lambda x: remove_stopwords(x, 'english'))
    df_sci.to_csv(parameters['sci_media_cleaned_dir'], index=False)
    print('sci_media.csv is cleaned')

    # Perform sentence stemming on pro_media.csv
    df_pro = pd.read_csv(parameters['pro_media_cleaned_dir']) # Load the CSV file
    df_pro['Content'] = df_pro['Content'].apply(lambda x: stemming(x, 'english'))
    df_pro.to_csv(parameters['pro_media_stemmed_dir'], index=False) # Save the embeddings to a new CSV file
    print('pro_media.csv is stemmed')

    # Perform sentence stemming on reg_media.csv
    df_reg = pd.read_csv(parameters['reg_media_cleaned_dir'])
    df_reg['Content'] = df_reg['Content'].apply(lambda x: stemming(x, 'english'))
    df_reg.to_csv(parameters['reg_media_stemmed_dir'], index=False)
    print('reg_media.csv is stemmed')

    # Perform sentence stemming on sci_media.csv
    df_sci = pd.read_csv(parameters['sci_media_cleaned_dir'])
    df_sci['Content'] = df_sci['Content'].apply(lambda x: stemming(x, 'english'))
    df_sci.to_csv(parameters['sci_media_stemmed_dir'], index=False)
    print('sci_media.csv is stemmed')
    
else:
    print('Data preprocessing is turned off in parameters.yaml')




if parameters['train_model'] == True:

    if parameters['what_data'] == 'raw':
        reg_media = pd.read_csv(parameters['reg_media_translated_dir'])
        pro_media = pd.read_csv(parameters['pro_media_translated_dir'])
    elif parameters['what_data'] == 'cleaned':
        reg_media = pd.read_csv(parameters['reg_media_cleaned_dir'])
        pro_media = pd.read_csv(parameters['pro_media_cleaned_dir'])
        sci_media = pd.read_csv(parameters['sci_media_cleaned_dir'])
    elif parameters['what_data'] == 'stemmed':
        reg_media = pd.read_csv(parameters['reg_media_stemmed_dir'])
        pro_media = pd.read_csv(parameters['pro_media_stemmed_dir'])
        sci_media = pd.read_csv(parameters['sci_media_stemmed_dir'])

        
    else:
        print('No or wrong data source selected in parameters.yaml')

    reg_text_column = reg_media['Content']
    pro_text_column = pro_media['Content']
    sci_text_column = sci_media['Content']
    merged_media = pd.read_csv(parameters['merged_stemmed_dir'])
    merged_text_column = merged_media['Content']


    if parameters['train_lda'] == True:
        
        #LDAModel(reg_text_column, 'english', current_id, 'reg')
        #LDAModel(pro_text_column, 'english', current_id, 'pro')
        #LDAModel(sci_text_column, 'english', current_id, 'sci')
        LDAModel(merged_text_column, 'english', current_id, 'merged')
        print('LDA model is trained')
    
    if parameters['train_nmf'] == True:

        #NMF_model(reg_text_column, current_id, 'reg')
        #NMF_model(pro_text_column, current_id, 'pro')
        #NMF_model(sci_text_column, current_id, 'sci')
        NMF_model(merged_text_column, current_id, 'merged')
        print('NMF model is trained')

    #if parameters['train_specter2'] == True:

        #Specter2Model(reg_text_column, current_id, 'reg')
        #Specter2Model(pro_text_column, current_id, 'pro')
        #Specter2Model(sci_text_column, current_id, 'sci')
        #Specter2Model(merged_text_column, current_id, 'merged')
        #print('Specter2 embeddings are generated.')
    
    #if parameters['train_bert'] == True:
        #reg_topics, reg_topic_model = BERTopicModel(reg_text_column, current_id, "reg")
        #pro_topics, pro_topic_model = BERTopicModel(pro_text_column, current_id, "pro")
        #pro_topics, pro_topic_model = BERTopicModel(sci_text_column, current_id, "sci")
        #merged_topics, merged_topic_model = BERTopicModel(merged_text_column, current_id, "merged")
        #print('BERTopic model is trained')
    
    if parameters['train_minilm12'] == True:
        MiniLM12(reg_text_column, current_id, 'reg')
        MiniLM12(pro_text_column, current_id, 'pro')
        MiniLM12(sci_text_column, current_id, 'sci')
        MiniLM12(merged_text_column, current_id, 'merged')
        print('MiniLM12 embeddings are generated.')
    
    if parameters['train_xlm_roberta'] == True:
        XLM_Roberta_model(reg_text_column, current_id, 'reg')
        XLM_Roberta_model(pro_text_column, current_id, 'pro')
        XLM_Roberta_model(sci_text_column, current_id, 'sci')
        XLM_Roberta_model(merged_text_column, current_id, 'merged')
        print('XLM-Roberta embeddings are generated.')
    
    if parameters['train_specter2_Actually'] == True:
        Specter2ActuallyModel(reg_text_column, current_id, 'reg')
        Specter2ActuallyModel(pro_text_column, current_id, 'pro')
        Specter2ActuallyModel(sci_text_column, current_id, 'sci')
        Specter2ActuallyModel(merged_text_column, current_id, 'merged')
        print('Specter2 embeddings are generated.')

    merge_embeddings('MiniLm12', current_id)
    merge_embeddings('Specter2Actually', current_id)
    merge_embeddings('XLM_Roberta', current_id)
    print('Embeddings are merged')

else:
    print('Model training is turned off in parameters.yaml')

if parameters['Calculate_evaluations'] == True:

    # This mapplot only runs on embeddings based on merged files, this vissualization can also be run in visualization main script.
    if parameters['train_model'] == True:
        # This determines which model to use for the mapplot
        which_model = ['MiniLm12', 'Specter2Actually', 'XLM_Roberta']

        if parameters['train_minilm12'] == False:
            # remove minilm12 from which model list
            which_model.remove('MiniLm12')
        
        if parameters['train_specter2_Actually'] == False:
            # remove specter2 from which model list
            which_model.remove('Specter2Actually')
        
        if parameters['train_xlm_roberta'] == False:
            # remove xlm_roberta from which model list
            which_model.remove('XLM_Roberta')
        
        df = pd.read_csv('data/merged_media_stemmed_eng.csv')
        for model in which_model:
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{current_id}_merged_{model}_embeddings.csv')
            TFIDF_clustering(embeddings, df, current_id, 'merged', model)
        
        for model in which_model:
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{current_id}_merged_embeddings_{model}_embeddings.csv')
            TFIDF_clustering(embeddings, df, current_id, 'merged_embeddings', model)
        
    else:
        print('Model training is turned off in parameters.yaml and therefore datamaplot cannot run in main script.')

else:
    print('Calculate evaluations is turned off in parameters.yaml')


if parameters['Create_Visualizations'] == True:

    # This mapplot only runs on embeddings based on merged files, this vissualization can also be run in visualization main script.
    if parameters['train_model'] == True:
        # This determines which model to use for the mapplot
        which_model = ['MiniLm12', 'Specter2Actually', 'XLM_Roberta']

        if parameters['train_minilm12'] == False:
            # remove minilm12 from which model list
            which_model.remove('MiniLm12')
        
        if parameters['train_specter2_Actually'] == False:
            # remove specter2 from which model list
            which_model.remove('Specter2Actually')
        
        if parameters['train_xlm_roberta'] == False:
            # remove xlm_roberta from which model list
            which_model.remove('XLM_Roberta')
        
        df = pd.read_csv('data/merged_media_stemmed_eng.csv')
        
        for model in which_model:
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{current_id}_merged_{model}_embeddings.csv')
            kmeans = pd.read_csv(f'evaluations/outputs/{current_id}_merged_{model}_Kmeans.csv')
            data_mapplot_with_naming(kmeans, embeddings, df, current_id, 'merged', model)
            topic_source_plot(kmeans, current_id, 'merged', model)
            create_bokeh_plot(kmeans, embeddings, df, current_id, 'merged', model)
        
        for model in which_model:
            embeddings = pd.read_csv(f'modelling/outputs/{model}/{current_id}_merged_embeddings_{model}_embeddings.csv')
            kmeans = pd.read_csv(f'evaluations/outputs/{current_id}_merged_embeddings_{model}_Kmeans.csv')
            data_mapplot_with_naming(kmeans, embeddings, df, current_id, 'merged_embeddings', model)
            topic_source_plot(kmeans, current_id, 'merged_embeddings', model)
            create_bokeh_plot(kmeans, embeddings, df, current_id, 'merged_embeddings', model)

        
    else:
        print('Model training is turned off in parameters.yaml and therefore datamaplot cannot run in main script.')

else:
    print('Visualizations is turned off in parameters.yaml')
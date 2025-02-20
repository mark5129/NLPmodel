import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)


if parameters['preprocess_data'] == True:
    # Define stop words
    stop_words_dk = set(stopwords.words('danish'))

    # Function to remove stop words
    def remove_stopwords(text):
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words_dk]
        return ' '.join(filtered_text)

    # Perform cleaning on pro_media.csv
    # Load the CSV file
    df_pro = pd.read_csv('data/pro_media.csv')

    # Apply the function to the text column
    df_pro['Full text'] = df_pro['Full text'].apply(remove_stopwords)

    # Save the cleaned text to a new CSV file
    df_pro.to_csv('data/pro_media_cleaned.csv', index=False)

    # Perform cleaning on reg_media.csv
    df_reg = pd.read_csv('data/reg_media.csv')
    df_reg['Content'] = df_reg['Content'].apply(remove_stopwords)
    df_reg.to_csv('data/reg_media_cleaned.csv', index=False)


# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

# Huggingface sentence transformer
from sentence_transformers import SentenceTransformer
import pandas as pd

def sentence_transformer(text: str):
    """
    Transforms a text into a sentence embedding.

    Parameters:
    text (str): The text to transform.

    Returns:
    text (str): The transformed text.
    """

    model = SentenceTransformer('distiluse-base-multilingual-cased')
    return model.encode(text).tolist()

# this cannot currenthly be run as the sentence transformer model is too large to be run on the local machine.
# Perform sentence embedding on pro_media.csv
df_pro = pd.read_csv(parameters['pro_media_cleaned_dir']) # Load the CSV file
#df_pro['Full text'] = sentence_transformer(df_pro['Full text']) # Encode the text column
df_pro.to_csv(parameters['pro_media_embedded_dir'], index=False) # Save the embeddings to a new CSV file
print('pro_media.csv is embedded')

# Perform cleaning on reg_media.csv
df_reg = pd.read_csv(parameters['reg_media_cleaned_dir'])
#df_reg['Content'] = sentence_transformer(df_reg['Content'])
df_reg.to_csv(parameters['reg_media_embedded_dir'], index=False)
print('reg_media.csv is embedded')
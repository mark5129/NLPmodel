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
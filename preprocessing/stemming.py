# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

from nltk.stem.snowball import DanishStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stemming(text: str, language: str):
    """
    Stems words in a text.

    Parameters:
    text: The text to stem.
    language (str): What language to use.

    Returns:
    text (str): The text with words stemmed.
    """

    if language == 'danish':
        stemmer = DanishStemmer()
    elif language == 'english':
        stemmer = PorterStemmer()
    else:
        raise ValueError('Language not supported')

    word_tokens = text.split()
    stemmed_text = [stemmer.stem(word) for word in word_tokens]
    return ' '.join(stemmed_text)

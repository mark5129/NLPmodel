from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

def remove_stopwords(text: str,language: str):
    """
    Removes stopwords from a text.

    Parameters:
    text: The text to remove stopwords from.
    language (str): What language to use.

    Returns:
    text (str): The text with stopwords removed.
    """


    if language == 'danish':
        stop_words = set(stopwords.words('danish'))
    elif language == 'english':
        stop_words = set(stopwords.words('english'))
    else:
        raise ValueError('Language not supported')
    
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)
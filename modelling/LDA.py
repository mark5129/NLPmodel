# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import nltk
from nltk.corpus import stopwords  #stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def LDAModel(text_column, number_of_topics):
    """
    Transforms a text into a sentence embedding.

    Parameters:
    text (str): The text to transform.
    number_of_topics (int): The number of topics to generate.

    Returns:
    text (str): The transformed text.
    """
    stop_words=set(nltk.corpus.stopwords.words('danish'))

    vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)
    vect_text=vect.fit_transform(text_column)

    lda_model=LatentDirichletAllocation(n_components=number_of_topics,
    learning_method='online',random_state=42,max_iter=1) 
    lda_top=lda_model.fit_transform(vect_text)

    vocab = vect.get_feature_names_out()
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
        print("Topic "+str(i)+": ")
        for t in sorted_words:
                print(t[0],end=" ")
                print("n")

    return lda_top
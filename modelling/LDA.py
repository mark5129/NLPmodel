# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

import nltk
from nltk.corpus import stopwords  #stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import csv
import os

def LDAModel(text_column, language, current_id, doc_type):
    """
    Transforms a text into a sentence embedding.

    Parameters:
    text (str): The text to transform.
    number_of_topics (int): The number of topics to generate.

    Returns:
    text (str): The transformed text.
    """
    stop_words=list(nltk.corpus.stopwords.words(language))


    vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)
    vect_text=vect.fit_transform(text_column)

    lda_model=LatentDirichletAllocation(n_components=parameters['num_topics'], learning_method='online',random_state=42,max_iter=1) 
    lda_top=lda_model.fit_transform(vect_text)

    output_dir = 'modelling/outputs/LDA'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f'{current_id}_{doc_type}_LDA_topics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Topic', 'Words'])
        vocab = vect.get_feature_names_out()
        for i, comp in enumerate(lda_model.components_):
            vocab_comp = zip(vocab, comp)
            sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:10]
            topic_words = " ".join([t[0] for t in sorted_words])
            writer.writerow([f'{i}', topic_words])


    return lda_top
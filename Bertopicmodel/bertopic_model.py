import pandas as pd

# Load the BERTopic library
from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

models = ['Specter2'] #['XLM_Roberta', 'Specter2', 'MiniLm12']

sources = ['pro', 'reg', 'sci']

for model in models:

    # Load a different sentence transformer model
    if model == 'XLM_Roberta':
        embedding_model = SentenceTransformer("xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    elif model == 'Specter2':
        embedding_model = SentenceTransformer("allenai/specter2_base")
    elif model == 'MiniLm12':
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    # Define dimensions for umap
    umap_model = umap.UMAP(
        n_neighbors=30, 
        min_dist=0.0, 
        n_components=10, 
        metric='cosine', 
        random_state=42
    )
    
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=15, 
        min_samples=1, 
        metric='euclidean',
        prediction_data=True  # Enable prediction data
    )
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(2, 3)  # Use bigrams and trigrams
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        verbose=True
    )
    
    # list of dataframes to store topic information
    tp_info = {
        'pro': pd.DataFrame(),
        'reg': pd.DataFrame(),
        'sci': pd.DataFrame()
    }

    for source in sources:

        df_file1 = pd.read_csv(f'data/{source}_media_cleaned_eng.csv')

        df_text_column = df_file1['Content']

        topic_model.fit_transform(df_text_column)

        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[
            [
                'Topic', 
                'Count', 
                'Name'
                ]
        ]

        # add column for the source
        topic_info['Source'] = source
        
        tp_info[source] = topic_info

    # Merge the DataFrames
    merged_df = pd.concat([tp_info['pro'], tp_info['reg'], tp_info['sci']], ignore_index=True)

    merged_df.to_csv(f'Bertopicmodel/{model}_topic_info.csv', index=False)

    print("")
    print(f"BERTopic topic info for {model} saved.")
    print("")
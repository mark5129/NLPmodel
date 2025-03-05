
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from time import time

start_time = time()

print(time())
# Load the XLM-Roberta model
model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

print(f"Time taken to load the model: {time() - start_time} seconds")
# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans


# Load pre-trained multilingual BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Example text
text = "Your example text here"

# Tokenize input
inputs = tokenizer(text, return_tensors='pt', max_length=512, show_progress_bar=True, truncation=True, padding='max_length')

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings
embeddings = outputs.last_hidden_state

# Perform topic modeling (example using KMeans)

# Flatten embeddings for clustering
embeddings_flat = embeddings.view(embeddings.size(0), -1).numpy()

# Fit KMeans
num_topics = parameters['num_topics']
kmeans = KMeans(n_clusters=num_topics, random_state=0).fit(embeddings_flat)

# Get topic predictions
topics = kmeans.predict(embeddings_flat)
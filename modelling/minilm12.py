# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

from sentence_transformers import SentenceTransformer
import time
import pandas as pd

# Step 1: Load the CSV OR XLSX file
df = pd.read_csv('pro_media_cleaned.csv') # Your csv file name
#df = pd.read_excel ('TEXT.xlsx')

# Step 2: Prepare the text data
text_data = df['Full text'][1].tolist()

# Step 3: Generate embeddings with progress bar and timer
#model = SentenceTransformer('all-MiniLM-L6-v2') ## English model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') ## Multilingual model
# Step 3: Generate embeddings
start_time = time.time()

embeddings = model.encode(text_data, show_progress_bar=True, batch_size=32)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Embeddings generated in {elapsed_time:.2f} seconds.")
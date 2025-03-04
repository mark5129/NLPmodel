# load parameters from yaml file.
import yaml
with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)
    
# modelling/specter2.py

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns


def Specter2Model(text_column, current_id, doc_type):
    """
    Generates embeddings using Specter2 for scientific literature comparison.

    Parameters:
    text_column (pd.Series): Text data (scientific/media articles).
    current_id (str): Unique ID for saving outputs.

    Returns:
    embeddings (torch.Tensor): Specter2 embeddings for all documents.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")


    embeddings_list = []

    # Generate embeddings for each document
    for text in text_column:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings_list.append(embeddings.flatten())

    # Save embeddings to CSV
    output_dir = 'outputs/Specter2_op'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings.to_csv(os.path.join(output_dir, f'{current_id}_{doc_type}_Specter2_embeddings.csv'), index=False)
    print(f'Specter2 embeddings saved for ID {current_id}')

    return df_embeddings


#if parameters['train_specter2'] == True:

    #Specter2Model(reg_text_column, current_id)
    #Specter2Model(pro_text_column, current_id)
    #print('Specter2 embeddings are generated.')

#else:
    #print('Model training is turned off in parameters.yaml')
    

#if __name__ == "__main__":
    # Dummy example data for testing
    import pandas as pd

    # Example test data
    sample_texts = pd.Series([
        "This is an example of a scientific article discussing renewable energy.",
        "Another article exploring the challenges and opportunities of energy islands."
    ])

    # Provide a test ID for saving outputs
    test_current_id = "test_run"

    # Call the function
    Specter2Model(sample_texts, test_current_id)
    print("Specter2 model ran successfully with test data.")

# Define the correct file path
file_path = os.path.join("outputs/Specter2_op", "1285842042_Specter2_embeddings.csv")

# Load the embeddings using the correct path
embeddings = pd.read_csv(file_path)

# Compute cosine similarity between first two embeddings
similarity = cosine_similarity([embeddings.iloc[0]], [embeddings.iloc[1]])

# Print the similarity score
print(f"Cosine Similarity: {similarity[0][0]}")

# Reduce dimensionality to 2D
umap_2d = umap.UMAP(n_components=2, random_state=42)
reduced_2d = umap_2d.fit_transform(embeddings)

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_2d[:, 0], y=reduced_2d[:, 1])
plt.title("Specter2 Embeddings (2D UMAP)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()

# find labels for the data to color the plot

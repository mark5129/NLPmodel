import pandas as pd
import os

from modelling.minilm12 import MiniLM12
from modelling.Specter2Actually import Specter2ActuallyModel
from modelling.XLM_Roberta import XLM_Roberta_model

# Input path
resume_path = 'evaluations/developments/cluster_resume.csv'
df = pd.read_csv(resume_path)

# Output base folder
output_folder = 'evaluations/resumeOutputs'
os.makedirs(output_folder, exist_ok=True)

# Text column
summaries = df['Summary']
custom_id = 'resumes'
tag = 'resume_clusters'

# Function wrappers with custom save paths
def MiniLM12_custom(texts):
    embeddings = MiniLM12(texts, custom_id, tag, return_data=True)
    embeddings.to_csv(f'{output_folder}/MiniLM12_resume_embeddings.csv', index=False)
    print("Saved MiniLM12 embeddings.")

def Specter2Actually_custom(texts):
    embeddings = Specter2ActuallyModel(texts, custom_id, tag, return_data=True)
    embeddings.to_csv(f'{output_folder}/Specter2Actually_resume_embeddings.csv', index=False)
    print("Saved Specter2Actually embeddings.")

def XLM_Roberta_custom(texts):
    embeddings = XLM_Roberta_model(texts, custom_id, tag, return_data=True)
    embeddings.to_csv(f'{output_folder}/XLM_Roberta_resume_embeddings.csv', index=False)
    print("Saved XLM-Roberta embeddings.")

# Run models
MiniLM12_custom(summaries)
Specter2Actually_custom(summaries)
XLM_Roberta_custom(summaries)

print("✅ All resume embeddings are saved in evaluations/resumeOutputs/")

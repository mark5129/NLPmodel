import pandas as pd
import ast
import yaml

from collections import Counter

from Titlefinder.pdf_titles_to_csv import pdf_to_sentences
from Titlefinder.Overwrite_titles import overwrite_Titles_draft

environment = 'Prod/'

# Load parameters from YAML file
with open('parameters.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Access parameters
Units, pdf_path = params['Units'], params['pdf_path']

# Create the path for the output
output_path = f"{environment}{'Titlefinder/Outputs/Titles_draft'}"
csv_path = f"{output_path}_{pdf_path.split('/')[-1].replace('.pdf', '.csv')}"

# Scan the pdf to find the titles for each page
pdf_to_sentences(pdf_path, csv_path, Units)


# Create the path for the output
output_path = f"{environment}{'Titlefinder/Outputs/Titles'}"
Page_title_path = f"{output_path}_{pdf_path.split('/')[-1].replace('.pdf', '.csv')}"

overwrite_Titles_draft(csv_path, Page_title_path)


import pdfplumber
import pandas as pd
import nltk

# Download punkt tokenizer if not already downloaded
from nltk.tokenize import sent_tokenize

def pdf_to_sentences(pdf_path, output_csv):
    sentences = []
    
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text from the page
            text = page.extract_text()
            if text:
                # Tokenize text into sentences
                page_sentences = sent_tokenize(text)
                sentences.extend(page_sentences)
    
    # Create a DataFrame with each sentence as a row
    df = pd.DataFrame(sentences, columns=["Sentence"])
    
    # Save to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Extracted sentences saved to {output_csv}")

# Example usage
#pdf_to_sentences('p6-39-arsrejseforsikring-basisplus_dk_03.24.pdf', 'sentences_output.csv')

# Usage
pdf_path = 'Policer/p6-39-arsrejseforsikring-basisplus_dk_03.24.pdf'
csv_path = 'output2.csv'
pdf_to_sentences(pdf_path, csv_path)

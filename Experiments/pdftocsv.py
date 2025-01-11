import pdfplumber
import pandas as pd

def pdf_to_csv(pdf_path, csv_path):
    data = []  # List to store extracted text
    
    # Open the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()  # Extract text from each page
            if text:
                lines = text.split('\n')  # Split text into lines
                data.extend([line.split() for line in lines])  # Split lines into words and add to data list

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    
    # Save to a CSV file
    df.to_csv(csv_path, index=False, header=False)
    print(f"CSV file saved to {csv_path}")

# Usage
pdf_path = 'Policer/104.-arsrejseforsikring_basis-plus_04.2022.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


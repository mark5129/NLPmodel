import pdfplumber
import csv

# Specify the input PDF and output CSV file paths
input_pdf = "example.pdf"
output_csv = "output.csv"

def pdf_to_sentences3(pdf_path, output_csv):
    # Open the PDF
    with pdfplumber.open(pdf_path) as pdf:
        # Create a CSV writer
        with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Loop through all pages in the PDF
            for page_number, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_number}...")

                # Extract tables on the current page
                tables = page.extract_tables()

                # Write each table to the CSV
                for table in tables:
                    for row in table:
                        csv_writer.writerow(row)  # Write row to the CSV file

    print(f"Data has been written to {output_csv}")

pdf_path = 'Policer/p6-39-arsrejseforsikring-basisplus_dk_03.24.pdf'
csv_path = 'output3.csv'
pdf_to_sentences3(pdf_path, csv_path)

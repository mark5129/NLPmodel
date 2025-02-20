import pdfplumber
import csv
from collections import Counter

def extract_header_with_font_sizes(page, Units: int):
    """
    Extract the header and font sizes from a page.

    page: Inputs the selected PDF page

    Units: Declares how much of the page needs to be assesed for the title
    """
    header_area = page.within_bbox((0, 0, page.width, Units))  # Top 100 units
    header_text = header_area.extract_text() if header_area else None
    font_sizes = []

    # remove line breaks
    if header_text:
        header_text = header_text.replace('\n', ' ')

    if header_area:
        for char in header_area.chars:

            # For at genkende undertitler undersøger jeg font størrelsen på hver side i de første 100 karakterer
            font_sizes.append(char['size'])
        
        # Der tælles hvor mange forskellige fontstørrelser der er til stede og hvor mange tegn de har
        font_sizes = Counter(font_sizes)

        # Der isoleres den største font størrelse for kun at bevare overskrifter
        if font_sizes:
            largest_font_size = max(font_sizes)
            header_text = ''.join([char['text'] for char in header_area.chars if char['size'] == largest_font_size])

    return header_text.strip() if header_text else None, font_sizes

# Replace the extract_header function call with extract_header_with_font_sizes
def pdf_to_sentences4(pdf_path, output_csv, Units):
    with pdfplumber.open(pdf_path) as pdf:
        # Create a CSV writer
        with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write the header row
            csv_writer.writerow(["Page Number", "Title", "Font Sizes"])
            
            # Initialize the last seen header
            last_seen_header = None
            last_seen_font_sizes = None
            
            # Loop through all pages in the PDF
            for page_number, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_number}...")
                
                # Extract the header and font sizes for the current page
                header, font_sizes = extract_header_with_font_sizes(page, Units)
                if header:  # If a header exists, update the last seen header and font sizes
                    last_seen_header = header
                    last_seen_font_sizes = font_sizes

                # Write the page data to the CSV
                csv_writer.writerow([page_number, last_seen_header, dict(last_seen_font_sizes)])

#pdf_path = 'Policer/p6-39-arsrejseforsikring-basisplus_dk_03.24.pdf'
#csv_path = 'Dev/Output folder/output4_4.csv'
#pdf_to_sentences4(pdf_path, csv_path)

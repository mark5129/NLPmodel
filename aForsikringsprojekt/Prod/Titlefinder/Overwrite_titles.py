import pandas as pd
import ast

from collections import Counter

# Function to find the largest font size in each row
def largest_font_size_per_row(font_sizes_column):
    largest_font_sizes = []
    for font_sizes in font_sizes_column:
        if pd.notnull(font_sizes):  # Handle potential nulls
            font_dict = ast.literal_eval(font_sizes)  # Safely parse the string dictionary
            if font_dict:  # Check if the dictionary is not empty
                largest_font_size = max(font_dict, key=font_dict.get)
                largest_font_sizes.append(largest_font_size)
    return largest_font_sizes

def overwrite_Titles_draft(csv_path1, csv_path2):

    df = pd.read_csv(csv_path1)
    # Find the largest font size in each row
    largest_font_sizes = largest_font_size_per_row(df['Font Sizes'])

    # Find the most frequent largest font size
    most_frequent_largest_font_size = Counter(largest_font_sizes).most_common(1)[0]

    # Find rows where the most frequent largest font size does not appear
    rows_without_most_frequent_largest_font_size = df[df['Font Sizes'].apply(lambda x: pd.notnull(x) and most_frequent_largest_font_size[0] not in ast.literal_eval(x))]

    # Get the list of row numbers where the most frequent largest font size does not appear
    row_numbers_without_most_frequent_largest_font_size = rows_without_most_frequent_largest_font_size.index.tolist()

    # Replace the title in the rows listed with the title from the row above
    for row_number in row_numbers_without_most_frequent_largest_font_size:
        if row_number > 0:  # Ensure there is a row above
            df.at[row_number, 'Title'] = df.at[row_number - 1, 'Title']

    # Remove the 'Font Sizes' column from the DataFrame
    df.drop(columns=['Font Sizes'], inplace=True)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(csv_path2, index=False)
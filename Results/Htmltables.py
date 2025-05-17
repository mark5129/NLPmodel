import pandas as pd
import ast

clustering = ['AP']
models = ['MiniLm12', 'Specter2', 'XLM_Roberta']

def format_tf_idf_field_as_nested_table(value):
    entries = value.split(" ; ")
    nested_html = """
    <table style='border-collapse: collapse; width: 100%; margin: 0;'>
        <thead>
            <tr>
                <th style='border: 1px solid #999; padding: 4px;'>Term</th>
                <th style='border: 1px solid #999; padding: 4px;'>TF-IDF</th>
                <th style='border: 1px solid #999; padding: 4px;'>Doc Count</th>
                <th style='border: 1px solid #999; padding: 4px;'>Term Count</th>
            </tr>
        </thead>
        <tbody>
    """
    for entry in entries:
        parts = entry.strip().split(", ")
        if len(parts) == 4:
            term, tfidf, doc, word = parts
            nested_html += (
                f"<tr>"
                f"<td style='border: 1px solid #999; padding: 4px;'>{term}</td>"
                f"<td style='border: 1px solid #999; padding: 4px;'>{tfidf}</td>"
                f"<td style='border: 1px solid #999; padding: 4px;'>{doc}</td>"
                f"<td style='border: 1px solid #999; padding: 4px;'>{word}</td>"
                f"</tr>"
            )
    nested_html += "</tbody></table>"
    return nested_html

for cluster in clustering:
    for model in models:
        # Load the naming table
        naming_table = pd.read_csv(f'Results/{cluster}_{model}_merged_naming_clusters.csv')

        with open(f'Results/{cluster}_{model}_Results.html', 'w') as html_file:
            # Global styles for the HTML
            html_file.write("""
            <html>
            <head>
            <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 40px;
                }
                th, td {
                    border: 1px solid #999;
                    padding: 8px;
                    text-align: left;
                    vertical-align: top;
                }
            </style>
            </head>
            <body>
            """)

            html_file.write(f"<h1>{model} Results<h1>\n")

            # Drop 'Topic_terms' column if it exists
            if 'Topic_terms' in naming_table.columns:
                naming_table.drop(columns=['Topic_terms'], inplace=True)
                naming_table.drop(columns=['percentage_of_documents'], inplace=True)
                naming_table.drop(columns=['percentage_limit'], inplace=True)

            # Format the TF_IDF_topic_name column
            if 'TF_IDF_topic_name' in naming_table.columns:
                naming_table['TF_IDF_topic_name'] = naming_table['TF_IDF_topic_name'].apply(format_tf_idf_field_as_nested_table)

            # Write the table header
            html_file.write("<table>\n")
            html_file.write("<tr>")
            for col in naming_table.columns:
                html_file.write(f"<th>{col}</th>")
            html_file.write("</tr>\n")

            # Write the table rows
            for _, row in naming_table.iterrows():
                html_file.write("<tr>")
                for col in naming_table.columns:
                    html_file.write(f"<td>{row[col]}</td>")
                html_file.write("</tr>\n")

            html_file.write("</table>\n")
            html_file.write("</body></html>")

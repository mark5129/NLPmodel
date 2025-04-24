import pandas as pd
import ast

clustering = ['AP', 'hdbscan']
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
        naming_groups = pd.read_csv(f'NewPipeline/clustering_outputs/{cluster}_{model}_merged_naming_groups.csv')
        naming_table = pd.read_csv(f'NewPipeline/clustering_outputs/{cluster}_{model}_merged_naming.csv')

        with open(f'NewPipeline/clustering_outputs/{cluster}_{model}_merged_output.html', 'w') as html_file:
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

            for _, group_row in naming_groups.iterrows():
                group_number = group_row['Group']
                average_score = group_row['Average Score']
                rows = group_row['Rows']
                row_indices = ast.literal_eval(rows)
                group_rows = naming_table.loc[row_indices].copy()

                # Drop 'Topic_terms' column
                if 'Topic_terms' in group_rows.columns:
                    group_rows.drop(columns=['Topic_terms'], inplace=True)

                # Format the TF_IDF_topic_name column
                group_rows['TF_IDF_topic_name'] = group_rows['TF_IDF_topic_name'].apply(format_tf_idf_field_as_nested_table)

                # Write the group header
                html_file.write(f"<h2>Group {group_number} - Average Score: {average_score}</h2>\n")
                html_file.write("<table>\n")
                
                # Write header
                html_file.write("<tr>")
                for col in group_rows.columns:
                    html_file.write(f"<th>{col}</th>")
                html_file.write("</tr>\n")

                # Write data rows
                for _, row in group_rows.iterrows():
                    html_file.write("<tr>")
                    for col in group_rows.columns:
                        html_file.write(f"<td>{row[col]}</td>")
                    html_file.write("</tr>\n")

                html_file.write("</table>\n")

            html_file.write("</body></html>")

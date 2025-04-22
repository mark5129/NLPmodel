import pandas as pd
import ast  # Import ast to safely evaluate the Rows column

models = ['Specter2']  # Add other models if needed

# the naming groups contains columns: Group, Rows, Average Score

def format_tf_idf_field(value):
    entries = value.split(" ; ")
    header = "Term ; TF-IDF ; Doc Count ; Word Count"
    lines = [header]
    for entry in entries:
        parts = entry.strip().split(", ")
        if len(parts) == 4:
            term, tfidf, doc, word = parts
            lines.append(f"{term} ; {tfidf} ; {doc} ; {word}")
    return "<br>".join(lines)

for model in models:
    naming_groups = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_naming_groups.csv')
    naming_table = pd.read_csv(f'NewPipeline/clustering_outputs/{model}_naming.csv')

    with open(f'NewPipeline/clustering_outputs/{model}_output.md', 'w') as md_file:
        for _, group_row in naming_groups.iterrows():
            group_number = group_row['Group']
            average_score = group_row['Average Score']
            rows = group_row['Rows']

            # Parse the Rows column as a list of integers
            row_indices = ast.literal_eval(rows)

            # Write the group header
            md_file.write(f"# Group {group_number} - Average Score: {average_score}\n\n")

            # Filter rows in the naming table based on the indices found in the 'Rows' column of the group row
            group_rows = naming_table.loc[row_indices]

            # Write the table header
            md_file.write("| " + " | ".join(group_rows.columns) + " |\n")
            md_file.write("|" + " --- |" * len(group_rows.columns) + "\n")

            # Write the table rows
            for _, row in group_rows.iterrows():
                row = row.copy()  # Avoid SettingWithCopyWarning
                row['TF_IDF_topic_name'] = format_tf_idf_field(row['TF_IDF_topic_name'])
                md_file.write("| " + " | ".join(map(str, row.values)) + " |\n")

            md_file.write("\n")

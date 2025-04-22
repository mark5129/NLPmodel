import pandas as pd
import ast  # Import ast to safely evaluate the Rows column

models = ['Specter2']  # Add other models if needed

# the naming groups contains columns: Group, Rows, Average Score

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
                md_file.write("| " + " | ".join(map(str, row.values)) + " |\n")

            md_file.write("\n")

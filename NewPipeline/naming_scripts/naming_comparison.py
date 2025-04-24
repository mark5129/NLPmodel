import pandas as pd

clustering = ['AP', 'hdbscan']
models = ['MiniLm12', 'Specter2', 'XLM_Roberta']

for cluster in clustering:
    for model in models:
        naming_table = pd.read_csv(f'NewPipeline/clustering_outputs/{cluster}_{model}_merged_naming_clusters.csv')

        naming_table = naming_table[naming_table['percentage_limit'] > 0.5]

        # Initialize an empty scoring matrix
        num_rows = len(naming_table)
        scoring_matrix = [[0] * num_rows for _ in range(num_rows)]

        # Process each row and calculate scores
        for i, row_i in naming_table.iterrows():
            terms_i = set(term.strip() for term in row_i['Topic_terms'].split(';'))  # Split and clean terms

            for j in range(i + 1, num_rows):  # Only iterate over the upper triangular part
                row_j = naming_table.iloc[j]
                terms_j = set(term.strip() for term in row_j['Topic_terms'].split(';'))  # Split and clean terms

                # Calculate the score
                score = 0
                for term_i in terms_i:
                    if term_i in terms_j:
                        score += 3  # Full term match
                    else:
                        words_i = set(term_i.split())  # Split term into words
                        for term_j in terms_j:
                            words_j = set(term_j.split())  # Split term into words
                            if words_i & words_j:  # Check for word overlap
                                score += 1  # Partial match

                scoring_matrix[i][j] = score  # Store the score in the upper triangular part

        # Convert the scoring matrix to a DataFrame for better visualization
        scoring_df = pd.DataFrame(scoring_matrix, columns=naming_table.index, index=naming_table.index)

        # Save the scoring matrix to a CSV file
        #scoring_df.to_csv(f'NewPipeline/clustering_outputs/{model}_merged_naming_score.csv')

        #print(f"Scoring matrix for model {model} saved to NewPipeline/clustering_outputs/{model}_scoring_matrix.csv")

        # Define groups based on scores higher than 3
        groups = {}
        group_id = 1
        rows_in_groups = set()  # Keep track of rows already assigned to groups

        for i in range(num_rows):
            for j in range(i + 1, num_rows):  # Only consider the upper triangular part
                if scoring_matrix[i][j] > 8:
                    if group_id not in groups:
                        groups[group_id] = set()
                    groups[group_id].add(i)
                    groups[group_id].add(j)
                    rows_in_groups.add(i)
                    rows_in_groups.add(j)
            group_id += 1

        # Add a final group for rows not in any group
        unassigned_rows = set(range(num_rows)) - rows_in_groups
        if unassigned_rows:
            groups['Unassigned clusters'] = unassigned_rows

        # Calculate the average score for each group
        group_averages = {}
        for group_id, rows in groups.items():
            scores = []
            for i in rows:
                for j in rows:
                    if i != j:  # Avoid self-comparison
                        scores.append(scoring_matrix[i][j])
            group_averages[group_id] = sum(scores) / len(scores) if scores else 0

        # Convert groups to a DataFrame for better visualization
        group_assignments = [
            {'Group': group_id, 'Rows': list(rows), 'Average Score': round(group_averages[group_id],2)}
            for group_id, rows in groups.items()
        ]
        group_df = pd.DataFrame(group_assignments)

        # Rename group numbers from 1 to the number of groups, excluding the last group
        group_mapping = {old_id: new_id for new_id, old_id in enumerate(groups.keys(), start=1) if old_id != 'Unassigned clusters'}
        group_df['Group'] = group_df['Group'].map(lambda x: group_mapping.get(x, x))  # Keep 'Unassigned clusters' unchanged

        # Save the group assignments to a CSV file
        group_df.to_csv(f'NewPipeline/clustering_outputs/{cluster}_{model}_merged_naming_groups.csv', index=False)

        print(f"Group assignments for model {model} saved to NewPipeline/clustering_outputs/{model}_merged_group_assignments.csv")



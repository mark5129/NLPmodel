import csv

def process_tf_idf_topic_name(tf_idf_topic_name):
    """Breaks the TF_IDF_topic_name column into subrows."""
    terms = tf_idf_topic_name.split(" ; ")
    subrows = []
    for term in terms:
        parts = term.split(", ")
        if len(parts) == 4:
            subrows.append(parts)
    return subrows

def generate_latex_table(input_csv, output_tex):
    """Reads the CSV file and generates a LaTeX table."""
    with open(input_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_tex, 'w') as texfile:
            texfile.write("\\begin{table}[h!]\n")
            texfile.write("\\centering\n")
            texfile.write("\\begin{tabular}{|c|c|c|c|l|c|c|}\n")
            texfile.write("\\hline\n")
            texfile.write("Cluster & Pro & Reg & Sci & TF-IDF Topics & per. of doc & limit \\\\\n")
            texfile.write("\\hline\n")
            
            for row in reader:
                cluster = row['cluster']
                pro = row['pro']
                reg = row['reg']
                sci = row['sci']
                tf_idf_topic_name = row['TF_IDF_topic_name']
                percentage_of_documents = row['percentage_of_documents']
                percentage_limit = row['percentage_limit']
                subrows = process_tf_idf_topic_name(tf_idf_topic_name)
                
                # Write the first row with cluster, pro, reg, sci
                texfile.write(f"{cluster} & {pro} & {reg} & {sci} & Term (TF-IDF diff, doc count, text count)& {percentage_of_documents} & {percentage_limit} \\\\\n")
                
                # Write subrows for TF-IDF topics
                for subrow in subrows:
                    term, tf_idf, doc_count, text_count = subrow
                    texfile.write(f" &  &  &  & {term} ({tf_idf}, {doc_count}, {text_count}) & &\\\\\n")
                
                texfile.write("\\hline\n")
            
            texfile.write("\\end{tabular}\n")
            texfile.write("\\caption{Generated LaTeX Table}\n")
            texfile.write("\\label{tab:latex_table}\n")
            texfile.write("\\end{table}\n")

models = ['XLM_Roberta', 'MiniLm12', 'Specter2']
for model in models:
# Example usage
    input_csv = f'NewPipeline/clustering_outputs/AP_{model}_merged_naming_clusters.csv'
    output_tex = f'NewPipeline/clustering_outputs/AP_{model}_latex_table.csv'
    generate_latex_table(input_csv, output_tex)

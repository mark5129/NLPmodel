import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "evaluations/outputs/manualrun_merged_embeddings_XLM_Roberta_UMAP.csv"  # Replace with the actual path to your CSV file
data = pd.read_csv(csv_file)

# Extract columns 0 and 1
x = data.iloc[:, 0]
y = data.iloc[:, 1]

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.7, edgecolors='black')  # Remove clustering and color mapping
plt.xlabel('X Coordinate (Column 0)')
plt.ylabel('Y Coordinate (Column 1)')
plt.title('2D Scatter Plot')
plt.grid(True)  # Add grid to the plot
plt.show()

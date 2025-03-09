import numpy as np
import pandas as pd
import os
# list files in the directory
os.listdir('modelling/outputs/BERTopic')
print(os.listdir('modelling/outputs/BERTopic'))
# Directory to save the .npy file

npy_dir = 'modelling/outputs/BERTopic/pro_media_BERTopic_embeddings.npy'

# Directory to save the .csv files
csv_dir = 'modelling/outputs/BERTopic/pro_media_BERTopic_embeddings.csv'

data = np.load(npy_dir)

# Convert the numpy array to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a .csv file
df.to_csv(csv_dir, index=False)

print(f'Converted {npy_dir} to {csv_dir}')
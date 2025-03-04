import csv
import time
import os
import random

def update_header_if_needed(parameters: dict):
    if os.path.exists('outputs/log.csv'):
        with open('outputs/log.csv', mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if rows:
                existing_header = rows[0]
                new_keys = [key for key in parameters.keys() if key not in existing_header]
                if new_keys:
                    updated_header = existing_header + new_keys
                    with open('log.csv', mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(updated_header)
                        writer.writerows(rows[1:])
                    print('Header updated in log.csv')

def log_parameters(parameters: dict):
    # Check if log.csv exists and get the last ID
    if os.path.exists('outputs/log.csv'):
        with open('outputs/log.csv', mode='r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) > 1:
                last_id = int(rows[-1][0])
            else:
                last_id = 0
    else:
        last_id = 0
    
    
    # generate random number with 10 digits
    current_id = random.randint(1000000000, 9999999999)

    # Increment the ID for the current run
    #current_id = last_id + 1

    with open('outputs/log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if last_id == 0:
            writer.writerow([
                'id', 
                'time', 
                'preprocess_data', 
                'train_model', 
                'num_topics', 
                'what_data', 
                'nmf_random_state', 
                'nmf_max_features', 
                'nmf_n_top_words', 
                'train_specter2', 
                'train_lda',
                'train_nmf'
            ])
            
        # Write the current ID, time, and parameters
        writer.writerow([
            current_id, 
            time.ctime(), 
            parameters['preprocess_data'], 
            parameters['train_model'], 
            parameters['num_topics'], 
            parameters['what_data'], 
            parameters['random_state'], 
            parameters['max_features'], 
            parameters['n_top_words'], 
            parameters['train_specter2'], 
            parameters['train_lda'],
            parameters['train_nmf']
        ])
        print('Parameters are saved to log.csv')
    
    return current_id, time.ctime()


def clean_rows(df):
    """
    Clean rows of the dataframe
    :param df: dataframe
    :return: cleaned dataframe
    """
    value = '[No abstract available]'
    df = df[df['Content'] != value]
    
    # remove all rows where the string in the content column contains more than 10000 letters

    df = df[df['Content'].apply(lambda x: len(x) < 15000)]
    
    return df
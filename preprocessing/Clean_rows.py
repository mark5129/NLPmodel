

def clean_rows(df):
    """
    Clean rows of the dataframe
    :param df: dataframe
    :return: cleaned dataframe
    """
    value = '[No abstract available]'
    df = df[df['Content'] != value]
    return df
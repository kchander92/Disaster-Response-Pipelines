import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories DataFrames from specified filepaths
    and merges them on ID column into one DataFrame
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(left=messages, right=categories, on='id')
    
    return df

def clean_data(df):
    '''
    Cleans DataFrame by extracting column names from top row of entries
    and changing data values to binary integer values to show classification
    for each category column
    '''
    
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    df['related'] = df['related'].replace(2, 1)
    
    return df

def save_data(df, database_filename):
    '''
    Creates SQLite database with specified filename and writes Dataframe
    to the database, replacing the data if it already exists.
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_responses', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
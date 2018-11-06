# import libraries
import sys
import requests
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,how='inner',on = 'id')
    return df, categories


def clean_data(df,categories):
    """
    Clean data- IN parameters - df,categories dataframes
    """
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',expand = True)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('(\d+)',expand = None).astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    #verify nan values in df
    print ('Check for nan values in df: ',np.where(np.isnan(categories)))
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join='inner')# used inner join to avoid NaN values and int changing to float
    #Remove duplicates
    # check number of duplicates
    print ('Number of duplicate ids: ',len(df[df.duplicated('id',keep=False)]))
    # drop duplicates
    df.drop_duplicates(['id'],keep='first',inplace=True)
    print ('Number of duplicate ids: ',len(df[df.duplicated('id',keep=False)]))
    df.reset_index(inplace=True) # Reset index of clean dataframe after many operations.
    
    return df


def save_data(df, database_filename):
    #database_filename = './ETLPipelineDatabase.db'
    engine = create_engine('sqlite:///ETLPipelineDatabase.db')
    df.to_sql('ETL_Table', engine,index=False)
    print(database_filename , 'created')

        
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df,categories)
        
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
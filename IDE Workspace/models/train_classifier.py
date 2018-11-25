# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])
import sys
import re
import numpy as np
import pandas as pd
from time import time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score,recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import sqlite3
from sqlalchemy import create_engine
from pathlib import Path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
nltk.download('averaged_perceptron_tagger')
import pickle
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Function to load data from sqllite database which was created by process_data.py script.
    input - database_filepath as user input
    output - 
    X as text messages from the database.
    Y - Categories of messages i.e. multilabel targets w.r.t messages in X
    columns - this out has category names linked with message text in X.
    '''
    # load data from database
    #print ('user provided database filepath : ', database_filepath)
    conn = sqlite3.connect(database_filepath)
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    #engine = create_engine('sqlite:///ETLPipelineDatabase.db')
    df = pd.read_sql('select * from ETLPipeline_Table', con=conn)
    #remove index column to cleanup.
    df.drop(['index'], axis=1, inplace=True)
    X = df['message'].values # text array
    Y = df.iloc[:,4:].values # text array
    columns = df.iloc[:,4:].columns
    #print (df.iloc[:,4:].columns)
    return X, Y, columns


def tokenize(text):
    '''
    This function would apply tokenizer, stopwords, lemmatizer on a given message text.
    In- message text as 'text'
    Out - tokenized form of words in provided text
    '''
    # Remove punctuation characters
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed

def build_model():
    '''
    This function would create an object of GridSearchCV pipeline using tokenizer function using CountVectorizer,
    TfIDTransformer and MultiOutputClassifier(RandomForestClassifier()) as classifier.
    In- None
    Out- GridSearchCV object of type GridSearchCV
    '''
    pipeline1 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1,1),(1, 2)),
        #'vect__max_df': [1.0],
        #'vect__min_df' : [1],
        #'tfidf__norm' : ['l2'],
        #'tfidf__use_idf' : ['True'],
        #'clf__estimator__min_samples_split': [2],
        #'clf__estimator__n_estimators': [100],
        #'clf__estimator__max_features': ['auto'],
        #'clf__estimator__criterion': ['gini'],
        #'clf__estimator__class_weight' : [{0: 1, 1: 1}]
    }
    cv = GridSearchCV(pipeline1, param_grid = parameters, verbose = 1, n_jobs=-1)
    return cv


def display_classification_report(y_test, y_pred, category_names):
    """"Function to print multi-label multioutput metrics using sklearn.metrics.classification_report.
    Docstring-
    Input parameters:
    y_test- y testing set from training and testing split.
    y_pred- predictions results after model prediction.
    df - original clean dataframe with column names to extract 36 column names as target labels.

    Output - print display precision score, recall score & f1-score for target columns
    """
    # Loop to evaluate metrics per column.
    print ('Printing precision score, recall score & f1-score for target columns...............')
    i = 0
    report = []
    for col in category_names:
        #print metrics per column as required.
        report.append(classification_report(y_test[:,i], y_pred[:,i],target_names =[col]))
        i+=1
    print(*report,sep='\n',end='\n')
    print ('..............................................Finished printing precision recall & f1-score for target columns.')

    
def evaluate_model(model, X_test, y_test, category_names):
    '''
    This function would predict and evaluate a given model. It would also print out best_estimator_,best_params_,best_score_ , cv_results_ and scorer_ for GridsearchCV model object.
    In- 
    model- Model object which was used to fit training set.
    X_test- dataset which needs to be predicted on.
    y_test - Multilabel target labels to verify evaluation metrics with.
    category_names - Target category labelnames to verify evaluation metrics.
    
    Out- None.
    This function would print the evaluation metrics on screen
    '''
    print('Predicting.........')
    y_pred = model.predict(X_test)
    #Evaluate using display_classification_report function
    print('Evaluating ...........')
    display_classification_report(y_test, y_pred, category_names)
    print ('best_estimator: {}'.format(model.best_estimator_),
           'best params:{}'.format(model.best_params_),
           'best score: {}'.format(model.best_score_),
           'Cross Validation results: {}'.format(model.cv_results_),
           'Scorer : {}'.format(model.scorer_),
           sep='\n')

    
def save_model(model, model_filepath):
    '''
    This function is used to save a model object used in training to a user defined pickle file.
    In - 
    model - model object.
    model_filepath - file path for pickle 3 file.
    Out - None
    '''
    # open the file for writing
    fileObject = open(model_filepath,'wb')
    # this writes the object a to the
    pickle.dump(model, fileObject)
    # # here we close the fileObject
    fileObject.close()

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
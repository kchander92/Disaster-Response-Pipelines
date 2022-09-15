import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def load_data(database_filepath):
    '''
    database_filepath - path to database (.db file) to query data from
    
    Returns:
    X - input variables, which are lines of text to be processed via TF-IDF
    Y - multiple binary classification variables that show whether or not text is classified
        under a given category
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_responses', con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(Y.columns)
    
    return X, Y, category_names

def tokenize(text):
    '''
    Lemmatizes and cleans text, then returns tokens to be processed and modeled
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds and returns machine learning model pipeline that processes text counts through TF-IDF,
    then runs random forest model to classify data under multiple categories
    '''
    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('moc_rf', MultiOutputClassifier(RandomForestClassifier())),
        ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Calculates predicted classification values using machine learning model, then prints out
    precision, recall and F1-score values for each category in DataFrame table
    '''
    
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    scores = {}
    for col in Y.columns:
        scores[col] = classification_report(Y_test[col], Y_pred[col], output_dict=True)['weighted avg']
    
    print(pd.DataFrame(scores))

def save_model(model, model_filepath):
    pickle.dump(cv, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
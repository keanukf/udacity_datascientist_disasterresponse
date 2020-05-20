# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlalchemy
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from sklearn.externals import joblib

import sys


def load_data(database_filepath):
    """
    Load data from database and create feature and label vectors

    Args:
      database_filepath (str): filepath where Sqlite DB is located
    Returns:
      X (pandas.DataFrame): Disaster Tweets as a dataframe,used as features
      Y (pandas.DataFrame): Disaster Response categories as dataframe. Used as labels
      category_names (list): label names for response categories
    """

    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and Lemmatizes a given text

    Args:
      text (str): Text to tokenize
    Returns:
      list: List of text tokens
    """

    # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build machine learning pipeline for disaster response tweets

    Returns:
      sklearn.model_selection.GridSearchCV: ML model (and pipeline) to build NLP model
                                            using GridSearchCV and DecisionTreeClassifier
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1)),
         ])


    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_features': ['log2', None]
        #'clf__estimator__n_estimators': [50, 100, 200],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i in range(len(y_pred[0,:])):
        print(category_names[i] + ":")
        print(classification_report(Y_test[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    # save the model to disk
    joblib.dump(model, open('{}'.format(model_filepath), 'wb'))


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

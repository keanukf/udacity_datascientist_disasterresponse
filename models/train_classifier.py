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

import joblib

import sys
import time


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
    X = df['message'].iloc[:1000].values # only first 1000 rows selected, for performance reasons
    Y = df.iloc[:1000,4:].values # only first 1000 rows selected, for performance reasons
    category_names = df.iloc[:1000,4:].columns # only first 1000 rows selected, for performance reasons

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and Lemmatizes a given text

    Args:
      text (str): Text to tokenize
    Returns:
      list: List of text tokens
    """

    # remove punctiation
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)

    # tokenize given text
    tokens = word_tokenize(text)
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # lemmatize each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build machine learning pipeline for disaster response tweets

    Returns:
      sklearn.model_selection.GridSearchCV: Multioutput ML model (and pipeline) to build NLP model
                                            using GridSearchCV and DecisionTreeClassifier
    """

    # build pipeline model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, min_df=0.0001)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=DecisionTreeClassifier())),
         ])

    # set parameters for hyperparameter tuning
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)#,
        #'clf__estimator__max_features': ['log2', None],
        #'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # tune hyperparamters using Gridsearch
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance

    Args:
      model (sklearn.model_selection.GridSearchCV): Trained ML model
      X_test (pandas.DataFrame): Test feature set, here these are disaster messages.
      Y_test (pandas.DataFrame): Test label set, here these are multiple disaster categories
      category_names: Labels for predictions
    Prints:
      str: classification preport and accuracy scores for each label
    Returns:
        None
    """

    # classify disaster messages using test set
    Y_pred = model.predict(X_test)

    # performance report for each prediction of category
    for i in range(len(Y_pred[0,:])):
        print(category_names[i] + ":")
        print(classification_report(Y_test[:,i],Y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Save model as python joblib object to disk

    Args:
      model (sklearn.model_selection.GridSearchCV): Trained ML model
      model_dilepath (str): filepath to save pickled ML model
    Returns:
      None
    """

    # save the model to disk
    joblib.dump(model, open('{}'.format(model_filepath), 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print('Model data contains {} rows.'.format(len(X)))

        print('Building model...')
        model = build_model()

        print('Training model...')
        time1 = time.time()
        model.fit(X_train, Y_train)
        time2 = time.time()

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved! Training took {}s'.format(round(time2-time1, 2)))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

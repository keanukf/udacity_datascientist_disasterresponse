# import libraries
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from app.tokenizer import tokenize


NLTK_PACKAGES = ["punkt", "wordnet"]
TRAINING_ROW_LIMIT = 1000


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

    engine = create_engine(f"sqlite:///{database_filepath}", future=True)
    with engine.connect() as connection:
        df = pd.read_sql_table("disaster_messages", connection)

    if TRAINING_ROW_LIMIT:
        df = df.head(TRAINING_ROW_LIMIT)

    X = df["message"]
    Y = (
        df.iloc[:, 4:]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )
    category_names = Y.columns.tolist()

    return X, Y, category_names


def build_model():
    """
    Build machine learning pipeline for disaster response tweets

    Returns:
      sklearn.model_selection.GridSearchCV: Multioutput ML model (and pipeline) to build NLP model
                                            using GridSearchCV and DecisionTreeClassifier
    """

    pipeline = Pipeline(
        steps=[
            (
                "vect",
                CountVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    min_df=2,
                    max_df=0.9,
                ),
            ),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                MultiOutputClassifier(
                    estimator=RandomForestClassifier(n_estimators=50, n_jobs=-1)
                ),
            ),
        ]
    )

    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "clf__estimator__max_depth": [None, 20],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

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
    for idx, label in enumerate(category_names):
        print(f"{label}:")
        print(
            classification_report(
                Y_test.iloc[:, idx],
                Y_pred[:, idx],
                zero_division=0,
            )
        )


def save_model(model, model_filepath):
    """
    Save model as python joblib object to disk

    Args:
      model (sklearn.model_selection.GridSearchCV): Trained ML model
      model_dilepath (str): filepath to save pickled ML model
    Returns:
      None
    """

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        database_path = Path(database_filepath).resolve()
        model_path = Path(model_filepath).resolve()

        print('Loading data...\n    DATABASE: {}'.format(database_path))
        X, Y, category_names = load_data(database_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        print('Model data contains {} rows.'.format(len(X)))

        print('Ensuring required NLTK assets are available...')
        for package in NLTK_PACKAGES:
            nltk.download(package, quiet=True)

        print('Building model...')
        model = build_model()

        print('Training model...')
        time1 = time.time()
        model.fit(X_train, Y_train)
        time2 = time.time()

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        print('Saving model...\n    MODEL: {}'.format(model_path))
        save_model(model, model_path)

        print('Trained model saved! Training took {}s'.format(round(time2-time1, 2)))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

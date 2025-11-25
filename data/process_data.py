# import libraries
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads two datasets and merges them

    Args:
      message_filepath (str): Disaster Tweets messages file path
      categories_filepath (str): Disaster categories file path
    Returns:
      pandas.DataFrame: Merged pandas dataframe of both disaster messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # create a dataframe of the 36 individual category columns
    categories_split = categories["categories"].str.split(";", expand=True)
    first_row = categories_split.iloc[0, :]
    category_colnames = first_row.apply(lambda value: value.split("-")[0])
    categories_split.columns = category_colnames

    for column in categories_split:
        categories_split[column] = (
            categories_split[column]
            .str.split("-")
            .str[-1]
            .astype(int)
            .clip(0, 1)
        )

    categories_split.insert(0, "id", categories["id"])
    df = messages.merge(categories_split, how="inner", on="id")

    return df


def clean_data(df):
    """
    Does data wrangling on merged dataset, returns cleaned dataframe

    Args:
      df (pandas.Dataframe): Uncleaned pandas dataframe after loading datasets
    Returns:
      pandas.Dataframe : Cleaned pandas dataframe object after wrangling
    """

    df = df.drop_duplicates(subset=["id"]).copy()
    category_columns = df.columns[4:]
    df.loc[:, category_columns] = (
        df[category_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )

    return df


def save_data(df, database_filename):
    """
    Saves cleaned data to Sqlite database

    Args:
      df (pandas.Dataframe): Cleaned pandas Dataframe for saving to databe
      database_filename (str): Path to save SqliteDB object
    Returns:
      None
    """

    database_path = Path(database_filename).resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{database_path}", future=True)
    df.to_sql("disaster_messages", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print("Database filepath: ",database_filepath)

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

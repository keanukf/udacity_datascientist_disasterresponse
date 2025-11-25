import json
from pathlib import Path

import joblib
import pandas as pd
import plotly
import plotly.graph_objects as graph_objs
from flask import Flask, render_template, request
from sqlalchemy import create_engine

from app.tokenizer import tokenize  # noqa: F401


BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = BASE_DIR / "data" / "DisasterResponse.db"
MODEL_PATH = BASE_DIR / "models" / "classifier.pkl"

app = Flask(__name__)

# load data
engine = create_engine(f"sqlite:///{DATABASE_PATH}", future=True)
with engine.connect() as connection:
    df = pd.read_sql_table("disaster_messages", connection)

# load model
model = joblib.load(MODEL_PATH)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    avg_label_prob = pd.melt(df,
                             id_vars=['id','message', 'original', 'genre'],
                             var_name='label',
                             value_name='count')\
                             .groupby('label')['count']\
                             .mean()\
                             .sort_values(ascending=False)
    label_names = list(avg_label_prob.index)

    # create visuals
    graphs = [
        {
            # set data for first chart
            'data': [
                graph_objs.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            # set layout for first chart
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            # set data for second chart
            'data': [
                graph_objs.Bar(
                    x=label_names,
                    y=avg_label_prob
                )
            ],
            # set layout for secon chart
            'layout': {
                'title': 'Probability of message (tweet) belonging to specific category',
                'yaxis': {
                    'title': "Probability"
                },
                'xaxis': {
                    'title': "Label"
                },
                'color': 'black'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

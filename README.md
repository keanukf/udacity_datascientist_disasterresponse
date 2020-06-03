# Disaster Response Pipeline  - Udacity Project
In this Udacity Data Science course, I've learned and built on your data engineering skills. In this Machine Learning Pipeline project, I've applied these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

# Repository structure
* app -
  * templates -
    * go.html -
    * master.html -
  * run.py -    
* data -
  * disaster_categories.csv -
  * disaster_messages.csv -
  * DisasterResponse.db -
  * process_data.py -
* models -
  * classifier.pkl - Pickle file with trained ML model
  * train_classifier.py - 
* reqiurements.txt

# Usage
### Instructions for local build:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
    `$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
    `$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:
    `$ python run.py`

3. Go to http://0.0.0.0:3001/

### Instructions for message classification:
Once the app is deployed, you can type in any message in the text field. After typing in a message, the trained model classifies your messages and shows every category your message was assigned to.

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. 3. Go to http://0.0.0.0:3001/

4. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/

The web app works and correctly classifies messages.

![alt text](https://github.com/AkshatBhardwaj1/Disaster-Response-Pipelines/blob/master/IDE%20Workspace/Capture.PNG)

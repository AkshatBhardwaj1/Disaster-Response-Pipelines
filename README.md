# Disaster-Response-Pipelines
Udacity's nano degree project to create ETL and ML pipelines for disaster response webapp.

In ETL pipeline notebook you can find Data loading, data cleaning and data saving code. In ML pipeline notebook, I have used RandomForestClassifier in three ways-
1. By creating a pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting the data with it.
2. By creating pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting data with GridSearchCV and parameters. This model didnt finish after several trials.
3. I removed TFID function and tried fitting data with GridSearchCV using MultiOutputClassifier(RandomForestClassifier()) as classifier and got results.

In the end model is saved in a pickle file called "GridSearchModel.pkl".

# Disaster-Response-Pipelines
Udacity's nano degree project to create ETL and ML pipelines for disaster response webapp.

In ETL pipeline notebook you can find Data loading, data cleaning and data saving code. In ML pipeline notebook, I have used RandomForestClassifier in three ways-
1. By creating a pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting the data with it.
2. By creating pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting data with GridSearchCV and parameters.
3. By removing TFID function and tried fitting data with GridSearchCV using MultiOutputClassifier(RandomForestClassifier()) as classifier.

At the end model is saved in a pickle file called "model1_pickle_file.pkl".
For further use, saved results from this pickle file can be used.


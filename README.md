# Disaster-Response-Pipelines
Udacity's nano degree project to create ETL and ML pipelines for disaster response webapp.

In ETL pipeline notebook you can find Data loading, data cleaning and data saving clean data in sqllite database. In ML pipeline notebook, clean data is fetched from sqllite database and then I have used RandomForestClassifier() in three ways-
1. By creating a pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting the data with it.
2. By creating pipeline with MultiOutputClassifier(RandomForestClassifier()) and fitting data with GridSearchCV and parameters.
3. By removing TFID function and tried fitting data with GridSearchCV using MultiOutputClassifier(RandomForestClassifier()) as classifier.

Running any model with gridsearchCV and TfID takes very long time so I have reduced the parameters to bare minimum to get results.
At the end model is saved in a pickle file called "model1_pickle_file.pkl".
For further use, saved results from this pickle file can be used.

![Alt text](https://github.com/AkshatBhardwaj1/Disaster-Response-Pipelines/blob/master/IDE%20Workspace/Capture.PNG)

Results are as follows-

best_estimator: Pipeline(memory=None,
     steps=[('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 2), preprocessor=None, stop_words=None,
        strip...oob_score=False, random_state=None, verbose=0,
            warm_start=False),
           n_jobs=1))])
best params:{'vect__ngram_range': (1, 2)}
best score: 0.1794025036479533
Cross Validation results: {'mean_fit_time': array([  78.6131297 ,  111.36622723]), 'std_fit_time': array([ 0.1120005,  2.268008 ]), 'mean_score_time': array([ 26.37832721,  27.19716477]), 'std_score_time': array([ 0.34394588,  0.50386717]), 'param_vect__ngram_range': masked_array(data = [(1, 1) (1, 2)],
             mask = [False False],
       fill_value = ?)
, 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}], 'split0_test_score': array([ 0.14674038,  0.17991246]), 'split1_test_score': array([ 0.1483871,  0.1781106]), 'split2_test_score': array([ 0.15230415,  0.18018433]), 'mean_test_score': array([ 0.14914369,  0.1794025 ]), 'std_test_score': array([ 0.00233359,  0.00092018]), 'rank_test_score': array([2, 1], dtype=int32), 'split0_train_score': array([ 0.70864055,  0.66647465]), 'split1_train_score': array([ 0.71132358,  0.66651307]), 'split2_train_score': array([ 0.70613985,  0.66501555]), 'mean_train_score': array([ 0.70870133,  0.66600109]), 'std_train_score': array([ 0.00211669,  0.00069706])}
Scorer : <function _passthrough_scorer at 0x7f6425552598>


Printing precision score, recall score & f1-score for target columns...............
             precision    recall  f1-score   support

    related       0.79      0.53      0.63      1212

avg / total       0.85      0.85      0.84      5209

             precision    recall  f1-score   support

    request       0.90      0.99      0.94      4329

avg / total       0.90      0.90      0.89      5209

             precision    recall  f1-score   support

      offer       1.00      1.00      1.00      5183

avg / total       1.00      1.00      1.00      5209

             precision    recall  f1-score   support

aid_related       0.74      0.95      0.83      3061

avg / total       0.80      0.77      0.76      5209

              precision    recall  f1-score   support

medical_help       0.95      1.00      0.97      4794

 avg / total       0.95      0.95      0.94      5209

                  precision    recall  f1-score   support

medical_products       0.97      1.00      0.98      4953

     avg / total       0.97      0.97      0.96      5209

                   precision    recall  f1-score   support

search_and_rescue       0.98      1.00      0.99      5064

      avg / total       0.98      0.98      0.98      5209

             precision    recall  f1-score   support

   security       0.99      1.00      0.99      5115

avg / total       0.99      0.99      0.98      5209

             precision    recall  f1-score   support

   military       0.98      1.00      0.99      5031

avg / total       0.98      0.98      0.97      5209

             precision    recall  f1-score   support

child_alone       1.00      1.00      1.00      5209

avg / total       1.00      1.00      1.00      5209

             precision    recall  f1-score   support

      water       0.96      1.00      0.98      4911

avg / total       0.96      0.96      0.96      5209

             precision    recall  f1-score   support

       food       0.93      1.00      0.96      4640

avg / total       0.94      0.93      0.92      5209

             precision    recall  f1-score   support

    shelter       0.95      1.00      0.97      4775

avg / total       0.95      0.95      0.94      5209

             precision    recall  f1-score   support

   clothing       0.99      1.00      1.00      5144

avg / total       0.99      0.99      0.99      5209

             precision    recall  f1-score   support

      money       0.98      1.00      0.99      5059

avg / total       0.98      0.98      0.97      5209

                precision    recall  f1-score   support

missing_people       0.99      1.00      1.00      5147

   avg / total       0.99      0.99      0.99      5209

             precision    recall  f1-score   support

   refugees       0.98      1.00      0.99      5045

avg / total       0.98      0.98      0.98      5209

             precision    recall  f1-score   support

      death       0.97      1.00      0.98      4959

avg / total       0.97      0.97      0.96      5209

             precision    recall  f1-score   support

  other_aid       0.91      1.00      0.95      4527

avg / total       0.92      0.91      0.90      5209

                        precision    recall  f1-score   support

infrastructure_related       0.96      1.00      0.98      4862

           avg / total       0.96      0.96      0.95      5209

             precision    recall  f1-score   support

  transport       0.97      1.00      0.98      4962

avg / total       0.97      0.97      0.96      5209

             precision    recall  f1-score   support

  buildings       0.97      1.00      0.98      4936

avg / total       0.97      0.97      0.96      5209

             precision    recall  f1-score   support

electricity       0.99      1.00      0.99      5111

avg / total       0.99      0.99      0.98      5209

             precision    recall  f1-score   support

      tools       1.00      1.00      1.00      5176

avg / total       0.99      1.00      0.99      5209

             precision    recall  f1-score   support

  hospitals       0.99      1.00      1.00      5151

avg / total       0.99      0.99      0.99      5209

             precision    recall  f1-score   support

      shops       1.00      1.00      1.00      5182

avg / total       1.00      1.00      1.00      5209

             precision    recall  f1-score   support

aid_centers       0.99      1.00      1.00      5148

avg / total       0.99      0.99      0.99      5209

                      precision    recall  f1-score   support

other_infrastructure       0.97      1.00      0.99      4975

         avg / total       0.97      0.97      0.97      5209

                 precision    recall  f1-score   support

weather_related       0.83      0.98      0.90      3740

    avg / total       0.86      0.84      0.83      5209

             precision    recall  f1-score   support

     floods       0.94      1.00      0.97      4759

avg / total       0.94      0.94      0.93      5209

             precision    recall  f1-score   support

      storm       0.94      1.00      0.97      4688

avg / total       0.94      0.94      0.92      5209

             precision    recall  f1-score   support

       fire       0.99      1.00      1.00      5166

avg / total       0.99      0.99      0.99      5209

             precision    recall  f1-score   support

 earthquake       0.95      1.00      0.97      4735

avg / total       0.95      0.95      0.94      5209

             precision    recall  f1-score   support

       cold       0.99      1.00      0.99      5109

avg / total       0.99      0.99      0.98      5209

               precision    recall  f1-score   support

other_weather       0.96      1.00      0.98      4914

  avg / total       0.96      0.96      0.96      5209

               precision    recall  f1-score   support

direct_report       0.88      0.99      0.93      4213

  avg / total       0.89      0.88      0.87      5209

..............................................Finished printing precision recall & f1-score for target columns.

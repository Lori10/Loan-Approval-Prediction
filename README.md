# Loan Approval Prediction

## Table of Content
  * [Business Problem Statement](#Business-Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Cleaning](#Data-Cleaning)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
  * [Other used Techniques](#Other-Used-Techniques)
  * [Demo](#demo)
  * [Run project in your local machine](#Run-the-project-in-your-local-machine)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Business Problem Statement
With the enhancement in the banking sector lots of people are applying for bank loans but the bank has its limited assets which it has to grant to limited people only, so finding out to whom the loan can be granted which will be a safer option for the bank is a typical process. The main objective of this project is to predict whether assigning the loan to particular person will be safe or not using a dataset of more than 100000 records.

## Data
Data Source : Private Data Source

Dataset is imbalanced (81%-19%)

## Used Libraries and Resources
**Python Version** : 3.6

**Libraries** : sklearn, pandas, numpy, matplotlib, seaborn, flask, json, pickle

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Cleaning
The interesting part of this project is that the dataset is highly imbalanced. To prepare the data for training I have performed these feature engineering steps :

* Split data into training and test before any data preprocessing approach in order to avoid data leakage. 
* Remove columns that are not relevant for prediction like Row ID.
* Removed some unwanted / strange values like 'NONE' and 'ANY' in column Home_Ownership.
* Dropped records with missing values since the percentage of missing values in features is very low.
* Used One-Hot encoding technique te encode nominal categorical features with low amount of categories. For features with high amount of categories like 'State' I used mean encoding approach.
* Feature Selection is done using Recursive Elemination with Cross Validation. Using this approach best features for prediction will be selected automatically.


## Model Building and Tuning

* The approaches I used to handle imbalanced data are : 1. Tuning XGBoost setting scale_pos_weight attribute = nr of negative instances / nr of positive instances; 2. Tuning Random Forest including class_weight attribute; 3. Using the combination of over- and undersampling with SMOTETomek; 4. Tuning EasyEnsembleClassifier
* Evaluated the models using AUC Score and confusion matrix because of the imbalanced data. The performance metric used to select the best model is AUC Score.
* When tuning the models using Cross Validation, I used sklearn pipeline including over-undersampling and hyperparameter tuning in order to avoid data leakage. This means that each fold of Cross Validation is resampled, model is trained using training set and then tested on test set which is not resampled.
* Hyperparameter Tuning  is done using RandomizedSearchCV.
* I evaluated each ML model using training score, cross validation mean score, cross validation scores, test score to get a better understanding about the model performances. The best model is selected using the test score.
* Every information about different performance metrics of default (model with default hyperparameters) and tuned models training is stored in a csv file.
* In Cross Validation Scores we expect higher accuracy than in test score because there is some data leakage during feature engineering.
* It seems like using SMOTETomek does not give good results in this case. Test score is improved after tuning XGBoost with scale_pos_weight and RandomForest with class_weight. EasyEnsembleClassifier gives similar performance after tuning. The best model that I got after tuning is XGBoost with scale_pos_weight.


| Model Name                    | Deafult Model Test Score |Default Model Training Score | Default Model CV Score | Tuned Model Test Score | Tuned Model Training Score | Tuned Model CV Score | 
|:-----------------------------:|:------------------------:|:---------------------------:|:----------------------:|:----------------------:|:--------------------------:|:------------------------:|
|XGBoost with scale_pos_weight  |     0.6433               |     0.7250                  |         0.6947         |      0.6525            |           0.6647           |     0.7081               |
|RandomForest with Class Weight |     0.5110               |     1.0                     |         0.6975         |      0.6094            |           0.6961           |     0.7067               |
|Easy Ensemble                  |     0.6534               |     0.7184                  |         0.7067         |      0.6328            |           0.7011           |  0.7072               |
|XGBoost SMOTETomek             |     0.5298               |     0.5549                  |         0.6948         |      0.5143            |           0.5173           |  0.7045               |

## Other Used Techniques

* Object oriented programming is used to build this project in order to create modular and flexible code.
* Built a client facing API (web application) using Flask.
* A retraining approach is implemented using Flask.
* Using Logging every information about data cleaning und model training HISTORY (since we may train the model many times using retraining approach) is stored is some txt files and csv files for example : the amount of missing values for each feature, the amount of records removed after dropping the missing values, best selected features, model accuracies and errors etc.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Loan-Approval-Prediction/blob/main/PyCharm%20Code%20Project/demo_photo.jpg "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory in PyCharm ((PyCharm Project Code folder)  and create a Python Interpreter using Conda Environment : Settings - Project : Project Code Pycharm - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
3. Run the following command in the terminal to install the required packages and libraries : pip install -r requirements.txt
4. Run the file app.py by clicking Run and open the API that shows up in the bottom of terminal.


## Directory Tree 
```
 ├── Project Code PyCharm├── static 
                             ├── css
                                 ├── styles.css
                         ├── templates
                         │   ├── home.html
                         ├── File_Operation
                             ├── FileOperation.py
                         ├── Functions
                             ├── functions.py
                         ├── Logs
                             ├── DataPreprocessing_Log.txt
                             ├── ModelTraining_Log.txt
                             ├── Prediction_Log.txt
                         ├── ModelTraining
                             ├── trainingModel.py
                         ├── Training_FileFromDB
                             ├── dataset.csv
                         ├── application_logging
                             ├── logger.py
                         ├── best_model_finder
                             ├── modelTuning.py
                         ├── data_ingestion
                             ├── data_loader.py
                         ├── data_preprocessing
                             ├── preprocessing.py
                         ├── models
                             ├── EasyEnsembleClassifier
                                 ├── EasyEnsembleClassifier.sav
                         
                         ├── EasyEnsembleClassifier_ConfusionMatrices
                             ├── DefaultModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── DefaultModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── TunedModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── TunedModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                                                    
                          ├── RandomForestClassifier_ConfusionMatrices
                             ├── DefaultModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── DefaultModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── TunedModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                             ├── TunedModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                                      
                          ├── XGBoostClassifier_ConfusionMatrices
                              ├── DefaultModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                              ├── DefaultModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                              ├── TunedModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                              ├── TunedModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                                  
                           ├── XGBoostOverUnderSampling_ConfusionMatrices
                               ├── DefaultModel_TestData_ConfusionMatrix
                                   ├── confusion_matrix.jpg
                               ├── DefaultModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                               ├── TunedModel_TestData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                               ├── TunedModel_TrainingData_ConfusionMatrix
                                  ├── confusion_matrix.jpg
                         
                           ├── app.py
                           ├── encoded_features.json
                           ├── model_infos.csv
                           ├── demo_photo.jpg
                           ├── dic_state.json
                           ├── nan_values.csv
                           ├── Training Infos.ipynb
                           ├── requirements.txt
```



## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Banglore-House-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use any other approach to handle imbalanced data like BalancedBaggigClassifier, BalancedRandomForestClassifier etc.
* Optimize Flask app.py Front End

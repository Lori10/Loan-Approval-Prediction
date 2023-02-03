
# Loan Approval Prediction

## Table of Content
  * [Business Problem Statement](#Business-Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Model Building and Tuning](#Model-Building-and-Tuning)
  * [Techniques for handling imbalanced data](#Techniques-for-handling-imbalanced-data)
  * [Other used techniques](#Other-used-techniques)
  * [Demo](#demo)
  * [Run project in your local machine](#Run-the-project-in-your-local-machine)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Business Problem Statement
This project consists of predicting using different machine learning models if the loan application will get approved or not based on informations like financial information, requested loan amount etc using a highly imbalanced dataset of more than 100000 records

## Data
Data Source : Privat Source.

## Used Libraries and Resources
**Python Version** : 3.6

**Libraries** : sklearn, pandas, numpy, matplotlib, seaborn, flask, json

**References** : https://towardsdatascience.com/, https://machinelearningmastery.com/


## Data Preprocessing

* Based on domain knowledge I removed columns which are not relevant for prediction.
* Handling missing and unwanted/strange values in the features. 
* Encoding categorical features.
* Feature Selection using Recursive Elemination with cross validation.

## Model Building and Tuning

* The ML Models I trained are : XGBoost, Random Forest, Easy Ensemble Classifier.
* The performance metric used to select the best model is AUC Score since we have imbalanced data (accuracy is not a good choice).
* Hyperparameter Tuning  is done using RandomizedSearchCV.
* Sklearn Pipelines are used during cross validation to only upsample/downsample each training fold (not the test fold) in order to avoid data leakage.
* I evaluated each ML model using training score, cross validation mean score, test score to get a better understanding about the model performances. The best model is selected using the test score.
* The best model I got from model tuning is XGBoost with an AUC score of 0.65.

| Model Name        | Deafult Model Test Score |Default Model Training Score | Default Model CV Score | Tuned Model Test Score | Tuned Model Training Score | Tuned Model CV Score | 
|:-----------------:|:------------------------:|:---------------------------:|:----------------------:|:----------------------:|:--------------------------:|:---------------------:|
|XGBoost            |     0.64                 |     0.72                    |         0.69           |      0.65              |              0.66          |     0.70          |
|Random Forest      |     0.51                 |      1.0                    |         0.70           |      0.61              |           0.69             |     0.71           
|EasyEnsemble       |     0.65                 |     0.72                    |         0.71           |      0.63              |           0.70             |  0.71              |


## Techniques for handling imbalanced-data

* Tuning XGBoost including scale_pos_weight parameter which handles imbalanced data.
* Tuning Random Forest including class_weight parameter. Using class_weight we can assign weights to the minority and majority class. Higher weight for minority class means that the error (in the cost function) when missclassifying data from the minority class will be higher.
* Tuning EasyEnsemble which is a bag of balanced boosted learners. This classifier is an ensemble of AdaBoost learners trained on different balanced bootstrap samples. The balancing is achieved by random under-sampling.
* Resample the dataset using SMOTEtomek which is a combination of oversampling and undesampling and tuning XGBoost.

## Other used techniques
* Object oriented programming is used to build this project in order to create modular and flexible code.
* Built a client facing API (web application) using Flask.
* A retraining approach is implemented using Flask framework.
* Using Logging every information about data cleaning und model training HISTORY (since we may train the model many times using retraining approach)  is stored is some txt files and csv files for example : the amount of missing values for each feature, the amount of records removed after dropping the missing values and outliers, the amount of at least frequent categories labeled with 'other' during encoding, the dropped constant features, highly correlated independent features, which features are dropping during handling multicolleniarity, best selected features, model accuracies and errors etc.

## Demo

This is how the web application looks like : 


![alt text](https://github.com/Lori10/Banglore-House-Price-Prediction/blob/master/Project%20Code%20Pycharm/demo_image.jpg "Image")



## Run the project in your local machine 

1. Clone the repository
2. Open the project directory (PyCharm Project Code folder) in PyCharm  and create a Python Interpreter using Conda Environment : Settings - Project : Project Code Pycharm - Python Interpreter - Add - Conda Environment - Select Python Version 3.6 - 
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
                             ├── RandomForestRegressor
                                 ├── RandomForestRegressor.sav
                         ├── app.py
                         ├── encoded_features.json
                         ├── model_infos.csv
                         ├── multicolleniarity_heatmap.jpg
                         ├── nan_values.csv
                         ├── Training Infos.ipynb
                         ├── requirements.txt
```



## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an [issue](https://github.com/Lori10/Banglore-House-Price-Prediction/issues) here by including your search query and the expected result

## Future Scope

* Use other ML Estimators
* Try other feature engineering approaches to get a possible higher model performance
* Optimize Flask app.py Front End

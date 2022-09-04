from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix, classification_report



class ModelTuner:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def train_model(self, model, X_train, y_train):
        try:
            self.logger_object.log(self.file_object, 'Entered the function train_model')
            model.fit(X_train, y_train)

            self.logger_object.log(self.file_object, 'Function train_model Completed Successfully! Exited this function.')
            return model
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function train_model. Error Messaeg : ' + str(e))

    def evaluate_model_classification(self, model, path, model_name, X, y):
        try:
            self.logger_object.log(self.file_object, 'Entered function evaluate_model_classification.')

            pred = model.predict(X)
            score = roc_auc_score(y,
                                  pred)  # scoring is chosen based on the problem statement. In this case we use auc score.

            parent_path = model_name + '_ConfusionMatrices/'
            if os.path.exists(parent_path) == False:
                os.mkdir(parent_path)

            model_path = parent_path + path
            if os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=False, onerror=None)
                os.mkdir(model_path)
            else:
                os.mkdir(model_path)

            self.logger_object.log(self.file_object, f'Classification Report : {classification_report(y, pred)}')
            plt.figure(figsize=(15, 12))
            plot_confusion_matrix(model, X, y)
            plt.savefig(model_path + 'confusion_matrix.jpg')

            self.logger_object.log(self.file_object, 'Function evaluate_model_classification Completed Successfully. Exited this function')

            return score

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function evaluate_model_classification. Error Message : ' + str(e))



    def evaluate_model_cross_validation(self, model, X_train, y_train, scoring, cv=5, verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered function evaluate_model_cross_validation.')
            scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1, verbose=verbose)

            self.logger_object.log(self.file_object, 'Function evaluate_model_cross_validation Completed Successfully.Exited this function.')

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function evaluate_model_cross_validation. Error Message : ' + str(e))
        return scores

    def HyperparameterTuning_RandomizedSearchCV(self, X_train, y_train, model, params, scoring, n_iter=20, cv=5,
                                                verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered the function HyperparameterTuning_RandomizedSearchCV.')

            search = RandomizedSearchCV(model, params, n_iter=n_iter, scoring=scoring,
                                        cv=cv, n_jobs=-1, verbose=verbose, random_state=1)

            search.fit(X_train, y_train)
            model.set_params(**search.best_params_)

            dic = {'tuned_model': model,
                   'best_hyperparameters': search.best_params_,
                   'best_cv_score': search.best_score_}

            self.logger_object.log(self.file_object, 'Function HyperparameterTuning_RandomizedSearchCV Completed Successfully! Exited this function.')
            return dic

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function HyperparameterTuning_RandomizedSearchCV. Error message : ' + str(e))



    def best_single_model_RandomizedSearchCV(self, X_train, y_train, X_test, y_test, model_name, defaultModel, params,
                                             cv_scoring, cv_kfold=5, cv_n_iter=20):

        try:
            self.logger_object.log(self.file_object, 'Entered function best_single_model_RandomizedSearchCV. ')

            # create copies of default model object to avoid unwanted changes
            def_model = clone(defaultModel)
            tun_model = clone(defaultModel)

            # 1. Default Model
            # A) Evaluate model using cross validation
            default_model_cv_scores = self.evaluate_model_cross_validation(def_model, X_train, y_train, scoring=cv_scoring,
                                                                      cv=cv_kfold)
            default_model_cv_mean_score = np.mean(default_model_cv_scores)

            # B) Train model with default hyperparameters
            default_model = self.train_model(def_model, X_train, y_train)

            # C) evaluate the model on test set and training set
            self.logger_object.log(self.file_object, f'Default {model_name} Model Performance on TEST SET : ')

            # in case of regression we should remove the line about the confusion matrix path and change the evaluation model function
            Path_DefaultModel_TestData = 'DefaultModel_TestData_ConfusionMatrix/'
            default_model_test_score = self.evaluate_model_classification(default_model, Path_DefaultModel_TestData,
                                                                     model_name, X_test, y_test)

            self.logger_object.log(self.file_object, f'Default {model_name} Model Performance on TRAINING SET : ')
            # in case of regression we should remove the line about the confusion matrix path and change the evaluation model function
            Path_DefaultModel_TrainingData = 'DefaultModel_TrainingData_ConfusionMatrix/'
            default_model_training_score = self.evaluate_model_classification(default_model, Path_DefaultModel_TrainingData,
                                                                         model_name,
                                                                         X_train, y_train)

            # 2. Tuned Model
            # If we want to use Optuna to tune Hyperparameters we call model_optuna function
            # A) Tune the model
            dic = self.HyperparameterTuning_RandomizedSearchCV(X_train=X_train, y_train=y_train, model=tun_model,
                                                          params=params, n_iter=cv_n_iter, scoring=cv_scoring,
                                                          cv=cv_kfold, verbose=True)
            tunedModel = dic['tuned_model']
            best_params = dic['best_hyperparameters']
            tuned_model_cv_mean_score = dic['best_cv_score']

            # B) Calculate the cross validation scores and mean on training set
            tuned_model_cv_scores = self.evaluate_model_cross_validation(tunedModel, X_train, y_train, scoring=cv_scoring,
                                                                    cv=cv_kfold)

            # B) Train the tuned model on training set
            tuned_model = self.train_model(tunedModel, X_train, y_train)

            # C) Evaluate the model on test set and training set
            self.logger_object.log(self.file_object, f'Tuned Model {model_name} Performance on TEST SET : ')
            # in case of regression we should remove the line about the confusion matrix path and change the evaluation model function
            Path_TunedModel_TestData = 'TunedModel_TestData_ConfusionMatrix/'
            tuned_model_test_score = self.evaluate_model_classification(tuned_model, Path_TunedModel_TestData, model_name,
                                                                   X_test, y_test)

            self.logger_object.log(self.file_object, f'Tuned Model {model_name} Performance on TRAINING SET : ')
            # in case of regression we should remove the line about the confusion matrix path and change the evaluation model function
            Path_TunedModel_TrainingData = 'TunedModel_TrainingData_ConfusionMatrix/'
            tuned_model_training_score = self.evaluate_model_classification(tuned_model, Path_TunedModel_TrainingData,
                                                                       model_name, X_train, y_train)

            # Find the best model with best score
            if tuned_model_test_score >= default_model_test_score:
                best_model_name = 'Tuned Model'
                best_model_object = tuned_model

            else:
                best_model_name = 'Default Model'
                best_model_object = default_model

            model_infos = {'Model Name': model_name,
                           'Default Model Object': default_model,
                           'Default Model Test Score': default_model_test_score,
                           'Default Model Training Score': default_model_training_score,
                           'Default Model CV Mean Score': default_model_cv_mean_score,
                           'Default Model CV Scores': default_model_cv_scores,
                           'Tuned Model Object': tuned_model,
                           'Best Hyperparameters': best_params,
                           'Tuned Model CV Mean Score': tuned_model_cv_mean_score,
                           'Tuned Model CV Scores': tuned_model_cv_scores,
                           'Tuned Model Test Score': tuned_model_test_score,
                           'Tuned Model Training Score': tuned_model_training_score,
                           'Final Best Model Name': best_model_name,
                           'Final Best Test Score': max(tuned_model_test_score, default_model_test_score),
                           'Final Best Model Object': best_model_object}

            self.logger_object.log(self.file_object, 'Function best_single_model_RandomizedSearchCV Completed Successfully. Exited this function.')
            return model_infos


        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function best_single_model_RandomizedSearchCV. Error message : ' + str(e))

    def best_model_OutOfManyModels_RandomizedSearchCV(self, X_train, y_train, X_test, y_test, cv_scoring, cv_kfold=5,
                                                      cv_niter=20, verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered the function best_model_OutOfManyModels_RandomizedSearchCV.')

            # model_df is dataframe which containss information about training and testing of each model
            model_df = pd.DataFrame()

            # dic is a dictionary which contains the best model object found for each ML algorithms and its score
            dic = {}

            # 1. Attempt : XGBoost Model using scale_pos_weigt attribute
            s = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            default_xgboost = XGBClassifier(random_state=1, scale_pos_weight=s, n_jobs=-1)

            params_xgboost = {'n_estimators': [25, 50, 75, 100],
                              'learning_rate': np.linspace(0.01, 0.2, 6)[:-1][1:],
                              'max_depth': [2, 5, 7, 10],
                              'subsample': np.linspace(0.5, 1, 6)[:-1][1:],
                              'colsample_bytree': np.linspace(0.5, 1, 6)[:-1][1:],
                              'min_child_weight': [300, 400, 500]}

            print('XGBoost Tuning')
            xgboost_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                                 model_name='XGBoostClassifier',
                                                                 defaultModel=default_xgboost, params=params_xgboost,
                                                                 cv_scoring='roc_auc', cv_kfold=5, cv_n_iter=30)

            model_df = model_df.append(xgboost_infos, ignore_index=True)

            # Get Best XGBoost Model Object and its Test score
            best_xgboost_object = xgboost_infos['Final Best Model Object']
            best_xgboost_test_score = xgboost_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_xgboost_object] = best_xgboost_test_score

            # Attempt 2 : Random Forest using class_weight attribute
            default_randomforest = RandomForestClassifier(random_state=1, n_estimators=400)

            weights = np.linspace(0.05, 0.45, 7)[:-1][1:]

            params_randomforest = {'class_weight': [{0: x, 1: 1 - x} for x in weights],
                                   'max_features': ['sqrt', 'log2', None],
                                   'min_samples_split': [50, 100, 150, 200],
                                   'min_samples_leaf': [5, 20, 40, 50],
                                   'criterion': ['gini', 'entropy'],
                                   'bootstrap': [True, False]}

            print('Random Forest Tuning')
            randomforest_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                                      model_name='RandomForestClassifier',
                                                                      defaultModel=default_randomforest,
                                                                      params=params_randomforest,
                                                                      cv_scoring='roc_auc', cv_kfold=5, cv_n_iter=30)

            # Add the RandomForest Infos to the dataframe
            model_df = model_df.append(randomforest_infos, ignore_index=True)
            # Get Best RandomForest Model Object and its test score
            best_randomforest_object = randomforest_infos['Final Best Model Object']
            best_randomforest_test_score = randomforest_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_randomforest_object] = best_randomforest_test_score

            # 3. Attempt : Easy Ensemble Classifier
            default_easyensemble = EasyEnsembleClassifier(random_state=1, base_estimator=XGBClassifier(random_state=1),
                                                          n_jobs=-1)

            params_easyensemble = {'sampling_strategy': np.linspace(0.1, 1, 6)}

            print('Easy Ensemble Tuning')
            knn_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                             model_name='EasyEnsembleClassifier',
                                                             defaultModel=default_easyensemble,
                                                             params=params_easyensemble,
                                                             cv_scoring='roc_auc', cv_kfold=5, cv_n_iter=30)

            # Add the knn infos to the dataframe
            model_df = model_df.append(knn_infos, ignore_index=True)
            # Get Best KNN Model Object and its test score
            best_knn_object = knn_infos['Final Best Model Object']
            best_knn_test_score = knn_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_knn_object] = best_knn_test_score

            # 4. Attempt : Combination of Oversampling and Undersampling using SMOTETomek

            default_ou = Pipeline([('smotetomek', SMOTETomek(random_state=1, n_jobs=-1)),
                                   ('estimator', XGBClassifier(random_state=1, n_jobs=-1))])

            params_ou = {'smotetomek__sampling_strategy': np.linspace(0.2, 1, 6),
                         'estimator__n_estimators': [50, 100, 300, 500, 700],
                         'estimator__learning_rate': np.linspace(0.01, 0.2, 6)[:-1][1:],
                         'estimator__max_depth': [3, 5, 7, 10],
                         'estimator__subsample': np.linspace(0.5, 1, 6)[:-1][1:],
                         'estimator__colsample_bytree': np.linspace(0.5, 1, 6)[:-1][1:],
                         'estimator__min_child_weight': [1, 15, 50, 100, 200]}

            ou_infos = self.best_single_model_RandomizedSearchCV(X_train, y_train, X_test, y_test,
                                                            model_name='XGBoostOverUnderSampling',
                                                            defaultModel=default_ou, params=params_ou,
                                                            cv_scoring='roc_auc', cv_kfold=5, cv_n_iter=30)

            # Add the knn infos to the dataframe
            model_df = model_df.append(ou_infos, ignore_index=True)
            # Get Best KNN Model Object and its test score
            best_ou_object = ou_infos['Final Best Model Object']
            best_ou_test_score = ou_infos['Final Best Test Score']
            # Add the information to the dictionary
            dic[best_ou_object] = best_ou_test_score

            # save information about all models into a csv file and the information about xgboost in separate file if we have used early stopping
            model_df.to_csv('model_infos.csv', index=False)

            # Finding best model out of all models based on test score
            best_model_object = max(dic, key=dic.get)
            best_test_score = max(dic.values())

            try:

                self.logger_object.log(self.file_object,
                    'Function best_model_OutOfManyModels_RandomizedSearchCV Completed Successfully. Exited this function.')
                return best_model_object, str(best_model_object.named_steps['estimator']).split('(')[0], best_test_score

            except Exception as e:
                return best_model_object, str(best_model_object).split('(')[0], best_test_score




        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function best_model_OutOfManyModels_RandomizedSearchCV. Error Message : ' + str(e))
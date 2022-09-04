from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self, data, columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Parameters : The dataframe and a list of column names to remove
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the function remove_columns of the Preprocessor class')

        try:
            new_df = data.drop(columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Function remove_columns of class Preprocessor Completed Successfully. Exited this function.')
            return new_df
        except Exception as e:
            self.logger_object.log(self.file_object,'Error occured in function remove_columns of the Preprocessor class. Error message:  '+str(e))


    def separate_features_label(self, data, target_variable):
        """
            Method Name: separate_features_label
            Description: This method separates the features and labels from a dataset
            Parameters : entire dataset and target variable name
            Output: X dataframe with the features and y label
            On Failure: Raise Exception
        """

        self.logger_object.log(self.file_object, 'Entered the function separate_features_label of class Preprocessor.')

        try:
            X = data.drop(target_variable, axis=1).copy()
            y = data[target_variable].copy()

            self.logger_object.log(self.file_object, 'Function separate_features_label of class Preprocessor Completed Succesfully. Exited this function.')
            return X, y

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function separate_features_label of class Preprocessor. Error Message : ' + str(e))


    def train_test_splitting(self, X, y, test_size, stratify=False):
        """
                    Method Name: train_test_splitting
                    Description: This method split the data into training and test data
                    Parameters : features dataframe, y label, size of test set, stratify=True if we want to obtain percetange of data in classification
                    Output: train and test dataframe
                    On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the function train_test_splitting of Preprocessor class.')

        try:
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    stratify=y, random_state=1)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=1)

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            self.logger_object.log(self.file_object, 'Function train_test_splitting of Preprocessor class Completed Successfully. Exited this function.')
            return train_data, test_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function train_test_splitting of Preprocessor class. Error Message : ' + str(e))



    def drop_missing_values(self, data):
        """
            Method Name: drop_missing_values
            Description: This method drop all rows of dataframe that have at least one missing value
            Parameters : dataset with features
            Output: new dataframe with dropped missing values
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the function drop_missing_values of Preprocessor class.')
        try:
            new_data = data.dropna()
            self.logger_object.log(self.file_object, 'Function drop_missing_values of Preprocessor class Completed Succesfully. Exited this function.')
            return new_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function drop_missing_values of Preprocessor class. Error message : ' + str(e))



    def encode_categorical_features(self, train_data, test_data):
        """
            Method Name: encode_categorical_features
            Description: This method encodes the categorical features
            Parameters : dataset
            Output: new dataframe with all numerical features
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered function encode_categorical_features of Preprocessor class.')
        try:
            # encoding target feature
            train_data['Bad_Loan'] = train_data['Bad_Loan'].map({'BAD': 1, "GOOD": 0})
            test_data['Bad_Loan'] = test_data['Bad_Loan'].map({'BAD': 1, "GOOD": 0})

            # encoding ordinal categorical features
            train_data["Term"] = train_data["Term"].map({'36 months': 36, '60 months': 60})
            test_data["Term"] = test_data["Term"].map({'36 months': 36, '60 months': 60})

            # encoding nominal categorical features
            # using mean encoding technique te encode the feature 'State'
            dic = train_data.groupby('State')['Bad_Loan'].mean().to_dict()
            train_data["State"] = train_data["State"].map(dic)
            test_data["State"] = test_data["State"].map(dic)

            # using one-hot encoding technique to encode features 'Home_Ownership', 'Verification_Status', 'Loan_Purpose'
            new_train_data = pd.get_dummies(train_data, columns=['Home_Ownership', 'Verification_Status', 'Loan_Purpose'])
            new_test_data = pd.get_dummies(test_data, columns=['Home_Ownership', 'Verification_Status', 'Loan_Purpose'])


            self.logger_object.log(self.file_object, 'Function encode_categorical_features of Preprocessor class Completed Succesfully. Exited this function')
            return new_train_data, new_test_data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function encode_categorical_features of Preprocessor class. Error message : ' + str(e))



    def handling_unwanted_values(self, data):
        """
            Method Name: handling_unwanted_values
            Description: This method handles unwanted values in columns
            Parameters : dataset
            Output: new dataframe with dropped unwanted values
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered function handling_unwanted_values of Preprocessor class.')

        try:
            data = data[(data['Home_Ownership'] != 'ANY') & (data['Home_Ownership'] != 'NONE')]

            self.logger_object.log(self.file_object, 'Function handling_unwanted_values of Preprocessor class Completed Successfully. Exited this function.')
            return data

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function handling_unwanted_values of Preprocessor class. Error message : ' + str(e))



    def check_for_missing_values(self, dataset):
        try:

            self.logger_object.log(self.file_object, 'Entered function check_for_missing_values of Preprocessor class.')
            if dataset.isna().sum().any():

                df_nan_values = pd.DataFrame()

                for col in dataset.columns:
                    nr_nan_values_percentage = dataset[col].isna().mean()
                    nr_nan_values = dataset[col].isna().sum()

                    if nr_nan_values > 0:
                        row = {'Feature': col,
                               'Percentage Nan Values': nr_nan_values_percentage,
                               'Amount Nan Values': nr_nan_values}
                        df_nan_values = df_nan_values.append(row, ignore_index=True)

                df_nan_values.to_csv('nan_values.csv', index=False)

                self.logger_object.log(self.file_object,
                                       'Function check_for_missing_values of Preprocessor class Completed Successfully. Exited this function.')
                return True

            else:
                self.logger_object.log(self.file_object, 'Function check_for_missing_values of Preprocessor class Completed Successfully. Exited this function.')
                return False


        except Exception as e:
            self.logger_object.log(self.file_object, 'Error Occured in function check_for_missing_values of Preprocessor class. Error Message : ' +str(e))




    def correlation_heatmap(self, X_train):
        self.logger_object.log(self.file_object, 'Entered function multicolleniarity of Preprocessor class.')
        try:
            plt.figure(figsize=(30, 22))
            cor = X_train.corr()
            sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
            plt.savefig('multicolleniarity.jpg')

            self.logger_object.log(self.file_object, 'Function multicolleniarity Completed Successfully. Exited this function.')
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function multicolleniarity of Preprocessor class. Error Message : ' + str(e))

    def feature_selection_recursive_elemination_cv(self, model, X_train, y_train, X_test, scoring, step=1,
                                                   min_features_to_select=1,
                                                   cv_kfold=5, verbose=False):
        try:
            self.logger_object.log(self.file_object, 'Entered function feature_selection_recursive_elemination_cv.')

            total_nr_features = X_train.shape[1]
            selector = RFECV(model, step=step, min_features_to_select=min_features_to_select, cv=cv_kfold,
                             scoring=scoring,
                             n_jobs=-1, verbose=verbose)
            selector.fit(X_train, y_train)
            nr_features_selected = selector.n_features_
            self.logger_object.log(self.file_object, str(nr_features_selected) + ' out of total ' + str(total_nr_features))

            dicti = {}
            for col, ranking in zip(X_train.columns, selector.ranking_):
                dicti[col] = ranking

            cols_selected = X_train.columns[selector.support_]

            self.logger_object.log(self.file_object, f'Features selected : {cols_selected}')

            new_X_train = X_train[cols_selected]
            new_X_test = X_test[cols_selected]

            self.logger_object.log(self.file_object, 'Function feature_selection_recursive_elemination_cv Completed Successfully. Exited this function.')
            return new_X_train, new_X_test

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function feature_selection_recursive_elemination_cv. Error Message : ' + str(e))

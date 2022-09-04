from application_logging.logger import App_Logger
from data_ingestion.data_loader import Data_Getter
from data_preprocessing.preprocessing import Preprocessor
from best_model_finder.modelTuning import  ModelTuner
from xgboost import XGBClassifier
from File_Operation.FileOperation import File_Operations
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action='ignore')

class trainModel:

    def __init__(self, training_file_path):
        self.logger = App_Logger()
        self.file_training = open("Logs/ModelTraining_Log.txt", 'a+')
        self.file_preprocessing = open('Logs/DataPreprocessing_Log.txt', 'a+')
        self.training_file_path = training_file_path

    def trainingModel(self):
        # Logging the start of Training
        self.logger.log(self.file_preprocessing, 'Start Preprocessing and Training')
        try:

            # DATA PREPROCESSING

            # FEATURE ENGINEERING
            # read the csv file (dataset)
            dg = Data_Getter(self.file_preprocessing, self.logger, self.training_file_path)
            df = dg.get_data()

            # DATA PREPROCESSING
            # FEATURE ENGINEERING

            # 1. Train Test Split
            preprocessor = Preprocessor(self.file_preprocessing, self.logger)
            X, y = preprocessor.separate_features_label(df, 'Bad_Loan')
            train_data, test_data = preprocessor.train_test_splitting(X, y, test_size=0.2, stratify=True)

            # 2. Dropping columns that are not relevant for prediction (we apply for training and test data)
            train_data = preprocessor.remove_columns(train_data, ['RowID'])
            test_data = preprocessor.remove_columns(test_data, ['RowID'])

            # 3. Handling unwanted values
            train_data = preprocessor.handling_unwanted_values(train_data)
            test_data = preprocessor.handling_unwanted_values(test_data)

            # 4. Handling Missing Values (we apply for training and test data)
            # 4.1 Check if there is any missing value
            result_train = preprocessor.check_for_missing_values(train_data)
            if result_train:
                # 4.2 : Handle Missing Values
                train_data = preprocessor.drop_missing_values(train_data)
                test_data = preprocessor.drop_missing_values(test_data)

            # 5. Encoding categorical features ( we apply for training and test data)
            train_data, test_data = preprocessor.encode_categorical_features(train_data, test_data)

            # 6. Separate features and label
            X_train, y_train = preprocessor.separate_features_label(train_data, 'Bad_Loan')
            X_test, y_test = preprocessor.separate_features_label(test_data, 'Bad_Loan')

            # 7. FEATURE SELECTION
            s = len(y_train[y_train] == 0) / len(y_train[y_train == 1])
            model = XGBClassifier(random_state=1, scale_pos_weight=s)
            X_train, X_test = preprocessor.feature_selection_recursive_elemination_cv(model=model, X_train=X_train,
                                                                                      y_train=y_train, X_test=X_test,
                                                                                      scoring='roc_auc')

            # 8. Model Tuning
            model_tuner = ModelTuner(self.file_training, self.logger)
            best_model_object, best_model_name, best_test_score = model_tuner.best_model_OutOfManyModels_RandomizedSearchCV(
                X_train, y_train, X_test, y_test, cv_scoring='roc_auc', cv_niter=20)

            self.logger.log(self.file_training, f'Best Model Name : {best_model_name}. Best Test Score : {best_test_score}')


            # 9. Save Model to a pickle file
            op = File_Operations(self.file_training, self.logger)
            op.save_model(best_model_object, best_model_name)

            self.logger.log(self.file_training, 'Successful End of Training!')
            self.file_training.close()
            self.file_preprocessing.close()

        except Exception as e:
            self.logger.log(self.file_training, 'Unsuccessful End of Training. Error Message : ' + str(e))
            self.file_training.close()
            self.file_preprocessing.close()
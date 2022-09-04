import pandas as pd

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.
    """
    def __init__(self, file_object, logger_object, training_file_path):
        self.training_file = 'Training_FileFromDB/data.csv'
        self.file_object = file_object
        self.logger_object = logger_object
        self.training_file = training_file_path

    def get_data(self):
        """
            Method Name: get_data
            Description: This method reads the csv data from source.
            Parameters : No parameters
            Output: A pandas DataFrame.
            On Failure: Raise Exception

        """

        self.logger_object.log(self.file_object, 'Entered the function get_data of class Data_Getter.')
        try:
            df = pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object, 'Function get_data of class Data_Getter Completed Successfully. Exited this function')
            return df
        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function get_data. Error Message : ' + str(e))

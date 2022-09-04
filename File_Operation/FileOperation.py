import pickle
import os
import shutil

class File_Operations:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object


    def save_model(self, model, filename):
        try:
            self.logger_object.log(self.file_object, 'Entered function save_model of File_Operations class.')
            model_directory = 'models/'
            if os.path.isdir(model_directory) == False:
                os.makedirs(model_directory)

            path = os.path.join(model_directory, filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + '/' + filename + '.sav', 'wb') as f:
                pickle.dump(model, f)


            self.logger_object.log(self.file_object, 'Function save_model of File_operations class Completed Successfully. Exited this function.')

            return 'success'

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function save_model of File_Operations class. Error Message : ' + str(e))



    def load_model(self, filename):
        try:
            model_directory = 'models/'
            self.logger_object.log(self.file_object, 'Entered function load_model of File Operation class.')

            with open(model_directory + filename + '/' + filename + '.sav', 'rb') as f:
                self.logger_object.log(self.file_object, 'Function load_model of File Operation class Completed Succesfully. Exited this function.')
                return pickle.load(f)

        except Exception as e:
            self.logger_object.log(self.file_object, 'Error occured in function load_model of FIle Operation class. Error Message : ' + str(e))
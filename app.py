from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
from application_logging.logger import App_Logger
from flask import Response
import pickle
import pandas as pd
import numpy as np
from Functions.functions import convert_sqft_to_num
import json
from ModelTraining.trainingModel import trainModel
from File_Operation.FileOperation import File_Operations

app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # get user input from html form
        loan_amount_form = request.form['loan_amount']
        term_form = request.form['term']
        interest_rate_form = request.form['interest_rate']
        employment_years_form = request.form['employment_years']
        annual_income_form = request.form['annual_income']
        state_form = request.form['state']
        debt_to_income_form = request.form['debt_to_income']
        delinquent_2yr_form = request.form['delinquent_2yr']
        revolving_cr_util_form = request.form['revolving_cr_util']
        total_accounts_form = request.form['total_accounts']
        longest_credit_length_form = request.form['longest_credit_length']
        home_ownership_form = request.form['home_ownership']
        verification_status_form = request.form['verification_status']
        loan_purpose_form = request.form['loan_purpose']


        # load the model
        logger = App_Logger()
        file_prediction = open("Logs/Prediction_Log.txt", 'a+')
        file_io = File_Operations(logger_object=logger, file_object=file_prediction)
        model = file_io.load_model('EasyEnsembleClassifier')

        # load the encoded features dictionary
        with open('encoded_features.json', 'r') as myfile:
            encoded_features_str = myfile.read()
            encoded_features = json.loads(encoded_features_str)

        # Features Preprocessing

        # Preprocess loan_amount
        # (no need to preprocess loan_amount since it will be a number. i will just convert to float)
        loan_amount = float(loan_amount_form)
        loan_amount_list = [loan_amount]

        # Preprocess term
        # term will either be '36 months' or 60 'months'. I will convert it to int since the datatype of this feature when training happend was int
        term_tokens = term_form.split('months')
        term = int(term_tokens[0])
        term_list = [term]

        # Preprocess interest_rate
        # no need to preprocess interest_rate since it is a number. i will just convert it to float
        interest_rate = float(interest_rate_form)
        interest_rate_list = [interest_rate]

        # Preprocess Employment_Years
        # no need to preprocess Employment_Years since it is a number, i will just convert it to float
        employment_years = float(employment_years_form)
        employment_years_list = [employment_years]

        # Preprocess Annual_Income
        # no need to preprocess Annual_Income sinnce it is a number. I will just convert it to float
        annual_income = float(annual_income_form)
        annual_income_list = [annual_income]

        # Preprocess State
        # load the dic_state dictionary
        with open('dic_state.json', 'r') as myfile:
            dic_state_str = myfile.read()
            dic_state = json.loads(dic_state_str)

        state = dic_state[state_form]
        state_list = [state]

        # Preprocess debt_to_income
        debt_to_income = float(debt_to_income_form)
        debt_to_income_list = [debt_to_income]

        # Preprocess Delinquent_2yr
        # no need to preprocess Delinquent_2yr since it is a number. I will just convert it to float.
        delinquent_2yr = float(delinquent_2yr_form)
        delinquent_2yr_list = [delinquent_2yr]

        # Preprocess Revolving_Cr_Util
        # no need to preprocess Revolving_Cr_Util since it is a number. I will just convert it to float.
        revolving_cr_util = float(revolving_cr_util_form)
        revolving_cr_util_list = [revolving_cr_util]

        # Preprocess Total_Accounts
        # no need to preprocess Total_Accounts since it is a number. I will just convert it to float.
        total_accounts = float(total_accounts_form)
        total_accounts_list = [total_accounts]

        # Preprocess Longest_Credit_Length
        # no need to preprocess Longest_Credit_Length since it is a number. I will just convert it to float
        longest_credit_length = float(longest_credit_length_form)
        longest_credit_length_list = [longest_credit_length]

        # Preprocess Home_Ownership
        if home_ownership_form == 'RENT':
            Home_Ownership_MORTGAGE = 0
            Home_Ownership_OWN = 0
            Home_Ownership_RENT = 1
        elif home_ownership_form == 'OWN':
            Home_Ownership_MORTGAGE = 0
            Home_Ownership_OWN = 1
            Home_Ownership_RENT = 0
        elif home_ownership_form == 'MORTGAGE':
            Home_Ownership_MORTGAGE = 1
            Home_Ownership_OWN = 0
            Home_Ownership_RENT = 0
        elif home_ownership_form == 'OTHER':
            Home_Ownership_MORTGAGE = 0
            Home_Ownership_OWN = 0
            Home_Ownership_RENT = 0
        elif home_ownership_form == 'NONE':
            Home_Ownership_MORTGAGE = 0
            Home_Ownership_OWN = 0
            Home_Ownership_RENT = 0

        home_ownership_list = [Home_Ownership_MORTGAGE, Home_Ownership_OWN, Home_Ownership_RENT]

        # Preprocess Verification Status
        if verification_status_form == 'VERIFIED - income':
            Verification_Status_VERIFIED_income = 1
            Verification_Status_VERIFIED_income_source = 0
            Verification_Status_not_verified = 0
        elif verification_status_form == 'VERIFIED - income source':
            Verification_Status_VERIFIED_income = 0
            Verification_Status_VERIFIED_income_source = 1
            Verification_Status_not_verified = 0
        elif verification_status_form == 'not verified':
            Verification_Status_VERIFIED_income = 0
            Verification_Status_VERIFIED_income_source = 0
            Verification_Status_not_verified = 1

        verification_status_list = [Verification_Status_VERIFIED_income, Verification_Status_VERIFIED_income_source,
                                    Verification_Status_not_verified]

        # Preprocess Loan Purpose
        new_loan_purpose = 'Loan_Purpose_' + loan_purpose_form
        encoded_loan_purpose = encoded_features['Loan_Purpose']
        loan_purpose_vec = np.zeros(len(encoded_loan_purpose))
        for i in range(len(encoded_loan_purpose)):
            if encoded_loan_purpose[i] == new_loan_purpose:
                loan_purpose_vec[i] = 1

        loan_purpose_list = list(loan_purpose_vec)

        # append all inputs into a single list
        X = loan_amount_list + term_list + interest_rate_list + employment_years_list + annual_income_list + state_list + debt_to_income_list + delinquent_2yr_list + revolving_cr_util_list + total_accounts_list + longest_credit_length_list + home_ownership_list + verification_status_list + loan_purpose_list
        X_arr = np.array([X])  # convert list to numpy array
        pred = model.predict(X_arr)

        if pred[0] == 1:
            final_pred =  'BAD LOAN. LOAN DISSAPROVAL!'
        else:
            final_pred = 'GOOD LOAN. LOAN APPROVAL!'


        return render_template('home.html', prediction_text=final_pred)

    return render_template("home.html")

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['folderTrainingPath'] is not None:
            training_file_path = request.json['folderTrainingPath']

            trainModelObj = trainModel(training_file_path=training_file_path) #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


if __name__ == "__main__":
    app.run(debug=True)

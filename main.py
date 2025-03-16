from copyreg import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
from ydata_profiling import ProfileReport

class LRModelling:

    def __init__(self, input_dataframe):
        """
            This is the initialisation function
        :param input_dataframe: inputs the dataframe containing data
        """
        try:
            self.raw_data_df = input_dataframe
            self.cleaned_data = pd.DataFrame()
            self.x_array = np.empty((1,1))
            self.y_array = np.empty((1, 1))
            self.x_train = np.empty((1, 1))
            self.y_train = np.empty((1, 1))
            self.x_test = np.empty((1, 1))
            self.y_test = np.empty((1, 1))

        except Exception as e:
            raise Exception ("(__init__): Error occurred while initialising object of class LRModelling, {}".format(e)) from None

    def data_visualisation(self):
        """
        :return:
        """
        try:
            profile_report_var = ProfileReport(self.raw_data_df)
            p_report_html_var = profile_report_var.to_html()
            report_file_handle = open("profile_report_save.html", "w")
            report_file_handle.write(p_report_html_var)
            report_file_handle.close()
        except Exception as e:
            raise Exception ("(data_visualisation): Error occurred while visualizing the data, {}".format(e)) from None

    def data_clean_fill_na(self):
        """
        :return:
        """
        try:
            dataframe_data = self.raw_data_df
            for i in dataframe_data.columns:
                if dataframe_data[i].isnull().sum() > 0:
                    dataframe_data[i] = dataframe_data[i].fillna(dataframe_data[i].mean())
                else:
                    pass
            self.cleaned_data = dataframe_data
        except Exception as e:
            raise Exception ("(data_clean_fill_na): Some error occurred while filling na data, {}".format(e)) from None

    def data_cleaning(self):
        """
        :return:
        """
        try:
            #dealing with missing values
            self.data_clean_fill_na()
        except Exception as e:
            raise Exception ("(data_cleaning): Some error occurred while cleaning raw data, {}".format(e)) from None

    def data_preprocessing(self, param_x_data, param_y_data):
        """
        :param param_x_data:
        :param param_y_data:
        :return:
        """
        try:
            self.x_array = np.array(param_x_data).reshape(-1,7)
            self.y_array = np.array(param_y_data).reshape(-1,1)
        except Exception as e:
            raise Exception ("(data_preprocessing): Something went wrong while pre-processing data, {}".format(e)) from None

    def standardizing_data(self, param_x_data):
        """

        :param param_x_data:
        :return:
        """
        try:
            scaler_obj = StandardScaler()
            scaler_obj.fit(param_x_data)
            file_handle_01 = open("standard_scaler_obj_save.sav","wb")
            pickle.dump(scaler_obj, file_handle_01)
            file_handle_01.close()
            return_x_data = scaler_obj.fit_transform(param_x_data)
            return return_x_data
        except Exception as e:
            raise Exception ("(standardizing_data): Some error occurred while doing standard scalar standardization, {}".format(e))

    def train_test_split(self):
        """
        :return:
        """
        try:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_array, self.y_array, test_size=0.25, random_state= 217)

        except Exception as e:
            raise Exception ("(train_test_split): Something went wrong while doing train test split".format(e)) from None

    def lasso_regression_fit_save(self):
        """
        :return:
        """
        try:
            lasso_cv_obj = LassoCV(cv = 5, random_state= 217)
            lasso_cv_obj.fit(self.x_train, self.y_train)

            lasso_obj = Lasso(alpha= lasso_cv_obj.alpha_)
            lasso_obj.fit(self.x_train, self.y_train)

            file_handle = open("lasso_reg_fit_model_save.sav", "wb")
            pickle.dump(lasso_obj, file_handle)
            file_handle.close()

            print(lasso_obj.coef_)

        except Exception as e:
            raise Exception ("(lasso_regression_fit_save): Some error occurred while fitting and saving lasso regression, {}".format(e)) from None

admission_predict_df = pd.read_csv("D:/Full_stack_data_science/ML 03. Linear Regression live coding demonstration part 1/Admission_Prediction.csv")

lr_modelling_object = LRModelling(admission_predict_df)

lr_modelling_object.data_cleaning()

#lr_modelling_object.data_visualisation()
print(lr_modelling_object.cleaned_data.columns)

x_data = lr_modelling_object.cleaned_data.drop(["Serial No.", "Chance of Admit"], axis = 1)
y_data = lr_modelling_object.cleaned_data["Chance of Admit"]

lr_modelling_object.data_preprocessing(x_data, y_data)

lr_modelling_object.x_array = lr_modelling_object.standardizing_data(lr_modelling_object.x_array)

lr_modelling_object.train_test_split()

lr_modelling_object.lasso_regression_fit_save()

print("You have done amazing job, Great work harshal")



















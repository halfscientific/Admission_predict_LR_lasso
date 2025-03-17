

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS,cross_origin

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNetCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import pickle

app_obj = Flask(__name__)

@app_obj.route("/", methods= ["GET", "POST"])
def home_page():
    return render_template("home_page.html")

@app_obj.route("/thanks", methods=["GET", "POST"])
def thanks_page():
    return render_template("thanks_page.html")

@app_obj.route("/ip_predict", methods=["GET", "POST"])
def lr_model_prediction():
    try:

        gre_score = float(request.form["gre_score"])
        toefl_score = float(request.form["toefl_score"])
        university_rating = float(request.form["university_rating"])
        sop = float(request.form["sop"])
        lor = float(request.form["lor"])
        cgpa = float(request.form["cgpa"])
        research = float(request.form["research"])

        predict_list_df = pd.DataFrame([gre_score, toefl_score, university_rating, sop, lor, cgpa, research])
        predict_array = np.array(predict_list_df).reshape(-1,7)
        file_handle_01 = open("standard_scaler_obj_save.sav", "rb")
        standard_scaler_obj = pickle.load(file_handle_01)
        file_handle_01.close()
        predict_array_transformed = standard_scaler_obj.transform(predict_array)

        file_handle = open("lasso_reg_fit_model_save.sav", "rb")
        lr_regression_obj = pickle.load(file_handle)
        file_handle.close()

        predicted_value_01 = lr_regression_obj.predict(predict_array_transformed)[0]*100
        predicted_value = "{} %".format(predicted_value_01)
        return render_template("result_prediction.html", value_01 = predicted_value)

    except Exception as e:
        raise Exception ("(lr_model_prediction): Something went wrong while prediction from a lr model, {}".format(e)) from None

if __name__ == "__main__":
    app_obj.run()
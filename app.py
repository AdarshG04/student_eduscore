from flask import Flask, request ,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

## Route for a prediction page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            hours_studied=request.form.get('hours_studied'),
            previous_scores=request.form.get('previous_scores'),
            extracurricular_activities=request.form.get('extracurricular_activities'),
            sleep_hours=request.form.get('sleep_hours'),
            sample_question_papers_practiced=request.form.get('sample_question_papers_practiced')
        )

        pred_df=data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True) 
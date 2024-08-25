import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:     
            model_path="artifacts\\model.pkl"
            preprocessor_path='artifacts\\preprocessor.pkl'
            # print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        hours_studied: int,
        previous_scores: int,
        extracurricular_activities,
        sleep_hours: int,
        sample_question_papers_practiced: int,
        ):

        self.hours_studied = hours_studied

        self.previous_scores = previous_scores

        self.extracurricular_activities = extracurricular_activities

        self.sleep_hours = sleep_hours

        self.sample_question_papers_practiced = sample_question_papers_practiced



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "hours_studied": [self.hours_studied],
                "previous_scores": [self.previous_scores],
                "extracurricular_activities": [self.extracurricular_activities],
                "sleep_hours": [self.sleep_hours],
                "sample_question_papers_practiced": [self.sample_question_papers_practiced],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
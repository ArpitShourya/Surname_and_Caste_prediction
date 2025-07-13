from data_validation import DataValidation
from preprocessing import Preprocessing
from prediction import Prediction
import pandas as pd
if __name__=="__main__":
    df=pd.read_excel("Testing Data/Testing_Data_1.xlsx")
    dvobj=DataValidation(df)
    pre_obj=Preprocessing(df)
    if dvobj.check_required_columns():
        preprocessed_data=pre_obj.start_preprocessing()
        dvobj=DataValidation(preprocessed_data)
        if dvobj.check_no_digits():
            predictor = Prediction(df)
            predicted_df = predictor.start_prediction()
            predictor.save_to_excel("Predicted_Surnames_Output_testing_data1.xlsx")
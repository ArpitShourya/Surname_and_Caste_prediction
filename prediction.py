from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import pandas as pd
from tqdm import tqdm

class Prediction:
    def __init__(self, data: pd.DataFrame):
        self.surname_model_path = "mt5_surname_finetuned"
        self.caste_model_path="mt5_caste_finetuned"

        self.surname_tokenizer = MT5Tokenizer.from_pretrained(self.surname_model_path)
        self.surname_model = MT5ForConditionalGeneration.from_pretrained(self.surname_model_path)
        
        self.caste_tokenizer=MT5Tokenizer.from_pretrained(self.caste_model_path)
        self.caste_model=MT5ForConditionalGeneration.from_pretrained(self.caste_model_path)
        
        self.data = data.copy()  # avoid modifying original DataFrame

    def format_input_surname(self, row):
        return (
            f"FM_NAME_EN: {row['FM_NAME_EN']}, LASTNAME_EN: {row['LASTNAME_EN']}, "
            f"FM_NAME_V1: {row['FM_NAME_V1']}, LASTNAME_V1: {row['LASTNAME_V1']}, "
            f"RLN_TYPE: {row['RLN_TYPE']}, "
            f"RLN_NAME_EN: {row['RLN_FM_NM_EN']} {row['RLN_L_NM_EN']}, "
            f"RLN_NAME_V1: {row['RLN_FM_NM_V1']} {row['RLN_L_NM_V1']}"
        )
    
    def format_input_cast(self, row):
        return (
        f"FM_NAME_EN: {row['FM_NAME_EN']}, LASTNAME_EN: {row['LASTNAME_EN']}, "
        f"FM_NAME_V1: {row['FM_NAME_V1']}, LASTNAME_V1: {row['LASTNAME_V1']}, "
        f"RLN_TYPE: {row['RLN_TYPE']}, "
        f"Correct_Hin_Surname: {row['Correct_Hin_Surname']}, "
        f"RLN_NAME_EN: {row['RLN_FM_NM_EN']} {row['RLN_L_NM_EN']}, "
        f"RLN_NAME_V1: {row['RLN_FM_NM_V1']} {row['RLN_L_NM_V1']}"
        )

    def predict_one_surname(self, input_text):
        inputs = self.surname_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        output = self.surname_model.generate(**inputs, max_length=16)
        return self.surname_tokenizer.decode(output[0], skip_special_tokens=True)
    
    def predict_one_caste(self, input_text):
        inputs = self.caste_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        output = self.caste_model.generate(**inputs, max_length=16)
        return self.caste_tokenizer.decode(output[0], skip_special_tokens=True)

    def start_prediction(self):
        tqdm.pandas(desc="Predicting Surnames")
        self.data["Correct_Hin_Surname"] = self.data.progress_apply(
            lambda row: self.predict_one_surname(self.format_input_surname(row)), axis=1
        )
        tqdm.pandas(desc="Predicting Caste")
        self.data["Final Caste"] = self.data.progress_apply(
            lambda row: self.predict_one_caste(self.format_input_cast(row)), axis=1
        )

        return self.data

    def save_to_excel(self, filename="predictions_.xlsx"):
        self.data.to_excel(filename, index=False)
        print(f"✅ Saved predictions to {filename}")

if __name__=="__main__":
    df=pd.read_excel("E:/Surname prediction/Files/Preprocessed_mydata3.xlsx",engine="openpyxl")
    predictor = Prediction(df)
    predicted_df = predictor.start_prediction()
    predictor.save_to_excel("Predicted_Surnames_and_caste_mydata.xlsx")

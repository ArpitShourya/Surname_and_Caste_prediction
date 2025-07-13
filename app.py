import joblib
import pandas as pd
from preprocessing import Preprocessing
from data_validation import DataValidation
from prediction import Prediction
import certifi
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

ca = certifi.where()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict_caste_and_surname")
async def predict_route(request: Request, file: UploadFile = File(...)):
    # Step 1: Read Excel file
    df = pd.read_excel(file.file)

    validator=DataValidation(df)

    missing = validator.check_required_columns()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "fail",
                "message": f"Missing required columns: {missing}"
            }
        )
    
    prepro_obj=Preprocessing(df)
    preprocessed_data=prepro_obj.start_preprocessing()
    validator=DataValidation(preprocessed_data)


    digit_violations = validator.check_no_digits()
    if digit_violations:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "fail",
                "message": "Digit-containing values found in columns.",
                "violations": {
                    col: values[:5] + (["..."] if len(values) > 5 else [])
                    for col, values in digit_violations.items()
                }
            }
        )
    
    predictor = Prediction(df)
    predicted_df = predictor.start_prediction()

    # Step 9: Save output Excel
    output_path = "Testing Output/Surname_Predicted_Output.xlsx"
    predicted_df.to_excel(output_path, index=False)

    # Step 10: Return Excel file as response
    return Response(
        content=open(output_path, "rb").read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=Output.xlsx"}
    )

if __name__ == "__main__":
    app_run(app=app, host="localhost", port=5000)

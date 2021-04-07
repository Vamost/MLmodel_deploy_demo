'''
Author: your name
Date: 2021-04-07 10:55:44
LastEditTime: 2021-04-07 18:24:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /testLR/api/main.py
'''

import numpy as np
import pandas as pd
from typing import List

from fastapi import FastAPI
from fastapi import Depends

# 以下引用用来支持.csv文件的处理
from fastapi import File, UploadFile, HTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


from pydantic import BaseModel
from pydantic import ValidationError, validator # 对数据进行校验

from .ml.model import get_model, n_features



class PredictRequest(BaseModel):
    data: List[List[float]]

    # 对用户的输入进行校验
    @validator('data')
    def check_dimensionality(cls, v):
        for point in v:
            if  len(point) != n_features:
                raise ValueError(f"Each data point must contain {n_features} features")
            

        return v

class PredictResponse(BaseModel):
    data: List[float]

app = FastAPI()

# @app.post("/predict", response_model=PredictResponse)
# def predict(input: PredictRequest):
#     return PredictResponse(data=[0.0])

@app.post("/predict", response_model=PredictResponse):
def predict(input: PredictRequest, model: Model = Depends(get_model)):
    X = np.array(input.data)
    y_pred = model.predict(X)
    result = PredictResponse(data=y_pred.to_list())

    return result

@app.post("/predict.csv")
def predict_csv(csv_file: UploadFile = File(...), model: Model = Depends(get_model)):
    try:
        df = pd.read_csv(csv_file.file).astype(float)
    except :
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file"
        )
    
    df_n_instances, df_n_features = df.shape
    if df_n_features != n_features:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Each data point must contain {n_features} features",
        )

    y_pred = model.predict(df.to_numpy().reshape(-1, n_features))
    result = PredictResponse(data=y_pred.to_list())

    return result
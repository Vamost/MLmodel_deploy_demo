'''
Author: your name
Date: 2021-04-07 10:55:44
LastEditTime: 2021-04-07 18:06:01
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /testLR/api/main.py
'''

import  numpy as np
from typing import List

from fastapi import FastAPI
from fastapi import Depends


from pydantic import BaseModel

from .ml.model import get_model



class PredictRequest(BaseModel):
    data: List[List[float]]


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
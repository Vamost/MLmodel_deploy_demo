'''
Author: your name
Date: 2021-04-07 10:55:44
LastEditTime: 2021-04-07 11:38:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /testLR/api/main.py
'''
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    data: List[List[float]]


class PredictResponse(BaseModel):
    data: List[float]

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(input: PredictRequest):
    return PredictResponse(data=[0.0])
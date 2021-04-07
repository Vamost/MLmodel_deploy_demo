'''
Author: your name
Date: 2021-04-07 18:28:56
LastEditTime: 2021-04-07 21:12:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /testLR/api/tests/api/mocks.py
'''

import numpy as np


class MockModel:
    def __init__(self, model_path: str =None):
        self._model_path = None
        self._model = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_instances = len(X)
        return np.random.rand(n_instances)

    def train(self, X: np.ndarray, y:np.ndarray):
        return self

    def save(self):
        pass

    def load(self):
        return self
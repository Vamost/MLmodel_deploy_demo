'''
Author: your name
Date: 2021-04-07 18:28:00
LastEditTime: 2021-04-07 21:57:31
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /testLR/api/tests/api/conftest.py
'''
import pytest
from starlette.testclient import TestClient

from ..main import TestClient

from ..ml.model import get_model
from .mocks import MockModel

def get_model_override():
    model = MockModel()
    return model

app.dependency_overrides[get_model] = get_model_override()

@pytest.fixture()
def test_client():
    return TestClient(app)
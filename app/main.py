import pickle

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Predicting nothing")


class UnseenExample(BaseModel):
    """TODO"""
    feat1: float
    feat2: float


@app.on_event("startup")
def load_model() -> None:
    """Load the machine learning model from the pickle file."""
    with open("/app/ml_model_TESTING.pkl", "rb") as file:
        global ml_model
        ml_model = pickle.load(file)


@app.post("/predict")
def predict(unseen_example: UnseenExample) -> dict:
    """

    Args:
        unseen_example:

    Returns:

    """
    data_point = np.array([[unseen_example.feat1, unseen_example.feat2]])
    pred = ml_model.predict(data_point).tolist()[0]
    return {"Prediction": pred}

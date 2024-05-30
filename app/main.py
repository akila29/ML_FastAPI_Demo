import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager


app = FastAPI(title="Iris Prediction")

# Description for features in Iris Data (without the class label)
class Iris(BaseModel):
    
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@asynccontextmanager
async def lifespan():
    print("Ran on startup")
    # Load model from pickle file
    with open("/app/model.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)
    yield
    

@app.get("/")
def home():
    return "Hi! We are up and running. Go to http://localhost:80/docs"


@app.post("/predict")
def predict(iris: Iris):

    labels = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
    data = np.array(
        [
            [
                iris.sepal_length,
                iris.sepal_width,
                iris.petal_length,
                iris.petal_width,
            ]
        ]
    )

    pred = clf.predict(data)
    pred = labels[pred[0]]
    return {"Prediction": pred}

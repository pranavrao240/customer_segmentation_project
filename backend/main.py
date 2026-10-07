from fastapi import FastAPI
from predictor import predict_customer


app = FastAPI(
    title="Intelligent Customer Segmentation API",
    description="API for customer segmentation using K-Means clustering",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Customer Segmentation API is running!"
    }


@app.post("/predict")
def predict(customer: dict):

    result = predict_customer(customer)

    return result
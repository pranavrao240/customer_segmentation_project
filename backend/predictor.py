import joblib
import pandas as pd
import os


# Find the model file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "customer_segmentation_model.pkl"
)


# Load the trained model
model_data = joblib.load(MODEL_PATH)


# Extract model components
preprocessor = model_data["preprocessor"]
pca = model_data["pca"]
kmeans = model_data["kmeans"]

cluster_names = model_data["cluster_names"]

feature_columns = model_data["feature_columns"]


def predict_customer(customer_data):

    # Convert input dictionary to DataFrame
    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Make sure columns are in the correct order
    customer_df = customer_df[
        feature_columns
    ]

    # Preprocessing
    processed_data = preprocessor.transform(
        customer_df
    )

    # PCA transformation
    pca_data = pca.transform(
        processed_data
    )

    # Predict cluster
    cluster = int(
        kmeans.predict(pca_data)[0]
    )

    # Get segment name
    segment_name = cluster_names[cluster]

    return {
        "cluster": cluster,
        "segment_name": segment_name
    }
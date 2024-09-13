"""
Project: Deploy a ML Model
Author: Tuan Doan
Date: 2024-09-14
"""

import os
import sys
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

try:
    from module.data import process_data
    from module.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_slices_data
    )
except ModuleNotFoundError:
    sys.path.append('./')
    from module.data import process_data
    from module.model import (
        inference,
        compute_model_metrics,
        train_model,
        compute_metrics_with_slices_data
    )
# Add code to load in the data.



logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


current_script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_script_dir), "data/census_clean.csv")
data = pd.read_csv(data_path)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)



# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

model = train_model(X_train=X_train, y_train=y_train)

if not os.path.exists(os.path.join(os.path.dirname(current_script_dir), "model/")):
    os.mkdir("model/")
with open(os.path.join(os.path.dirname(current_script_dir), "model/lr_model.pkl"), "wb") as f:
    pickle.dump([encoder, lb, model], f)

logging.info(f">>> Infering model")
preds = inference(model=model, X=X_test)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
logging.info(f">>> Precision {precision}")
logging.info(f">>> Recall {recall}")
logging.info(f">>> fbeta {fbeta}")

logging.info("Calculating model metrics on slices data...")
metrics = compute_metrics_with_slices_data(
        df=test,
        cat_columns=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
        slice_output_path=os.path.join(os.path.dirname(current_script_dir),"slice_output.txt")
)
logging.info(f">>>Metrics with slices data: {metrics}")
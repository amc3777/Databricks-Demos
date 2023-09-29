# Databricks notebook source

import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import dill as pickle
import tensorflow as tf
import keras

inference_path = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "inference_dataset")
CATALOG = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "catalog")
SCHEMA = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "schema")
LABEL_COL = "default_payment_next_month"
ID_COL = "id"

mlflow.set_registry_uri("databricks-uc")

df = (spark.read.table(inference_path))
X = df.drop(ID_COL, LABEL_COL).toPandas()
Y = df.select(LABEL_COL).toPandas().values.ravel()
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

champion_model = dbutils.jobs.taskValues.get(taskKey = "03-eval-models-task", key = "champion_model")

if champion_model == 'automl':

  loaded_model = mlflow.sklearn.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_automl_clf@candidate")

  display(loaded_model)

  with open('model.pkl', 'wb') as file:
      pickle.dump(loaded_model, file)

  with open('model.pkl', 'rb') as file:
      model = pickle.load(file)

  display(type(model))

  display(model.predict(X))

else:

  tf_loaded_model = mlflow.tensorflow.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_tf_clf@baseline")

  display(tf_loaded_model.summary())

  tf_loaded_model.save("model.keras")

  reconstructed_model = keras.models.load_model("model.keras")

  display(type(reconstructed_model))

  display(reconstructed_model.predict(X_scaled).round())
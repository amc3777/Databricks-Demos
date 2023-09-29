# Databricks notebook source

import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import dill as pickle
import tensorflow as tf
import keras
from google.cloud import storage

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

storage_client = storage.Client()

bucket_name = dbutils.widgets.get("gcs_bucket")

if champion_model == 'automl':

  loaded_model = mlflow.sklearn.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_automl_clf@candidate")

  with open('model.pkl', 'wb') as file:
      pickle.dump(loaded_model, file)

  
  source_file_name='model.pkl'
  destination_blob_name='model-artifacts/model.pkl'

  try:

    bucket = storage_client.create_bucket(bucket_name, location="us-central1")
    print(f"Bucket {bucket.name} created.")

  except Exception:

      print(f"Bucket {bucket_name} already exists.")

  def upload_blob(bucket_name, source_file_name, destination_blob_name):
    
    bucket_name = bucket_name
    source_file_name = source_file_name
    destination_blob_name = destination_blob_name

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

  try:

    upload_blob(bucket_name, source_file_name, destination_blob_name)

  except Exception:

    print(f"Object {destination_blob_name} already exists. You need additional permissions to overwrite.")

else:

  tf_loaded_model = mlflow.tensorflow.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_tf_clf@baseline")

  tf_loaded_model.save("model.keras")

  source_file_name='model.keras'
  destination_blob_name='model-artifacts/model.keras'

  try:

    bucket = storage_client.create_bucket(bucket_name, location="us-central1")
    print(f"Bucket {bucket.name} created.")

  except Exception:

      print(f"Bucket {bucket_name} already exists.")

  def upload_blob(bucket_name, source_file_name, destination_blob_name):
    
    bucket_name = bucket_name
    source_file_name = source_file_name
    destination_blob_name = destination_blob_name

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

  try:

    upload_blob(bucket_name, source_file_name, destination_blob_name)

  except Exception:

    print(f"Object {destination_blob_name} already exists. You need additional permissions to overwrite.")
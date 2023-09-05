import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import mlflow
from mlflow.models.model import get_model_info
from mlflow.models import infer_signature, set_signature
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow import MlflowClient


training_path = dbutils.jobs.taskValues.get(taskKey = "02-save-prepared-data-task", key = "training_path")

LABEL_COL = "price"
ID_COL = "ID"
CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"

mlflow.set_registry_uri("databricks-uc")

mlflow.tensorflow.autolog(log_input_examples=True, silent=True, registered_model_name=f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_tf_regressor')

experiment_name = "/Users/andrew.cooley@databricks.com/airbnb_price_prediction"

try:

  experiment_id = mlflow.create_experiment(
      experiment_name
  )

except Exception:

  search_query_string = "name ='{}'".format(experiment_name)
  experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id
  print("Experiment already exists.")

with mlflow.start_run(run_name="tf_regressor", experiment_id=experiment_id) as mlflow_run:

  train_df = (spark.read.table(training_path))

  X_train = train_df.drop(ID_COL, LABEL_COL).toPandas()
  Y_train = train_df.select(LABEL_COL).toPandas().values.ravel()
  min_max_scaler = MinMaxScaler()
  X_train = min_max_scaler.fit_transform(X_train)

  callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

  inputs = keras.Input(shape=(np.shape(X_train)[1],))
  x = layers.Dense(128, activation="relu")(inputs)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(64, activation="relu")(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(32, activation="relu")(x)
  outputs = layers.Dense(1)(x)
  model = keras.Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

  history = model.fit(x=X_train, y=Y_train, epochs=50, verbose="auto", batch_size=16, validation_split=0.4)

model_uri = 'runs:/{}/model'.format(mlflow_run.info.run_id)
model = mlflow.pyfunc.load_model(model_uri)

signature = infer_signature(X_train[:5], model.predict(X_train[:5]))
set_signature(model_uri, signature)

runs_uri = "runs:/{}/model".format(mlflow_run.info.run_id)
model_src = RunsArtifactRepository.get_underlying_uri(runs_uri)

client = MlflowClient()
mv = client.create_model_version(f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_tf_regressor', model_src, mlflow_run.info.run_id)

client.set_registered_model_alias(f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_tf_regressor', "baseline", mv.version)

dbutils.jobs.taskValues.set(key = 'baseline_model', value = model_uri)
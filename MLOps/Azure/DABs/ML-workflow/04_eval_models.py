import warnings
import mlflow
import pandas as pd
import numpy as np
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

training_path = dbutils.jobs.taskValues.get(taskKey = "02-save-prepared-data-task", key = "training_path")
test_path = dbutils.jobs.taskValues.get(taskKey = "02-save-prepared-data-task", key = "test_path")

LABEL_COL = "price"
ID_COL = "ID"
CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"

mlflow.set_registry_uri("databricks-uc")

experiment_name = "/Users/andrew.cooley@databricks.com/airbnb_price_prediction"

client = MlflowClient()

baseline_model = mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.airbnb_price_prediction_tf_regressor@baseline")
candidate_model = mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.airbnb_price_prediction_automl_regressor@candidate")

test_df = (spark.read.table(test_path))
X_test = test_df.drop(ID_COL, LABEL_COL).toPandas()
Y_test = test_df.select(LABEL_COL).toPandas().values.ravel()
min_max_scaler = MinMaxScaler()
X_test_scaled = min_max_scaler.fit_transform(X_test)

Y_pred_base = baseline_model.predict(X_test_scaled)
mse_base = mean_squared_error(Y_test, Y_pred_base)
rmse_base = np.sqrt(mse_base)

Y_pred_cand = candidate_model.predict(X_test)
mse_cand = mean_squared_error(Y_test, Y_pred_cand)
rmse_cand = np.sqrt(mse_cand)

search_query_string = "name ='{}'".format(experiment_name)
experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id

print(f"Baseline rmse: {rmse_base} vs. Candidate rmse: {rmse_cand}")

if rmse_base > rmse_cand:

  print(f"Candidate model is better! Deploying candidate model to Databricks Model Serving...")

  with mlflow.start_run(run_name="champion_model", experiment_id=experiment_id) as mlflow_run:

    model_uri = f"models:/{CATALOG}.{SCHEMA}.airbnb_price_prediction_automl_regressor@candidate"
    model = mlflow.sklearn.load_model(model_uri)

    mlflow.sklearn.log_model(
                  model,
                  artifact_path="champion-model",
                  input_example=X_test[:5],
                  registered_model_name=f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_regressor',
                  signature=infer_signature(X_test, Y_test)
    )
    mlflow.log_metric("test_root_mean_squared_error", rmse_cand)

else:

  print(f"Baseline model is better! Deploying baseline model to Databricks Model Serving...")

  with mlflow.start_run(run_name="champion_model", experiment_id=experiment_id) as mlflow_run:

    model_uri = f"models:/{CATALOG}.{SCHEMA}.airbnb_price_prediction_tf_regressor@baseline"
    model = mlflow.tensorflow.load_model(model_uri)

    mlflow.tensorflow.log_model(
                  model,
                  artifact_path="champion-model",
                  input_example=X_test_scaled[:5],
                  registered_model_name=f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_regressor',
                  signature=infer_signature(X_test_scaled, Y_test)
    )
    mlflow.log_metric("test_root_mean_squared_error", rmse_base)

search_query_string = f"name='{CATALOG}.{SCHEMA}.airbnb_price_prediction_regressor'"
mv = client.search_model_versions(search_query_string)
client.set_registered_model_alias(f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_regressor', "champion", mv[0].version)
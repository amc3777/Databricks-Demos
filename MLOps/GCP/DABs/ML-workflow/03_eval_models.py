import warnings
import mlflow
import pandas as pd
import numpy as np
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

LABEL_COL = "default_payment_next_month"
ID_COL = "id"
CATALOG = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "catalog")
SCHEMA = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "schema")
test_path = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "test_dataset")

mlflow.set_registry_uri("databricks-uc")

experiment_name = "/Users/andrew.cooley@databricks.com/credit_card_default"

client = MlflowClient()

baseline_model = mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_tf_clf@baseline")
candidate_model = mlflow.pyfunc.load_model(f"models:/{CATALOG}.{SCHEMA}.credit_card_default_automl_clf@candidate")

test_df = (spark.read.table(test_path))
X_test = test_df.drop(ID_COL, LABEL_COL).toPandas()
Y_test = test_df.select(LABEL_COL).toPandas().values.ravel()
min_max_scaler = MinMaxScaler()
X_test_scaled = min_max_scaler.fit_transform(X_test)

Y_pred_base = baseline_model.predict(X_test_scaled).round()
accuracy_base = accuracy_score(Y_test, Y_pred_base)

Y_pred_cand = candidate_model.predict(X_test)
accuracy_cand = accuracy_score(Y_test, Y_pred_cand)

search_query_string = "name ='{}'".format(experiment_name)
experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id

print(f"Baseline accuracy: {accuracy_base} vs. Candidate accuracy: {accuracy_cand}")

if accuracy_base < accuracy_cand:

  print(f"Candidate model is better! Deploying candidate model to Databricks Model Serving...")

  with mlflow.start_run(run_name="champion_model", experiment_id=experiment_id) as mlflow_run:

    model_uri = f"models:/{CATALOG}.{SCHEMA}.credit_card_default_automl_clf@candidate"
    model = mlflow.sklearn.load_model(model_uri)

    mlflow.sklearn.log_model(
                  model,
                  artifact_path="champion-model",
                  input_example=X_test[:5],
                  registered_model_name=f'{CATALOG}.{SCHEMA}.credit_card_default_clf',
                  signature=infer_signature(X_test, Y_test)
    )
    mlflow.log_metric("test_accuracy", accuracy_cand)

else:

  print(f"Baseline model is better! Deploying baseline model to Databricks Model Serving...")

  with mlflow.start_run(run_name="champion_model", experiment_id=experiment_id) as mlflow_run:

    model_uri = f"models:/{CATALOG}.{SCHEMA}.credit_card_default_tf_clf@baseline"
    model = mlflow.tensorflow.load_model(model_uri)

    mlflow.tensorflow.log_model(
                  model,
                  artifact_path="champion-model",
                  input_example=X_test_scaled[:5],
                  registered_model_name=f'{CATALOG}.{SCHEMA}.credit_card_default_clf',
                  signature=infer_signature(X_test_scaled, Y_test)
    )
    mlflow.log_metric("test_accuracy", accuracy_base)

search_query_string = f"name='{CATALOG}.{SCHEMA}.credit_card_default_clf'"
mv = client.search_model_versions(search_query_string)
client.set_registered_model_alias(f'{CATALOG}.{SCHEMA}.credit_card_default_clf', "champion", mv[0].version)
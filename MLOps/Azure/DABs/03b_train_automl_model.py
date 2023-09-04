import mlflow
import uuid
from databricks import automl
from mlflow.models import infer_signature
from mlflow import MlflowClient

training_path = dbutils.jobs.taskValues.get(taskKey = "02-save-prepared-data-task", key = "training_path")

LABEL_COL = "price"
ID_COL = "ID"
CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"

mlflow.set_registry_uri("databricks-uc")

experiment_name = "/Users/andrew.cooley@databricks.com/airbnb_price_prediction"

client = MlflowClient()

summary = automl.regress(
  dataset=training_path,
  target_col=LABEL_COL,
  exclude_columns = [ID_COL],
  experiment_name="automl_airbnb_price_prediction_{}".format(str(uuid.uuid4())[:6]),
  primary_metric="rmse",
  timeout_minutes=10
)

model_uri = summary.best_trial.model_path
model = mlflow.sklearn.load_model(model_uri)

try:

  experiment_id = mlflow.create_experiment(
      experiment_name
  )

except Exception:

  search_query_string = "name ='{}'".format(experiment_name)
  experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id
  print("Experiment already exists.")

with mlflow.start_run(run_name="automl_regressor", experiment_id=experiment_id) as mlflow_run:
                mlflow.sklearn.log_model(
                              sk_model=model,
                              artifact_path="automl-model",
                              input_example=(spark.read.table(training_path)).drop(ID_COL, LABEL_COL).toPandas()[:5],
                              registered_model_name=f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_automl_regressor',
                              signature=infer_signature((spark.read.table(training_path)).drop(ID_COL, LABEL_COL).toPandas(), (spark.read.table(training_path)).select(LABEL_COL).toPandas().values.ravel())
                )

search_query_string = f"name='{CATALOG}.{SCHEMA}.airbnb_price_prediction_automl_regressor'"
mv = client.search_model_versions(search_query_string)
client.set_registered_model_alias(f'{CATALOG}.{SCHEMA}.airbnb_price_prediction_automl_regressor', "candidate", mv[0].version)              

dbutils.jobs.taskValues.set(key = 'candidate_model', value = model_uri)
import mlflow
import uuid
from databricks import automl
from mlflow.models import infer_signature
from mlflow import MlflowClient

LABEL_COL = "default_payment_next_month"
ID_COL = "id"
CATALOG = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "catalog")
SCHEMA = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "schema")
training_path = dbutils.jobs.taskValues.get(taskKey = "01-bq-load-transform-save-task", key = "training_dataset")

mlflow.set_registry_uri("databricks-uc")

experiment_name = "/Users/andrew.cooley@databricks.com/credit_card_default"

client = MlflowClient()

summary = automl.classify(
  dataset=training_path,
  target_col=LABEL_COL,
  exclude_columns = [ID_COL],
  experiment_name="automl_credit_card_default_{}".format(str(uuid.uuid4())[:6]),
  primary_metric="accuracy",
  timeout_minutes=15
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

with mlflow.start_run(run_name="automl_clf", experiment_id=experiment_id) as mlflow_run:
                mlflow.sklearn.log_model(
                              sk_model=model,
                              artifact_path="automl-model",
                              input_example=(spark.read.table(training_path)).drop(ID_COL, LABEL_COL).toPandas()[:5],
                              registered_model_name=f'{CATALOG}.{SCHEMA}.credit_card_default_automl_clf',
                              signature=infer_signature((spark.read.table(training_path)).drop(ID_COL, LABEL_COL).toPandas(), (spark.read.table(training_path)).select(LABEL_COL).toPandas().values.ravel())
                )
                mlflow.log_metric("val_accuracy", summary.best_trial.evaluation_metric_score)

search_query_string = f"name='{CATALOG}.{SCHEMA}.credit_card_default_automl_clf'"
mv = client.search_model_versions(search_query_string)
client.set_registered_model_alias(f'{CATALOG}.{SCHEMA}.credit_card_default_automl_clf', "candidate", mv[0].version)  

dbutils.jobs.taskValues.set(key = 'candidate_model', value = model_uri)
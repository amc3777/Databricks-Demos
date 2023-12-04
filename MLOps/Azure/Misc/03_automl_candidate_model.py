# Databricks notebook source
# MAGIC %md
# MAGIC ### Notebook set-up steps

# COMMAND ----------

import warnings

warnings.filterwarnings("ignore")

dbutils.widgets.removeAll()

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
uc_prefix = user.replace(".", "").split("@")[0]
catalog = uc_prefix + "_catalog"
schema = uc_prefix + "_schema"
volume = uc_prefix + "_managedvolume"

dbutils.widgets.text("catalog", catalog)
dbutils.widgets.text("schema", schema)
dbutils.widgets.text("volume",volume)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training set for AutoML regression experiment

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from sklearn.model_selection import train_test_split

from pyspark.sql.types import DoubleType

airbnb_df = spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_indexed")
numeric_cols = [x.name for x in airbnb_df.schema.fields if x.dataType == DoubleType()]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)

pdf = numeric_features_df.toPandas()

train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=42)
train_lookup_df = spark.createDataFrame(train_pdf[["index", "price"]])
test_lookup_df = spark.createDataFrame(test_pdf[["index", "price"]])

feature_lookups = [
    FeatureLookup(
      table_name = f'{catalog}.{schema}.airbnb_sf_listings_features',
      feature_names = numeric_features_df.select([c for c in numeric_features_df.columns if c not in {'index','price'}]).columns,
      lookup_key = 'index',
    )]

fs = feature_store.FeatureStoreClient()

training_set = fs.create_training_set(
  df=train_lookup_df,
  feature_lookups = feature_lookups,
  label = 'price',
  exclude_columns = ['index']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the Databricks AutoML Python API to initiate experiment with provided parameters

# COMMAND ----------

import datetime
from databricks import automl
import uuid

summary = automl.regress(
  dataset=training_set.load_df(),
  target_col="price",
  # data_dir: Optional[str] = None,
  # exclude_columns: Optional[List[str]] = None,                      # <DBR> 10.3 ML and above
  # exclude_frameworks: Optional[List[str]] = None,                   # <DBR> 10.3 ML and above
  # experiment_dir: Optional[str] = None,                             # <DBR> 10.4 LTS ML and above
  experiment_name="automl_airbnb_price_prediction_{}".format(str(uuid.uuid4())[:6]),
  # feature_store_lookups: Optional[List[Dict]] = None,               # <DBR> 11.3 LTS ML and above
  # imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None, # <DBR> 10.4 LTS ML and above
  primary_metric="rmse",
  # time_col: Optional[str] = None,
  timeout_minutes=10
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the best trial's model and create/re-use an MLflow Experiment

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

mlflow.set_registry_uri("databricks-uc")

model_uri = summary.best_trial.model_path
model = mlflow.sklearn.load_model(model_uri)

import mlflow

experiment_name = f"/Users/{user}/airbnb_price_prediction"

try:

  experiment_id = mlflow.create_experiment(
      experiment_name
  )

except Exception:

  search_query_string = f"name ='{experiment_name}'"
  experiment_id = mlflow.search_experiments(filter_string=search_query_string)[0].experiment_id
  print("Experiment already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the best trial's model to a run in the MLflow Experiment specified above and register the model

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="fs_automl_candidate_{}".format(str(datetime.datetime.now()).replace(" ", "_").strip()), experiment_id=experiment_id) as run:
                fs.log_model(
                              model=model,
                              artifact_path="automl-feature-store-model",
                              flavor=mlflow.sklearn,
                              training_set=training_set,
                              registered_model_name=f"{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor",
                              input_example=training_set.load_df().drop('price').toPandas()[:5],
                              signature=infer_signature(training_set.load_df().drop('price').toPandas(), training_set.load_df().toPandas()[["price"]])
                             )
mlflow.end_run()

with mlflow.start_run(run_name="automl_candidate_{}".format(str(datetime.datetime.now()).replace(" ", "_").strip()), experiment_id=experiment_id) as run:
              mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="automl-model",
                            input_example=training_set.load_df().drop('price').toPandas()[:5],
                            registered_model_name=f"{catalog}.{schema}.airbnb_sf_listings_price_predictor",
                            signature=infer_signature(training_set.load_df().drop('price').toPandas(), training_set.load_df().toPandas()[["price"]])
                            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate MSE & RMSE for the best AutoML model against the test dataset

# COMMAND ----------


import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

y_test = test_pdf['price']
X_test = test_pdf.drop(['price'], axis=1)

model = mlflow.pyfunc.load_model(model_uri)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
display(mse)
display(rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model alias

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

mv = client.search_model_versions(f"name='{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor'")
client.set_registered_model_alias(f"{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor", "candidate", mv[0].version)

mv = client.search_model_versions(f"name='{catalog}.{schema}.airbnb_sf_listings_price_predictor'")
client.set_registered_model_alias(f"{catalog}.{schema}.airbnb_sf_listings_price_predictor", "candidate", mv[0].version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve baseline model for evaluation with candidate model

# COMMAND ----------

from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

baseline_model = mlflow.pyfunc.load_model(f"models:/{catalog}.{schema}.airbnb_sf_listings_price_predictor@baseline")

y_pred_base = baseline_model.predict(X_test)
mse_base = mean_squared_error(y_test, y_pred_base)

eval_data = X_test
eval_data["label"] = y_test

thresholds = {
    "mean_squared_error": MetricThreshold(
        threshold=int(mse_base),
        min_absolute_change=1000,
        min_relative_change=0.05,
        greater_is_better=False,
    ),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use MLflow evaluate API to log an evaluation run and determine best model with test dataset

# COMMAND ----------

from mlflow.entities import ViewType

run = MlflowClient().search_runs(
    experiment_ids=experiment_id,
    filter_string="attributes.`run_name` LIKE 'sklearn_rf_baseline%'",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1
)[0]

baseline_model_uri = f'dbfs:/databricks/mlflow-tracking/{experiment_id}/{run.info.run_id}/artifacts/baseline-model'

with mlflow.start_run(run_name="model_validation", experiment_id=experiment_id):

    baseline_model_uri = baseline_model_uri
    candidate_model_uri = model_uri

    try:
      mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="regressor",
        evaluator_config={"log_model_explainability": False},
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri,
    )
    except ModelValidationFailedException:
      print("Candidate model is worse than baseline model.")
    else:
       print("Candidate model is better than baseline model.")
      
mlflow.end_run()

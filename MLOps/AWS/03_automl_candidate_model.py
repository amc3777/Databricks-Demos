# Databricks notebook source
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

# secret_access_key = dbutils.secrets.get(scope="scope", key="secret-key")
 
online_store_spec = AmazonDynamoDBSpec(
  region="us-west-2",
#   access_key_id="access-key_id",
#   secret_access_key=secret_access_key,
  table_name = "online_airbnb_sf_listings_features"
)

try:

  fs.publish_table(
    "andrewcooleycatalog.airbnb_data.airbnb_sf_listings_features", 
    online_store_spec
  )

except Exception:

  online_store = False
  print("Permissions on Amazon DynamoDB to publish to an online feature store are missing. Online feature store not created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create training set for AutoML regression experiment

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from sklearn.model_selection import train_test_split

from pyspark.sql.types import DoubleType

airbnb_df = spark.read.table("andrewcooleycatalog.airbnb_data.airbnb_sf_listings_indexed")
numeric_cols = [x.name for x in airbnb_df.schema.fields if x.dataType == DoubleType()]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)

pdf = numeric_features_df.toPandas()

train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=42)
train_lookup_df = spark.createDataFrame(train_pdf[["index", "price"]])
test_lookup_df = spark.createDataFrame(test_pdf[["index", "price"]])

feature_lookups = [
    FeatureLookup(
      table_name = 'andrewcooleycatalog.airbnb_data.airbnb_sf_listings_features',
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

summary = automl.regress(
  dataset=training_set.load_df(),
  target_col="price",
  # data_dir: Optional[str] = None,
  # exclude_columns: Optional[List[str]] = None,                      # <DBR> 10.3 ML and above
  # exclude_frameworks: Optional[List[str]] = None,                   # <DBR> 10.3 ML and above
  # experiment_dir: Optional[str] = None,                             # <DBR> 10.4 LTS ML and above
  experiment_name="automl_airbnb_price_prediction",
  # feature_store_lookups: Optional[List[Dict]] = None,               # <DBR> 11.3 LTS ML and above
  # imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None, # <DBR> 10.4 LTS ML and above
  primary_metric="rmse",
  # time_col: Optional[str] = None,
  timeout_minutes=60
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

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = "/Users/{}/airbnb_price_prediction".format(user)

try:

  experiment_id = mlflow.create_experiment(
      experiment_name
  )

except Exception:

  search_query_string = "name ='{}'".format(experiment_name)
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
                              registered_model_name="andrewcooleycatalog.airbnb_data.fs_airbnb_sf_listings_price_predictor",
                              input_example=training_set.load_df().drop('price').toPandas()[:5],
                              signature=infer_signature(training_set.load_df().drop('price').toPandas(), training_set.load_df().toPandas()[["price"]])
                             )
mlflow.end_run()

if not online_store:

  with mlflow.start_run(run_name="automl_candidate_{}".format(str(datetime.datetime.now()).replace(" ", "_").strip()), experiment_id=experiment_id) as run:
                mlflow.sklearn.log_model(
                              sk_model=model,
                              artifact_path="automl-model",
                              input_example=training_set.load_df().drop('price').toPandas()[:5],
                              registered_model_name="andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor",
                              signature=infer_signature(training_set.load_df().drop('price').toPandas(), training_set.load_df().toPandas()[["price"]])
                              )
mlflow.end_run()

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

mv = client.search_model_versions("name='andrewcooleycatalog.airbnb_data.fs_airbnb_sf_listings_price_predictor'")
client.set_registered_model_alias("andrewcooleycatalog.airbnb_data.fs_airbnb_sf_listings_price_predictor", "candidate", mv[0].version)

if not online_store:
  mv = client.search_model_versions("name='andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor'")
  client.set_registered_model_alias("andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor", "candidate", mv[0].version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieve baseline model for evaluation with candidate model

# COMMAND ----------

from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

baseline_model = mlflow.pyfunc.load_model("models:/andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor@baseline")

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

baseline_mv = client.get_model_version_by_alias("andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor", "baseline")
baseline_model_uri = client.get_model_version_download_uri("andrewcooleycatalog.airbnb_data.airbnb_sf_listings_price_predictor", int(baseline_mv.version))

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

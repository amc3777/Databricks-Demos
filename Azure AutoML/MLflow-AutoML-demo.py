# Databricks notebook source
# MAGIC %md
# MAGIC ### Load scikit learn dataset - California Housing

# COMMAND ----------

from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)

print(california_housing.DESCR)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display dataset as pandas dataframe

# COMMAND ----------

california_housing.frame.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read pandas dataframe into Spark dataframe for additional feature engineering steps (if needed)

# COMMAND ----------

spark_df = spark.createDataFrame(california_housing.frame)
display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create additional holdout dataset with sklearn train_test_split

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
train, test = train_test_split(california_housing.frame, test_size=0.2, random_state=2)
display(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use databricks-automl-runtime package to create AutoML experiment programmatically

# COMMAND ----------

from databricks import automl

summary = automl.regress(
  dataset=train,
  target_col="MedHouseVal",
  # data_dir: Optional[str] = None,
  # exclude_columns: Optional[List[str]] = None,                      # <DBR> 10.3 ML and above
  # exclude_frameworks: Optional[List[str]] = None,                   # <DBR> 10.3 ML and above
  # experiment_dir: Optional[str] = None,                             # <DBR> 10.4 LTS ML and above
  # feature_store_lookups: Optional[List[Dict]] = None,               # <DBR> 11.3 ML and above
  # imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None, # <DBR> 10.4 LTS ML and above
  # max_trials: Optional[int] = None,                                 # <DBR> 10.5 ML and below
  primary_metric="rmse",
  # time_col: Optional[str] = None,
  timeout_minutes=90
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Take best run model artifact and load for evaluation with final holdout dataset

# COMMAND ----------

import mlflow

y_test = test['MedHouseVal']
X_test = test.drop(['MedHouseVal'], axis=1)

eval_data = X_test
eval_data["label"] = y_test

with mlflow.start_run(run_name="automl_best_trial") as run:
  
    model = summary.best_trial.load_model()
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")

    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="regressor",
        evaluators=["default"],
    )
    
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine prediction correlation to build uncorrelated ensemble

# COMMAND ----------

import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

sgd_query = "attributes.run_name = 'sgd_regression'"

sgd_run = MlflowClient().search_runs(
  experiment_ids=summary.experiment.experiment_id,
  filter_string=sgd_query,
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.val_rmse"]
)[0]

xgb_query = "attributes.run_name = 'xgboost'"

xgb_run = MlflowClient().search_runs(
  experiment_ids=summary.experiment.experiment_id,
  filter_string=xgb_query,
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.val_rmse"]
)[0]

lgbm_query = "attributes.run_name = 'lightgbm'"

lgbm_run = MlflowClient().search_runs(
  experiment_ids=summary.experiment.experiment_id,
  filter_string=lgbm_query,
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.val_rmse"]
)[0]

rf_query = "attributes.run_name = 'random_forest_regressor'"

rf_run = MlflowClient().search_runs(
  experiment_ids=summary.experiment.experiment_id,
  filter_string=rf_query,
  run_view_type=ViewType.ALL,
  max_results=1,
  order_by=["metrics.val_rmse"]
)[0]

sgd_model = sgd_run.info.run_id
xgb_model = xgb_run.info.run_id
lgbm_model = lgbm_run.info.run_id
rf_model = rf_run.info.run_id

print(sgd_model)
print(xgb_model)
print(lgbm_model)
print(rf_model)

# COMMAND ----------

from sklearn.metrics import mean_squared_error
import numpy as np

sgd = "runs:/{}/model".format(sgd_model)
xgb = "runs:/{}/model".format(xgb_model)
lgbm = "runs:/{}/model".format(lgbm_model)
rf = "runs:/{}/model".format(rf_model)

models_list = []

models_list.extend([sgd,xgb,lgbm,rf])

def y_pred(model_uri):
  model = mlflow.pyfunc.load_model(model_uri)
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  print(rmse)
  return y_pred

pred_list = []

for model_uri in models_list:
  pred_list.append(y_pred(model_uri))
  
df_pred = pd.DataFrame(data=np.array(pred_list).T,
columns=['sgd', 'xgb','lgbm', 'rf'])

df_pred.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load best model with 2 least correlated, fit Voting Regressor, and evaluate with logging to MLflow

# COMMAND ----------

from sklearn.ensemble import VotingRegressor

y_train = train['MedHouseVal']
X_train = train.drop(['MedHouseVal'], axis=1)

mlflow.autolog(disable=True, log_models=False)

ereg = VotingRegressor(estimators=[('xgb', mlflow.sklearn.load_model(xgb)), ('sgd', mlflow.sklearn.load_model(sgd)), ('rf', mlflow.sklearn.load_model(rf))])
model = ereg.fit(X_train, y_train)

with mlflow.start_run(run_name="ensemble") as run:
  
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")
    
    result = mlflow.evaluate(      
        model_uri,
        eval_data,
        targets="label",
        model_type="regressor",
        evaluators="default",
        evaluator_config={"log_model_explainability": False}
    )
    
    ens_run = mlflow.active_run()
    
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run model validation against a metric threshold in MLflow

# COMMAND ----------

from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

thresholds = {
    "root_mean_squared_error": MetricThreshold(
        threshold=0.5,             # rmse should be <=0.5
        min_absolute_change=0.05,  # rmse should be at least 0.05 lower than baseline model rmse
        min_relative_change=0.05,  # rmse should be at least 5 percent lower than baseline model rmse
        higher_is_better=False
    ),
}

with mlflow.start_run(run_name="model_validation") as run:
    candidate_model_uri = "runs:/{}/model".format(ens_run.info.run_id)
    baseline_model_uri = "runs:/{}/model".format(summary.best_trial.mlflow_run_id)

    try:
      mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="regressor",
        evaluators="default",
        evaluator_config={"log_model_explainability": False},
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri,
    )
    except ModelValidationFailedException:
      print("Candidate model is worse than baseline model.")
      
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register model and version

# COMMAND ----------

model_name = "XGBoost Regressor"

model_uri = baseline_model_uri

registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition model version to staging

# COMMAND ----------

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=registered_model_version.version,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the model as a Spark UDF and generate predictions for the holdout dataset again

# COMMAND ----------

from pyspark.sql.functions import struct, col

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=baseline_model_uri)

test_spark = spark.createDataFrame(X_test)

display(test_spark.withColumn('MedHouseVal_predictions', loaded_model(struct(*map(col, test_spark.columns)))))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Spark UDF as a SQL function

# COMMAND ----------

spark.udf.register("medhouseval_predict", loaded_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Delta Lake temporary view

# COMMAND ----------

test_spark.createOrReplaceTempView("ca_housing_holdout")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the model as a SQL function

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, medhouseval_predict(struct(*)) as MedHouseVal_predictions from ca_housing_holdout;

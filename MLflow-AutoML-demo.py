# Databricks notebook source
import mlflow
mlflow.autolog(
    log_input_examples=False,
    log_model_signatures=False,
    log_models=False,
    disable=True,
    exclusive=True,
    disable_for_unsupported_versions=True,
    silent=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use BigQuery-Spark connector to read BigQuery table into Spark Dataframe

# COMMAND ----------

table = "bigquery-public-data.ml_datasets.penguins"
df = spark.read.format("bigquery").option("table",table).load()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create temporary view in Delta Lake format

# COMMAND ----------

df.createOrReplaceTempView("penguins_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use SQL to check for NULL in target column

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from penguins_raw where body_mass_g IS NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create new Delta temporary view with NULLs removed

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view penguins as 
# MAGIC select * from penguins_raw where body_mass_g IS NOT NULL;
# MAGIC select * from penguins where body_mass_g IS NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read back into Spark Dataframe and add ID column

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df = spark.read.table("penguins")
df = df.withColumn("penguin_id", monotonically_increasing_id())
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create feature store table with training dataset

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
import uuid
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

spark.sql(f"CREATE DATABASE IF NOT EXISTS penguin_db")

table_name = f"penguin_db_" + str(uuid.uuid4())[:6]

fs = feature_store.FeatureStoreClient()

features_df = df.drop('body_mass_g')

fs.create_table(
    name=table_name,
    primary_keys=["penguin_id"],
    df=features_df,
    description="penguin features"
)

print(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create additional holdout dataset with sklearn train_test_split and create training set from feature store look-up

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
pdf = df.toPandas()

train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=2)

train_lookup_df = spark.createDataFrame(train_pdf[["penguin_id", "body_mass_g"]])
test_lookup_df = spark.createDataFrame(test_pdf[["penguin_id", "body_mass_g"]])

training_set = fs.create_training_set(train_lookup_df, [FeatureLookup(table_name=table_name, lookup_key="penguin_id")], label="body_mass_g", exclude_columns="penguin_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use databricks-automl-runtime package to create AutoML experiment programmatically

# COMMAND ----------

from databricks import automl

summary = automl.regress(
  dataset=training_set.load_df(),
  target_col="body_mass_g",
  # data_dir: Optional[str] = None,
  # exclude_columns: Optional[List[str]] = None,                      # <DBR> 10.3 ML and above
  # exclude_frameworks: Optional[List[str]] = None,                   # <DBR> 10.3 ML and above
  # experiment_dir: Optional[str] = None,                             # <DBR> 10.4 LTS ML and above
  experiment_name="BQ-AutoML-reg_" + str(uuid.uuid4())[:6],
  # feature_store_lookups: Optional[List[Dict]] = None,               # <DBR> 11.3 LTS ML and above
  # imputers: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None, # <DBR> 10.4 LTS ML and above
  # max_trials: Optional[int] = None,                                 # <DBR> 10.5 ML and below
  primary_metric="rmse",
  # time_col: Optional[str] = None,
  timeout_minutes=30
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log best model packaged with feature store metadata

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

model_uri = summary.best_trial.model_path
model = mlflow.sklearn.load_model(model_uri)

with mlflow.start_run(
  run_name="best trial",
  experiment_id=summary.experiment.experiment_id,
  tags={"version": "v1", "priority": "P1"},
  description="best run from experiment") 
  as run:
                fs.log_model(
                              model=model,
                              artifact_path="penguin_model",
                              flavor=mlflow.sklearn,
                              training_set=training_set,
                              registered_model_name="penguin_model",
                             )
mlflow.end_run(run_name="best trial")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transition model version to staging

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
registered_model = client.transition_model_version_stage(
    name="penguin_model", version=4, stage="Staging"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load model as MLflow Python function flavor for framework-agnostic & light-weight serving

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

y_test = test_pdf['body_mass_g']
X_test = test_pdf.drop(['body_mass_g'], axis=1)

model = mlflow.pyfunc.load_model(model_uri)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse

# Feature Store score batch
# test_lookup_df = test_lookup_df.drop('body_mass_g') #drop label

# predictions_df = fs.score_batch('models:/penguin_model/Staging', test_lookup_df)
                                  
# display(predictions_df["penguin_id", "prediction"])

# COMMAND ----------

from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

with mlflow.start_run(
  run_name="baseline model",
  experiment_id=summary.experiment.experiment_id,
  tags={"version": "v1", "priority": "P1"},
  description="baseline model for validation") 
  as run:

  dummy_regr = DummyRegressor(strategy="mean")
  baseline_model = dummy_regr.fit(train_pdf.drop(['penguin_id', 'body_mass_g'], axis=1), train_pdf[['body_mass_g']])
  mlflow.sklearn.log_model(baseline_model, "baseline_model")
  

mlflow.end_run(run_name="baseline model")

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

with mlflow.start_run() as run:

  baseline_model_uri = mlflow.get_artifact_uri("baseline_model")

  candidate_model_uri = model_uri
  
  mlflow.evaluate(
        candidate_model_uri,
        eval_data,
        targets="label",
        model_type="regressor",
        validation_thresholds=thresholds,
        baseline_model=baseline_model_uri)

mlflow.end_run()

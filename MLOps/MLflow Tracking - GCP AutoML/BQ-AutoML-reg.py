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
# MAGIC ### Create table in Delta Lake format to Unity Catalog

# COMMAND ----------

table_name = "uc_demo_upgraded.penguins_demo.penguins_raw"
df.write.mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use SQL to check for NULL in target column

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from uc_demo_upgraded.penguins_demo.penguins_raw where body_mass_g IS NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create new Delta Lake table with NULLs removed

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table uc_demo_upgraded.penguins_demo.penguins as 
# MAGIC select * from uc_demo_upgraded.penguins_demo.penguins_raw where body_mass_g IS NOT NULL;
# MAGIC select * from uc_demo_upgraded.penguins_demo.penguins where body_mass_g IS NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read back into Spark Dataframe and add ID column

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

df = spark.read.table("uc_demo_upgraded.penguins_demo.penguins")
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

table_name = "uc_demo_upgraded.penguins_demo.penguins_features"

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

test_lookup_df.write.format("parquet").save("/Volumes/uc_demo_upgraded/penguins_demo/penguins_datasets/penguins_test_dataset")

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
  primary_metric="rmse",
  # time_col: Optional[str] = None,
  timeout_minutes=15
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log best model packaged with feature store metadata

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error
import numpy as np

mlflow.set_registry_uri("databricks-uc")

model_uri = summary.best_trial.model_path
model = mlflow.sklearn.load_model(model_uri)

with mlflow.start_run(
  run_name="best trial",
  description="best run from experiment"):
                fs.log_model(
                              model=model,
                              artifact_path="penguin_model",
                              flavor=mlflow.sklearn,
                              training_set=training_set,
                              registered_model_name="uc_demo_upgraded.penguins_demo.penguin_model",
                             )
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Give new model version an alias

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
mv = client.search_model_versions("name='uc_demo_upgraded.penguins_demo.penguin_model'")
registered_model = client.set_registered_model_alias("uc_demo_upgraded.penguins_demo.penguin_model", "candidate", mv[0].version)

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store score batch with look-up using data from a Unity Catalog Volume

# COMMAND ----------

test_lookup_df = spark.read.format("parquet").load("/Volumes/uc_demo_upgraded/penguins_demo/penguins_datasets/penguins_test_dataset").drop('body_mass_g') #drop label

predictions_df = fs.score_batch("models:/uc_demo_upgraded.penguins_demo.penguin_model@candidate", test_lookup_df)
                                  
display(predictions_df["penguin_id", "prediction"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a dummy baseline model for evaluation of candidate model

# COMMAND ----------

from sklearn.dummy import DummyRegressor
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="baseline model", description="baseline model for validation") as run:

  dummy_regr = DummyRegressor(strategy="mean")
  baseline_model = dummy_regr.fit(train_pdf.drop(['penguin_id', 'body_mass_g'], axis=1), train_pdf[['body_mass_g']])
  mlflow.sklearn.log_model(
    sk_model=baseline_model,
    artifact_path="baseline-model",
    registered_model_name="uc_demo_upgraded.penguins_demo.penguin_model",
    input_example=train_pdf.drop(['penguin_id', 'body_mass_g'], axis=1)[:5],
    signature=infer_signature(train_pdf.drop(['penguin_id', 'body_mass_g'], axis=1), train_pdf[['body_mass_g']]))
  
  baseline_model_run_id = run.info.run_id

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Give dummy model an alias

# COMMAND ----------

client = MlflowClient()
mv = client.search_model_versions("name='uc_demo_upgraded.penguins_demo.penguin_model'")
client.set_registered_model_alias("uc_demo_upgraded.penguins_demo.penguin_model", "baseline", mv[0].version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use MLflow evaluation to determine if candidate beats baseline

# COMMAND ----------

from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException

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

with mlflow.start_run(run_name="model_validation"):

    baseline_model_uri = f"runs:/{baseline_model_run_id}/baseline-model"
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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Declare candidate model as the champion with a new alias

# COMMAND ----------


client.set_registered_model_alias("uc_demo_upgraded.penguins_demo.penguin_model", "champion", 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load champion model as a Spark user-defined function and run batch inference against a Spark dataframe

# COMMAND ----------

from pyspark.sql.functions import struct, col

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

test_spark = spark.createDataFrame(X_test)

display(test_spark.withColumn('penguins_predictions', loaded_model(struct(*map(col, test_spark.columns)))))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the UDF as a SparkSQL function

# COMMAND ----------

spark.udf.register("penguins_predict", loaded_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a temporary view for the holdout test dataset

# COMMAND ----------

test_spark.createOrReplaceTempView("penguins_holdout")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run batch inference against the temporary view with the champion model

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, penguins_predict(struct(*)) as penguins_predictions from penguins_holdout;

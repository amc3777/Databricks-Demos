# Databricks notebook source
# MAGIC %md
# MAGIC ### Install Python client

# COMMAND ----------

# MAGIC %pip install "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.3.0-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define variables to set up inference table

# COMMAND ----------

CATALOG = "andrewcooleycatalog"
SCHEMA = "monitoring_example"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_inferencelogs"
BASELINE_TABLE = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_baseline"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction"
TIMESTAMP_COL = "timestamp"
MODEL_ID_COL = "model_id"
PREDICTION_COL = "prediction"
LABEL_COL = "price"
ID_COL = "ID"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load raw data into Spark dataframe and add ID column (optional)

# COMMAND ----------

from pyspark.sql import functions as F

raw_df = (spark.read.format("parquet")
  .load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/")
  .withColumn(ID_COL, F.expr("uuid()"))
)

display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split dataset into train, test (baseline), and inference segments

# COMMAND ----------

features_list = ["bedrooms", "neighbourhood_cleansed", "accommodates", "cancellation_policy", "beds", "host_is_superhost", "property_type", "minimum_nights", "bathrooms", "host_total_listings_count", "number_of_reviews", "review_scores_value", "review_scores_cleanliness"]

train_df, baseline_test_df, inference_df = raw_df.select(*features_list+[ID_COL, LABEL_COL]).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train random forest model with scikit-learn and register to Unity Catalog

# COMMAND ----------

import mlflow
import sklearn

from datetime import timedelta, datetime
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_registry_uri("databricks-uc")

X_train = train_df.drop(ID_COL, LABEL_COL).toPandas()
Y_train = train_df.select(LABEL_COL).toPandas().values.ravel()

categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]
one_hot_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("onehot", one_hot_pipeline, categorical_cols)], remainder="passthrough", sparse_threshold=0)

skrf_regressor = RandomForestRegressor(
  bootstrap=True,
  criterion="squared_error",
  max_depth=5,
  max_features=0.5,
  min_samples_leaf=0.1,
  min_samples_split=0.15,
  n_estimators=36,
  random_state=42,
)

model = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", skrf_regressor),
])

mlflow.sklearn.autolog(log_input_examples=True, silent=True, registered_model_name=MODEL_NAME)

with mlflow.start_run(run_name="random_forest_regressor") as mlflow_run:
  model.fit(X_train, Y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create baseline table for monitoring, with CDF enabled (recommended)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

query_string = f"name='{MODEL_NAME}'"
model_version = client.search_model_versions(query_string)[0].version
client.set_registered_model_alias(f"{MODEL_NAME}", "baseline", model_version)

model_uri = f"models:/{MODEL_NAME}@baseline"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
features = list(X_train.columns)

baseline_test_df_with_pred =(baseline_test_df
  .withColumn(PREDICTION_COL, loaded_model(*features))
  .withColumn(MODEL_ID_COL, F.lit(model_version))
)

(baseline_test_df_with_pred
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(BASELINE_TABLE)
)

display(baseline_test_df_with_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create simulated scoring data and store in an inference table to be monitored

# COMMAND ----------

test_labels_df = inference_df.select(ID_COL, LABEL_COL)
scoring_df1, scoring_df2 = inference_df.drop(LABEL_COL).randomSplit(weights=[0.5, 0.5], seed=42)

timestamp1 = (datetime.now() + timedelta(1)).timestamp()

pred_df1 = (scoring_df1
  .withColumn(TIMESTAMP_COL, F.lit(timestamp1).cast("timestamp")) 
  .withColumn(PREDICTION_COL, loaded_model(*features))
)

(pred_df1
  .withColumn(MODEL_ID_COL, F.lit(model_version))
  .withColumn(LABEL_COL, F.lit(None).cast("double"))
  .write.format("delta").mode("overwrite") 
  .option("mergeSchema",True) 
  .option("delta.enableChangeDataFeed", "true") 
  .saveAsTable(TABLE_NAME)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create monitor for inference logs

# COMMAND ----------

import databricks.lakehouse_monitoring as lm

PROBLEM_TYPE = "regression"
GRANULARITIES = ["1 day"]                       
SLICING_EXPRS = ["cancellation_policy", "accommodates > 2"]

print(f"Creating monitor for {TABLE_NAME}")

info = lm.create_monitor(
  table_name=TABLE_NAME,
  profile_type=lm.InferenceLog(
    granularities=GRANULARITIES,
    timestamp_col=TIMESTAMP_COL,
    model_id_col=MODEL_ID_COL,
    prediction_col=PREDICTION_COL,
    problem_type=PROBLEM_TYPE,
    label_col=LABEL_COL
  ),
  baseline_table_name=BASELINE_TABLE,
  slicing_exprs=SLICING_EXPRS,
  output_schema_name=f"{CATALOG}.{SCHEMA}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for monitor to be created...

# COMMAND ----------

import time

while info.status == lm.MonitorStatus.PENDING:
  info = lm.get_monitor(table_name=TABLE_NAME)
  time.sleep(10)
    
assert(info.status == lm.MonitorStatus.ACTIVE)

# COMMAND ----------

lm.get_monitor(table_name=TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Profile metrics table

# COMMAND ----------

profile_table = f"{TABLE_NAME}_profile_metrics"
display(spark.sql(f"SELECT * FROM {profile_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drift metrics table

# COMMAND ----------

drift_table = f"{TABLE_NAME}_drift_metrics"
display(spark.sql(f"SELECT * FROM {drift_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create data drift in 3 different features

# COMMAND ----------

remove_top_neighbourhood_list = ["South of Market", "Western Addition", "Downtown/Civic Center", "Bernal Heights", "Castro/Upper Market"]

scoring_df2_simulated = (scoring_df2
  .withColumn("neighbourhood_cleansed", 
    F.when(F.col("neighbourhood_cleansed").isin(remove_top_neighbourhood_list), "Mission")
    .otherwise(F.col("neighbourhood_cleansed"))
  )
  .withColumn("cancellation_policy", 
    F.when(F.col("cancellation_policy")=="flexible", "super flexible")
    .otherwise(F.col("cancellation_policy"))
  )
  .withColumn("accommodates", F.lit(1).cast("double"))
)
display(scoring_df2_simulated.select(["neighbourhood_cleansed", "cancellation_policy", "accommodates"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score model with drifted feature distributions

# COMMAND ----------

timestamp2 = (datetime.now() + timedelta(2)).timestamp()
pred_df2 = (scoring_df2_simulated
  .withColumn(TIMESTAMP_COL, F.lit(timestamp2).cast("timestamp")) 
  .withColumn(PREDICTION_COL, loaded_model(*features))
  .withColumn(MODEL_ID_COL, F.lit(model_version))
  .write.format("delta").mode("append")
  .saveAsTable(TABLE_NAME)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge later-arriving ground truth labels with inference table to capture model performance metrics

# COMMAND ----------

late_labels_view_name = f"airbnb_price_prediction_late_labels"
test_labels_df.createOrReplaceTempView(late_labels_view_name)

merge_info = spark.sql(
  f"""
  MERGE INTO {TABLE_NAME} AS i
  USING {late_labels_view_name} AS l
  ON i.{ID_COL} == l.{ID_COL}
  WHEN MATCHED THEN UPDATE SET i.{LABEL_COL} == l.{LABEL_COL}
  """
)
display(merge_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create custom metrics to monitor

# COMMAND ----------

from pyspark.sql import types as T
from math import exp

CUSTOM_METRICS = [
  lm.Metric(
    type="aggregate",
    name="log_avg",
    input_columns=["price"],
    definition="avg(log(abs(`{{input_column}}`)+1))",
    output_data_type=T.DoubleType()
  ),
  lm.Metric(
    type="derived",
    name="exp_log",
    input_columns=["price"],
    definition="exp(log_avg)",
    output_data_type=T.DoubleType()
  ),
  lm.Metric(
    type="drift",
    name="delta_exp",
    input_columns=["price"],
    definition="{{current_df}}.exp_log - {{base_df}}.exp_log",
    output_data_type=T.DoubleType()
  )
]

lm.update_monitor(
  table_name=TABLE_NAME,
  updated_params={"custom_metrics" : CUSTOM_METRICS}
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Refresh and inspect monitoring dashboard

# COMMAND ----------

run_info = lm.run_refresh(table_name=TABLE_NAME)
while run_info.state in (lm.RefreshState.PENDING, lm.RefreshState.RUNNING):
  run_info = lm.get_refresh(table_name=TABLE_NAME, refresh_id=run_info.refresh_id)
  time.sleep(30)

assert(run_info.state == lm.RefreshState.SUCCESS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a monitor alert

# COMMAND ----------

# MAGIC %md
# MAGIC [Instructions on creating an alert](https://docs.databricks.com/en/lakehouse-monitoring/monitor-alerts.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete monitor

# COMMAND ----------

lm.delete_monitor(table_name=TABLE_NAME)

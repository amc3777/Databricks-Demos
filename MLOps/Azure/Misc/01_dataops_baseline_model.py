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
# MAGIC ### Copy Parquet file to a Unity Catalog Volume

# COMMAND ----------

dbutils.fs.cp("/Volumes/odl_instructor_1127516_catalog/odl_instructor_1127516_schema/odl_instructor_1127516_managedvolume/sf-listings-2019-03-06-clean.parquet", f"dbfs:/Volumes/{catalog}/{schema}/{volume}/sf-listings-2019-03-06-clean.parquet", recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Delta Lake table (managed by Unity Catalog) with a DEEP CLONE of a Parquet file

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ${catalog}.${schema}.airbnb_sf_listings_raw DEEP CLONE parquet.`/Volumes/${catalog}/${schema}/${volume}/sf-listings-2019-03-06-clean.parquet`;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Add an ID column as a look-up key for the dataset

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_raw").coalesce(1).withColumn("index", monotonically_increasing_id()).write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.airbnb_sf_listings_indexed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop non-numeric fields and label to create model input features

# COMMAND ----------

from pyspark.sql.types import DoubleType

airbnb_df = spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_indexed")
numeric_cols = [x.name for x in airbnb_df.schema.fields if (x.dataType == DoubleType()) and (x.name != "price")]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)
numeric_features_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.airbnb_sf_listings_numeric_only")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Import Databricks Feature Store Python API library and instantiate client

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Create or overwrite feature store table

# COMMAND ----------

try:

  fs.write_table(
  name=f"{catalog}.{schema}.airbnb_sf_listings_features",
  df=spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_numeric_only"),
  mode="overwrite"
)
  
except ValueError:

  print("Feature table does not exist. Creating new one.")

  fs.create_table(
    name=f"{catalog}.{schema}.airbnb_sf_listings_features",
    primary_keys=["index"],
    df=spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_numeric_only"),
    schema=spark.read.table(f"{catalog}.{schema}.airbnb_sf_listings_numeric_only").schema,
    description="Numeric features of airbnb data"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create feature look-ups on feature table and MLflow dataset

# COMMAND ----------

import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from databricks.feature_store import feature_table, FeatureLookup

feature_lookups = [
    FeatureLookup(
      table_name = f'{catalog}.{schema}.airbnb_sf_listings_features',
      feature_names = numeric_features_df.columns[1:],
      lookup_key = 'index',
    )]

training_set = fs.create_training_set(
  df=airbnb_df.select(["index", "price"]),
  feature_lookups = feature_lookups,
  label = 'price',
  exclude_columns = ['index']
)

training_pd = training_set.load_df().toPandas()

dataset: PandasDataset = mlflow.data.from_pandas(
  df = training_pd, 
  source = f"{catalog}.{schema}.airbnb_sf_listings_features",
  targets = "price",
  name = "airbnb_sf_listings_train")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disable Databricks Autologging to MLflow Tracking

# COMMAND ----------

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create an empty MLflow Experiment to track training runs with

# COMMAND ----------

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
# MAGIC ### Train a scikit-learn random forest regressor with logged features and a model registered in Unity Catalog

# COMMAND ----------

import datetime
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_registry_uri("databricks-uc")

y = training_pd.loc[:, 'price']
X = training_pd.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test, training_set, fs):

    with mlflow.start_run(run_name="fs_sklearn_rf_baseline_{}".format(str(datetime.datetime.now()).replace(" ", "_").strip()), experiment_id=experiment_id) as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_input(dataset, context="training")

        for param, value in rf.get_params(deep=True).items():
            mlflow.log_param(param, value)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mlflow.log_metric("test_rmse", rmse)
        
        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

    mlflow.end_run()

    with mlflow.start_run(run_name="sklearn_rf_baseline_{}".format(str(datetime.datetime.now()).replace(" ", "_").strip()), experiment_id=experiment_id) as run:

          rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
          rf.fit(X_train, y_train)
          y_pred = rf.predict(X_test)

          mlflow.log_input(dataset, context="training")

          for param, value in rf.get_params(deep=True).items():
              mlflow.log_param(param, value)

          mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
          mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
          mse = mean_squared_error(y_test, y_pred)
          rmse = np.sqrt(mse)
          mlflow.log_metric("test_rmse", rmse)

          mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="baseline-model",
            input_example=X_train[:5],
            registered_model_name=f"{catalog}.{schema}.airbnb_sf_listings_price_predictor",
            signature=infer_signature(X_train, y_train)

          )

    mlflow.end_run()

train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create model alias

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

mv = client.search_model_versions(f"name='{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor'")
client.set_registered_model_alias(f"{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor", "baseline", mv[0].version)

mv = client.search_model_versions(f"name='{catalog}.{schema}.airbnb_sf_listings_price_predictor'")
client.set_registered_model_alias(f"{catalog}.{schema}.airbnb_sf_listings_price_predictor", "baseline", mv[0].version)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run local batch inference on test dataset with feature store look-up

# COMMAND ----------

X = airbnb_df.select(["index"]).toPandas()
y = airbnb_df.select(["price"]).toPandas()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_lookup_df = spark.createDataFrame(X_test)

predictions_df = fs.score_batch(f"models:/{catalog}.{schema}.fs_airbnb_sf_listings_price_predictor@baseline", test_lookup_df)

display(predictions_df.join(airbnb_df, predictions_df.index == airbnb_df.index).select(predictions_df.prediction, airbnb_df.price))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run local batch inference on test dataset with SQL

# COMMAND ----------

from pyspark.sql.functions import struct, col

spark_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{catalog}.{schema}.airbnb_sf_listings_price_predictor@baseline")

numeric_cols = [x.name for x in airbnb_df.schema.fields if x.dataType == DoubleType()]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)

pdf = numeric_features_df.toPandas()

train_pdf, test_pdf = train_test_split(pdf, test_size=0.2, random_state=42)

y_test = test_pdf['price']
X_test = test_pdf.drop(['price'], axis=1)

test_spark = spark.createDataFrame(X_test)

spark.udf.register("airbnb_price_predict", spark_model)

test_spark.createOrReplaceTempView("airbnb_sf_listings_holdout")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute SQL statement

# COMMAND ----------

# MAGIC %sql
# MAGIC select *, airbnb_price_predict(struct(*)) as prediction from airbnb_sf_listings_holdout;

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow + SHAP for Explainable AI

# COMMAND ----------

import shap
from mlflow.artifacts import download_artifacts

model = mlflow.pyfunc.load_model(f"models:/{catalog}.{schema}.airbnb_sf_listings_price_predictor@baseline")

baseline_explainer = shap.Explainer(model.predict, X_test, algorithm="permutation")

with mlflow.start_run(run_name="baseline_log_model_explainer", experiment_id=experiment_id) as run:
    mlflow.shap.log_explainer(baseline_explainer, artifact_path="baseline_shap_explainer")

test_explainer = mlflow.shap.load_explainer(f"runs:/{run.info.run_id}/baseline_shap_explainer")

shap_values = test_explainer(X_test[:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary plot of explanations

# COMMAND ----------

shap.summary_plot(shap_values, X_test[:100], plot_type="violin", plot_size=0.25)

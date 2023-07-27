# Databricks notebook source
dbutils.fs.cp("dbfs:/mnt/dbacademy-datasets/scalable-machine-learning-with-apache-spark/v02/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet", "/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume/sf-listings-2019-03-06-clean.parquet", recurse=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_raw DEEP CLONE parquet.`/Volumes/andrewcooleycatalog/andrewcooleyschema/managedvolume/sf-listings-2019-03-06-clean.parquet`;

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

spark.read.table("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_raw").coalesce(1).withColumn("index", monotonically_increasing_id()).write.mode("overwrite").saveAsTable("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_indexed")

# COMMAND ----------

from pyspark.sql.types import DoubleType

airbnb_df = spark.read.table("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_indexed")
numeric_cols = [x.name for x in airbnb_df.schema.fields if (x.dataType == DoubleType()) and (x.name != "price")]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)
numeric_features_df.write.mode("overwrite").saveAsTable("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_numeric_features")

# COMMAND ----------

from databricks import feature_store

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

fs.create_table(
    name="andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_features",
    primary_keys=["index"],
    df=spark.read.table("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_numeric_features"),
    schema=spark.read.table("andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_numeric_features").schema,
    description="Numeric features of airbnb data"
)

# COMMAND ----------

import mlflow.data
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset
from databricks.feature_store import feature_table, FeatureLookup

feature_lookups = [
    FeatureLookup(
      table_name = 'andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_features',
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
  source = "andrewcooleycatalog.andrewcooleyschema.airbnb_sf_listings_features",
  targets = "price",
  name = "airbnb_sf_listings_train")

# COMMAND ----------

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

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_registry_uri("databricks-uc")

y = training_pd.loc[:, 'price']
X = training_pd.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test, training_set, fs):

    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_input(dataset, context="training")

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name="andrewcooleycatalog.andrewcooleyschema.feature_store_airbnb_sf_listings",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

train_model(X_train, X_test, y_train, y_test, training_set, fs)

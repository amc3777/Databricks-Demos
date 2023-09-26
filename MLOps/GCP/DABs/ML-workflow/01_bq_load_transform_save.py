# Databricks notebook source

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TRAINING = f"{CATALOG}.{SCHEMA}.credit_card_default_training"
TEST = f"{CATALOG}.{SCHEMA}.credit_card_default_test"
INFERENCE = f"{CATALOG}.{SCHEMA}.credit_card_default_inference"

table = "bigquery-public-data.ml_datasets.credit_card_default"
df = spark.read.format("bigquery").option("table",table).load().drop('predicted_default_payment_next_month')

raw_df = df.toPandas()
bins = [20, 30, 40, 50, 60, 70, 80]
labels = [1,2,3,4,5,6]
raw_df['age'] = pd.cut(raw_df['age'], bins=bins, labels=labels)
raw_df[["age"]] = raw_df[["age"]].astype(object)
raw_df[["pay_5", "pay_6"]] = raw_df[["pay_5", "pay_6"]].astype(float)
raw_df[["default_payment_next_month"]] = raw_df[["default_payment_next_month"]].astype(int)

non_numeric = raw_df.select_dtypes(exclude=np.number).columns.to_list()

raw_df = pd.get_dummies(raw_df, columns=non_numeric, dtype=float)
raw_df.columns = raw_df.columns.str.replace(' ', '_')

raw_df = spark.createDataFrame(raw_df)

features_list = raw_df.columns

train_df, test_df, inference_df = raw_df.select(*features_list).randomSplit(weights=[0.6, 0.2, 0.2], seed=42)

(train_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(TRAINING)
)

(test_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(TEST)
)

(inference_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(INFERENCE)
)

dbutils.jobs.taskValues.set(key = 'catalog', value = CATALOG)
dbutils.jobs.taskValues.set(key = 'schema', value = SCHEMA)
dbutils.jobs.taskValues.set(key = 'training_dataset', value = TRAINING)
dbutils.jobs.taskValues.set(key = 'test_dataset', value = TEST)
dbutils.jobs.taskValues.set(key = 'inference_dataset', value = INFERENCE)
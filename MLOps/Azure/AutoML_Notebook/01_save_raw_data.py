# Databricks notebook source
from pyspark.sql import functions as F

ID_COL = "ID"
CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"
VOLUME = "managedvolume"
PATH = '/Volumes/{}/{}/{}/sf-airbnb-clean.parquet/'.format(CATALOG, SCHEMA, VOLUME)

raw_df = (spark.read.format("parquet")
  .load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/")
  .withColumn(ID_COL, F.expr("uuid()"))
)

raw_df.write.format("parquet").mode("overwrite").save(PATH)

dbutils.jobs.taskValues.set(key = 'volumes_path', value = PATH)

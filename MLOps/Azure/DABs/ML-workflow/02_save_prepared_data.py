from pyspark.sql import functions as F
import pandas as pd
import numpy as np

volumes_path = dbutils.jobs.taskValues.get(taskKey = "01-save-raw-data-task", key = "volumes_path")

CATALOG = "andrewcooleycatalog"
SCHEMA = "andrewcooleyschema"
TRAINING = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_training"
TEST = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_test"
INFERENCE = f"{CATALOG}.{SCHEMA}.airbnb_price_prediction_inference"

raw_df = (spark.read.format("parquet")
  .load(volumes_path))

raw_df = raw_df.toPandas()

non_numeric = raw_df.select_dtypes(exclude=[np.number]).columns.to_list()
non_numeric.pop(-1)

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
  .option("delta.enableChangeDataFeed", "true")
  .saveAsTable(TEST)
)

(inference_df
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema",True)
  .saveAsTable(INFERENCE)
)

dbutils.jobs.taskValues.set(key = 'training_path', value = TRAINING)
dbutils.jobs.taskValues.set(key = 'test_path', value = TEST)
dbutils.jobs.taskValues.set(key = 'inference_path', value = INFERENCE)